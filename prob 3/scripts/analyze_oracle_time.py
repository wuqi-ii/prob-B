"""Offline diagnostic only: exact point-tour optimum and bounds for 20 m clears.

Truth is used only here, never supplied to DogStrategy. The unknown-target
search problem is not solved by this oracle. Requires NumPy for subset DP.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cumcm_b3.backend_offline import OfflineSimulator, generate_case
from cumcm_b3.config import StrategyConfig, DOG_SPEED_MPS, CLEAR_RADIUS_M
from cumcm_b3.metrics import write_json, append_jsonl
from cumcm_b3.strategy import DogStrategy


def exact_open_tour(points):
    """Held-Karp: start at origin, visit each point, free endpoint (no return).

    dp[mask,j] = min_k dp[mask without j,k] + distance(k,j).
    Cardinality batches vectorize independent subsets. O(n^2 2^n) time.
    """
    n = len(points)
    if n == 0:
        return 0.0, []
    pts = np.asarray(points, dtype=float)
    distances = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    dp = np.full((1 << n, n), np.inf)
    for j in range(n):
        dp[1 << j, j] = np.linalg.norm(pts[j])
    masks = np.arange(1 << n, dtype=np.int64)
    sizes = np.fromiter((int(m).bit_count() for m in masks), dtype=np.int8)
    for size in range(2, n + 1):
        layer = masks[sizes == size]
        for j in range(n):
            selected = layer[(layer & (1 << j)) != 0]
            prev = selected ^ (1 << j)
            dp[selected, j] = np.min(dp[prev] + distances[:, j], axis=1)
    mask = (1 << n) - 1
    j = int(np.argmin(dp[mask]))
    optimum = float(dp[mask, j])
    backwards = []
    while mask:
        backwards.append(j)
        mask ^= 1 << j
        if mask:
            j = int(np.argmin(dp[mask] + distances[:, j]))
    order = list(reversed(backwards))
    pos = (0.0, 0.0)
    length = 0.0
    for j in order:
        length += math.dist(pos, points[j])
        pos = points[j]
    assert len(set(order)) == n and math.isclose(length, optimum, abs_tol=1e-7)
    return optimum, order


def self_check():
    for n in range(1, 8):
        pts = [(math.cos(j * 1.37) * (j + 1), math.sin(j * 1.37) * (j + 1))
               for j in range(n)]
        exact, _ = exact_open_tour(pts)
        brute = min(sum(math.dist(a, b) for a, b in zip([(0, 0), *p], p))
                    for p in itertools.permutations(pts))
        assert math.isclose(exact, brute, abs_tol=1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='outputs/scheduler_ab_200/route_insert')
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    out = ROOT / args.outdir
    out.mkdir(parents=True, exist_ok=False)
    self_check()
    reference = ROOT / args.reference
    summary = json.loads((reference / 'summary.json').read_text(encoding='utf-8'))
    details = json.loads((reference / 'details.json').read_text(encoding='utf-8'))
    raw_cfg = summary['effective_config'].copy()
    if raw_cfg.get('virtual_time_budget_s') is None:
        raw_cfg['virtual_time_budget_s'] = float('inf')
    cfg = replace(StrategyConfig(), **raw_cfg)
    cfg.validate()
    write_json(out / 'effective_config.json', cfg.to_dict())
    records = []
    for index, (old, detail) in enumerate(zip(summary['results'], details)):
        seed = old['seed']
        case = generate_case(seed)
        for source, saved in zip(case.sources, detail['truth']):
            assert source.channel == saved['channel']
            assert round(source.x, 2) == saved['x'] and round(source.y, 2) == saved['y']
        points = [s.position for s in case.sources]
        length, order = exact_open_tour(points)
        n = case.total
        # For any feasible disc-visiting route of length L, replacing each
        # clearance point by its source centre increases the first edge by
        # at most r and each of the n-1 later edges by at most 2r.
        # Thus L >= exact_point_tour - (2n-1)r. This is a lower bound, not
        # the exact disc-tour optimum. A centre tour supplies a feasible UB.
        lower = max(0.0, length - (2*n - 1)*CLEAR_RADIUS_M) / DOG_SPEED_MPS + 5*n
        upper = length / DOG_SPEED_MPS + 5*n
        oracle = OfflineSimulator(generate_case(seed))
        oracle.enter()
        for j in order:
            src = oracle.case.sources[j]
            assert oracle.clear(src.x, src.y, src.channel)['clear_result'] == 'success'
        oracle.exit()
        assert math.isclose(oracle.virtual_time_s, upper, abs_tol=1e-7)

        backend = OfflineSimulator(case)
        stats = DogStrategy(backend, cfg).run()
        assert stats.cleared_count == n
        assert math.isclose(stats.virtual_time_s, old['total_time_s'], abs_tol=1e-6)
        measures = [e for e in backend.trace if e['kind'] == 'measure']
        clears = [e for e in backend.trace if e['kind'] == 'clear']
        switch_s = sum(e['switch_cost_s'] for e in measures)
        measure_s = 5 * len(measures)
        clear_s = sum(5 if e['result'] == 'success' else 3 for e in clears)
        move_s = stats.virtual_time_s - switch_s - measure_s - clear_s
        row = dict(seed=seed, sources=n, current_time_s=stats.virtual_time_s,
                   move_s=move_s, measure_s=measure_s, switch_s=switch_s,
                   clear_s=clear_s, measure_count=len(measures),
                   measure_results=dict(Counter(e['measure_result'] for e in measures)),
                   exact_centre_path_m=length, centre_order_channels=[case.sources[j].channel for j in order],
                   oracle_lower_s=lower, oracle_upper_s=upper,
                   oracle_clearance_radius_m=CLEAR_RADIUS_M)
        append_jsonl(out / 'cases.jsonl', row)
        records.append(row)
        if (index+1) % 10 == 0:
            print(f'{index+1}/{len(summary["results"])} cases; seed={seed}; '
                  f'current={stats.virtual_time_s:.1f}s; oracle=[{lower:.1f},{upper:.1f}]s', flush=True)
    keys = ['current_time_s', 'move_s', 'measure_s', 'switch_s', 'clear_s',
            'measure_count', 'exact_centre_path_m', 'oracle_lower_s', 'oracle_upper_s']
    means = {k: statistics.mean(r[k] for r in records) for k in keys}
    result = dict(cases=len(records), total_sources=sum(r['sources'] for r in records),
                  means=means,
                  max_possible_saving_fraction=1-means['oracle_lower_s']/means['current_time_s'],
                  comparison='Same 200 saved cases and effective config; current times reproduced.',
                  caveat='Oracle has full coordinates and channel identities; search cost omitted. '
                         'Bounds enclose the full-information optimum, not the unknown-target optimum.',
                  proof='L_centres - (2n-1)*20 <= L_discs <= L_centres; '
                        'time = length/5 + 5n; start at origin, endpoint free.',
                  verification='Subset DP checked against exhaustive permutations for n=1..7; '
                               'every recovered oracle route replayed successfully in OfflineSimulator.')
    write_json(out / 'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
