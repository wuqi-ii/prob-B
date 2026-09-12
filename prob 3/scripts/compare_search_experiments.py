"""Reproducible paired, offline-only experiment; refuses to overwrite logs."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cumcm_b3.backend_offline import OfflineSimulator, generate_case
from cumcm_b3.config import StrategyConfig
from cumcm_b3.experimental_strategy import make_strategy
from cumcm_b3.metrics import write_json, append_jsonl, build_result

VARIANTS = {'baseline': (False, False), 'adaptive': (True, False),
            'joint': (False, True), 'combined': (True, True)}
OVERRIDES = {
    'near_polish': {'shared_observations':'opportunistic','verify_near_fraction':.1,'route_polish':True},
    'route_only': {'verify_near_fraction':.1,'route_multistart':True},
    'forward_verify': {'shared_observations':'opportunistic','verify_near_fraction':.1,'adaptive_verify':True},
    'forward_route': {'shared_observations':'opportunistic','verify_near_fraction':.1,'route_multistart':True},
    'forward_both': {'shared_observations':'opportunistic','verify_near_fraction':.1,'adaptive_verify':True,'route_multistart':True},
    'shared_f005': {'shared_observations':'opportunistic','verify_near_fraction':.05},
    'shared': {'shared_observations': 'opportunistic'},
    'batch': {'shared_observations': 'batch'},
    'polish': {'route_polish': True},
    'shared_polish': {'shared_observations': 'opportunistic', 'route_polish': True},
    'shared_f02': {'shared_observations': 'opportunistic', 'verify_near_fraction': 0.2},
    'shared_f01': {'shared_observations': 'opportunistic', 'verify_near_fraction': 0.1},
    'batch_f02': {'shared_observations': 'batch', 'verify_near_fraction': 0.2},
    'baseline_f02': {'verify_near_fraction': 0.2},
    'baseline_f01': {'verify_near_fraction': 0.1},
}

BASE_REFINED = dict(shared_observations='opportunistic',verify_near_fraction=.1,route_multistart=True)
OPTIMIZED_THREEFIX = dict(**BASE_REFINED, recovery_verify=True,
    bearing_rounding_slack_deg=.005001, stop_at_source_upper_bound=True,
    scan_relocation=True)
OVERRIDES['threefix'] = OPTIMIZED_THREEFIX
OVERRIDES['dynamic_hybrid'] = dict(**OPTIMIZED_THREEFIX, dynamic_second_station=True)
OVERRIDES['dynamic_reuse'] = dict(**OPTIMIZED_THREEFIX, dynamic_second_station=True,
                                  planned_stop_reuse=True)
OVERRIDES['lean_scan'] = dict(**OPTIMIZED_THREEFIX,
                              scan_all_channels_at_scan_points=False)
OVERRIDES['known16_stop'] = dict(**OPTIMIZED_THREEFIX,
                                 stop_search_when_all_sources_known=True)
OVERRIDES['no_postcheck'] = dict(**OPTIMIZED_THREEFIX, post_clear_verify=False)
OVERRIDES['lean_actions'] = dict(**OPTIMIZED_THREEFIX,
    scan_all_channels_at_scan_points=False,
    stop_search_when_all_sources_known=True,
    post_clear_verify=False)
LEAN_ACTIONS = OVERRIDES['lean_actions']
OVERRIDES['lean_no_opportunistic'] = {
    **LEAN_ACTIONS, 'shared_observations':'off', 'opportunistic_scan':False}
OVERRIDES['lean_legacy_opportunistic'] = {
    **LEAN_ACTIONS, 'shared_observations':'off', 'opportunistic_scan':True}
OVERRIDES['lean_near005'] = {**LEAN_ACTIONS, 'verify_near_fraction':.05}
OVERRIDES['lean_near020'] = {**LEAN_ACTIONS, 'verify_near_fraction':.2}
OVERRIDES['lean_approach095'] = {**LEAN_ACTIONS, 'approach_step_ratio':.95}
OVERRIDES['lean_approach099'] = {**LEAN_ACTIONS, 'approach_step_ratio':.99}
OVERRIDES['lean_safe'] = {**LEAN_ACTIONS, 'safe_clear_point':True}
OVERRIDES['lean_step'] = {**LEAN_ACTIONS, 'step_replan':True}
OVERRIDES['lean_safe_step'] = {
    **LEAN_ACTIONS, 'safe_clear_point':True, 'step_replan':True}
OVERRIDES['lean_timed'] = {**LEAN_ACTIONS, 'route_time_score':True}
OVERRIDES['lean_timed_safe'] = {
    **LEAN_ACTIONS, 'route_time_score':True, 'safe_clear_point':True}
for _name,_changes in {
    'timed': dict(route_time_score=True),
    'step': dict(step_replan=True),
    'safe': dict(safe_clear_point=True),
    'timed_step': dict(route_time_score=True,step_replan=True),
    'refined_all': dict(route_time_score=True,step_replan=True,safe_clear_point=True),
    'safe_step': dict(step_replan=True,safe_clear_point=True),
    'route_adaptive': dict(adaptive_search=True),
    'relocated': dict(scan_relocation=True),
    'relocated_safe_step': dict(scan_relocation=True,safe_clear_point=True,step_replan=True),
    'range_bound': dict(received_range_bound=True),
    'range_safe_step': dict(received_range_bound=True,safe_clear_point=True,step_replan=True),
    'approach095': dict(approach_step_ratio=.95),
    'approach099': dict(approach_step_ratio=.99),
    'approach070': dict(approach_step_ratio=.7),
    'near005': dict(verify_near_fraction=.05),
    'near020': dict(verify_near_fraction=.2),
    'recovered': dict(recovery_verify=True),
    'recovered_safe_step': dict(recovery_verify=True,safe_clear_point=True,step_replan=True),
    'recovered_relocated': dict(recovery_verify=True,safe_clear_point=True,step_replan=True,scan_relocation=True),
    'hardened': dict(recovery_verify=True,bearing_rounding_slack_deg=.005001,stop_at_source_upper_bound=True),
    'robust': dict(recovery_verify=True,bearing_rounding_slack_deg=.005001),
    'hardened_relocate_only': dict(recovery_verify=True,bearing_rounding_slack_deg=.005001,stop_at_source_upper_bound=True,scan_relocation=True),
    'hardened_relocated': dict(recovery_verify=True,bearing_rounding_slack_deg=.005001,stop_at_source_upper_bound=True,safe_clear_point=True,step_replan=True,scan_relocation=True),
}.items():
    OVERRIDES[_name]={**BASE_REFINED,**_changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--start-seed', type=int, default=21000)
    parser.add_argument('--cases', type=int, default=30)
    parser.add_argument('--minimum-reception', action='store_true',
                        help='Set all receiver radii to the worst-case 1000 m.')
    parser.add_argument('--n-sources',type=int,default=0,help='0=random 10..16; otherwise fixed source count')
    parser.add_argument('--variants', nargs='+', choices=[*VARIANTS, *OVERRIDES], default=list(VARIANTS))
    parser.add_argument('--resume', action='store_true', help='Resume identical code/config, keeping completed cases.')
    args = parser.parse_args()
    out = ROOT/args.outdir
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT/'src'/'cumcm_b3').glob('*.py')}
    if args.resume:
        manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        old_args = dict(manifest['arguments'])
        old_args.pop('resume', None)
        new_args = vars(args).copy()
        new_args.pop('resume')
        if old_args != new_args or manifest['source_sha256'] != hashes:
            raise ValueError('Resume refused: source or arguments changed.')
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/'manifest.json', {'arguments': vars(args), 'source_sha256': hashes})
        write_json(out/'source_snapshot.json', {str(p.relative_to(ROOT)): p.read_text(encoding='utf-8')
                                               for p in (ROOT/'src'/'cumcm_b3').glob('*.py')})
    rows = {name: [] for name in args.variants}
    for name in rows:
        if name in VARIANTS:
            adaptive, joint = VARIANTS[name]
            changes = dict(adaptive_search=adaptive, joint_service=joint)
        else:
            changes = OVERRIDES[name]
        cfg = replace(StrategyConfig(), **changes)
        cfg.validate()
        config_path = out/name/'effective_config.json'
        if not config_path.exists():
            write_json(config_path, cfg.to_dict())
        row_path = out/name/'cases.jsonl'
        if args.resume and row_path.exists():
            rows[name] = [json.loads(line) for line in row_path.read_text(encoding='utf-8').splitlines()]
            for i, row in enumerate(rows[name]):
                assert row['seed'] == args.start_seed+i
                assert (out/name/f'case_{row["seed"]}.json').exists(), 'Incomplete case needs inspection'
            print(name, 'resume from', len(rows[name]), flush=True)
        for i in range(len(rows[name]), args.cases):
            seed = args.start_seed+i
            case = generate_case(seed,n_sources=args.n_sources or None)
            if args.minimum_reception:
                for source in case.sources:
                    source.receiver_radius_m = 1000.0
            sim = OfflineSimulator(case)
            strategy = make_strategy(sim, cfg)
            started = time.perf_counter()
            try:
                stats = strategy.run()
            except Exception:
                # Retain the failed attempt separately; do not skip its seed.
                attempt=1
                failure=out/name/f'failure_{seed}_{attempt}.json'
                while failure.exists():
                    attempt+=1
                    failure=out/name/f'failure_{seed}_{attempt}.json'
                write_json(failure,{'seed':seed,'traceback':traceback.format_exc(),
                    'python':sys.version,'executable':sys.executable,
                    'events':strategy.stats.events,'trace':sim.trace,
                    'truth':[vars(src) for src in case.sources]})
                raise
            result = build_result(str(seed), seed, case.total, stats).as_row()
            result['wall_s'] = time.perf_counter()-started
            result['events_count'] = dict(Counter(e['event'] for e in stats.events))
            result['status'] = strategy.status()
            measures = [e for e in sim.trace if e['kind'] == 'measure']
            result['measurement_s'] = 5*len(measures)
            result['switch_s'] = sum(e['switch_cost_s'] for e in measures)
            result['clear_s'] = 5*stats.clear_success+3*stats.clear_miss
            result['move_s'] = stats.virtual_time_s-result['measurement_s']-result['switch_s']-result['clear_s']
            rows[name].append(result)
            write_json(out/name/f'case_{seed}.json', {
                'result': result, 'events': stats.events, 'trace': sim.trace,
                'truth': [vars(src) for src in case.sources]})
            append_jsonl(out/name/'cases.jsonl', result)
            if (i+1) % 10 == 0:
                print(name, i+1, '/', args.cases, 'mean_s',
                      round(statistics.mean(r['total_time_s'] for r in rows[name]), 2), flush=True)
    report = {}
    for name, data in rows.items():
        report[name] = {k: statistics.mean(r[k] for r in data) for k in
                        ['total_time_s', 'requests', 'scan_points_visited', 'move_s',
                         'measurement_s', 'switch_s', 'wall_s']}
        report[name]['perfect_cases'] = sum(r['cleared_ratio'] == 1 and r['stop_reason'] == 'all_done' for r in data)
        report[name]['total_sources'] = sum(r['total_sources'] for r in data)
        report[name]['total_cleared'] = sum(r['cleared_count'] for r in data)
        if 'baseline' in rows and name != 'baseline':
            deltas = [a['total_time_s']-b['total_time_s'] for a, b in zip(data, rows['baseline'])]
            mean = statistics.mean(deltas)
            se = statistics.stdev(deltas)/(len(deltas)**0.5) if len(deltas)>1 else 0
            report[name]['paired'] = dict(mean_delta_s=mean, approx_95_ci_s=[mean-1.96*se, mean+1.96*se],
                                         faster=sum(d < 0 for d in deltas), slower=sum(d > 0 for d in deltas),
                                         change_percent=100*mean/statistics.mean(r['total_time_s'] for r in rows['baseline']))
    write_json(out/'comparison.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if all(r['perfect_cases'] == args.cases for r in report.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
