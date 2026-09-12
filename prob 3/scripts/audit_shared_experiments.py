"""Read-only checks of case evidence, with a saved audit summary."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cumcm_b3.config import StrategyConfig
from cumcm_b3.metrics import write_json


def main():
    report = {}
    for label in ('pilot', 'validation', 'min1000'):
        folder = ROOT / 'logs' / 'experiments' / f'shared_{label}_20260912'
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        snapshot = json.loads((folder / 'source_snapshot.json').read_text(encoding='utf-8'))
        for name, digest in manifest['source_sha256'].items():
            assert hashlib.sha256(snapshot[name].encode()).hexdigest() == digest
        args = manifest['arguments']
        truths, rows = {}, {}
        for variant in args['variants']:
            rows[variant] = [json.loads(line) for line in
                (folder / variant / 'cases.jsonl').read_text(encoding='utf-8').splitlines()]
            assert [r['seed'] for r in rows[variant]] == list(range(args['start_seed'], args['start_seed']+args['cases']))
            for row in rows[variant]:
                case = json.loads((folder / variant / f"case_{row['seed']}.json").read_text(encoding='utf-8'))
                assert row == case['result']
                assert row['stop_reason'] == 'all_done' and row['status']['all_clear']
                assert row['cleared_count'] == len(case['truth'])
                assert all(s['cleared'] for s in case['truth'])
                assert case['truth'] == truths.setdefault(row['seed'], case['truth'])
                if args['minimum_reception']:
                    assert all(s['receiver_radius_m'] == 1000 for s in case['truth'])
        report[label] = {'cases_per_variant': args['cases'], 'variants': len(rows),
                         'truth_and_source_hashes_verified': True}
        if label == 'validation':
            delta = [a['total_time_s']-b['total_time_s'] for a,b in zip(rows['shared_f01'], rows['baseline_f01'])]
            avg, se = statistics.mean(delta), statistics.stdev(delta)/len(delta)**.5
            report[label]['shared_added_to_fraction01'] = {'mean_delta_s': avg,
                'approx_95_ci_s': [avg-1.96*se, avg+1.96*se]}
            cfg = StrategyConfig.from_file(ROOT / 'configs' / 'shared_near_01.json')
            cfg.validate()
            expected = json.loads((folder / 'shared_f01' / 'effective_config.json').read_text(encoding='utf-8'))
            # JSON logging maps an unlimited float to null.
            actual = {k: (None if isinstance(v,float) and v == float('inf') else v) for k,v in cfg.to_dict().items()}
            assert actual == expected, 'Candidate differs from tested configuration'
            report[label]['candidate_config_matches'] = True
    write_json(ROOT / 'logs/experiments/shared_validation_20260912/audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
