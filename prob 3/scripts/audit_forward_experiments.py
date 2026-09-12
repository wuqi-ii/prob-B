"""Validate paired evidence and save exact paired differences for this round."""
import hashlib
import json
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cumcm_b3.config import StrategyConfig
from cumcm_b3.metrics import write_json


def main():
    audit={}
    for label in ('pilot','validation','min1000'):
        directory=ROOT/'logs/experiments'/f'forward_{label}_20260912'
        manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        snapshot=json.loads((directory/'source_snapshot.json').read_text(encoding='utf-8'))
        for path,digest in manifest['source_sha256'].items():
            assert hashlib.sha256(snapshot[path].encode()).hexdigest()==digest
        args=manifest['arguments']
        truth,rows={},{}
        for variant in args['variants']:
            rows[variant]=[json.loads(s) for s in (directory/variant/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
            assert [r['seed'] for r in rows[variant]]==list(range(args['start_seed'],args['start_seed']+args['cases']))
            for row in rows[variant]:
                case=json.loads((directory/variant/f"case_{row['seed']}.json").read_text(encoding='utf-8'))
                assert row==case['result']
                assert case['truth']==truth.setdefault(row['seed'],case['truth'])
                assert all(s['cleared'] for s in case['truth'])
                assert row['cleared_count']==len(case['truth']) and row['stop_reason']=='all_done'
                assert row['status']['all_clear']
                if args['minimum_reception']:
                    assert all(s['receiver_radius_m']==1000 for s in case['truth'])
        summary={'complete_runs':sum(map(len,rows.values())), 'evidence_checks_passed':True}
        deltas=[a['total_time_s']-b['total_time_s'] for a,b in zip(rows['forward_route'],rows['shared_f01'])]
        avg,se=statistics.mean(deltas),statistics.stdev(deltas)/len(deltas)**.5
        summary['vs_previous_candidate']={'mean_delta_s':avg,'approx_95_ci_s':[avg-1.96*se,avg+1.96*se],
            'change_percent':100*avg/statistics.mean(r['total_time_s'] for r in rows['shared_f01']),
            'faster':sum(d<0 for d in deltas),'slower':sum(d>0 for d in deltas)}
        summary['max_case_wall_s']=max(r['wall_s'] for r in rows['forward_route'])
        audit[label]=summary
        if label=='validation':
            cfg=StrategyConfig.from_file(ROOT/'configs/multistart_shared_01.json')
            cfg.validate()
            actual={k:None if isinstance(v,float) and v==float('inf') else v for k,v in cfg.to_dict().items()}
            assert actual==json.loads((directory/'forward_route/effective_config.json').read_text(encoding='utf-8'))
    write_json(ROOT/'logs/experiments/forward_validation_20260912/audit.json',audit)
    write_json(ROOT/'logs/experiments/forward_validation_20260912/harness_snapshot.json',
        {name:(ROOT/name).read_text(encoding='utf-8') for name in (
            'scripts/compare_search_experiments.py','scripts/audit_forward_experiments.py',
            'scripts/run_drill.py','tests/test_forward_strategy.py','tests/test_http_protocol.py')})
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
