"""Audit saved experiments, including source snapshots and paired differences."""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cumcm_b3.metrics import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--folders',nargs='+',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--reference',default='forward_route')
    args=p.parse_args()
    audit={}
    for name in args.folders:
        folder=ROOT/'logs/experiments'/name
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        snapshot=json.loads((folder/'source_snapshot.json').read_text(encoding='utf-8'))
        for path,digest in manifest['source_sha256'].items():
            assert hashlib.sha256(snapshot[path].encode()).hexdigest()==digest
        settings=manifest['arguments']
        rows,truth={},{}
        for variant in settings['variants']:
            rows[variant]=[json.loads(s) for s in (folder/variant/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
            assert [r['seed'] for r in rows[variant]]==list(range(settings['start_seed'],settings['start_seed']+settings['cases']))
            for row in rows[variant]:
                detail=json.loads((folder/variant/f"case_{row['seed']}.json").read_text(encoding='utf-8'))
                assert detail['result']==row
                assert detail['truth']==truth.setdefault(row['seed'],detail['truth'])
                assert all(s['cleared'] for s in detail['truth'])
                assert row['cleared_count']==len(detail['truth'])
                assert row['stop_reason']=='all_done' and row['status']['all_clear']
                if settings['minimum_reception']:
                    assert all(s['receiver_radius_m']==1000 for s in detail['truth'])
        comparison={}
        for variant,data in rows.items():
            ds=[a['total_time_s']-b['total_time_s'] for a,b in zip(data,rows[args.reference])]
            avg,se=statistics.mean(ds),statistics.stdev(ds)/len(ds)**.5
            comparison[variant]={'mean_s':statistics.mean(r['total_time_s'] for r in data),
                'delta_vs_reference_s':avg,'approx_95_ci_s':[avg-1.96*se,avg+1.96*se],
                'faster':sum(d<0 for d in ds),'slower':sum(d>0 for d in ds),
                'sources':sum(r['total_sources'] for r in data),
                'max_wall_s':max(r['wall_s'] for r in data)}
        audit[name]={'runs':sum(map(len,rows.values())),'verified':True,'comparison':comparison}
    write_json(ROOT/args.out,audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
