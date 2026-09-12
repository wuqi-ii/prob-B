"""Adversarial fixed +/-1 degree errors with boundary and inner sources."""
import argparse
import math
from pathlib import Path
import sys
from dataclasses import replace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cumcm_b3.backend_offline import WorldCases,Source,OfflineSimulator
from cumcm_b3.config import StrategyConfig
from cumcm_b3.experimental_strategy import make_strategy
from cumcm_b3.metrics import write_json

p=argparse.ArgumentParser()
p.add_argument('--outdir',required=True)
p.add_argument('--config')
args=p.parse_args()
out=ROOT/args.outdir
out.mkdir(parents=True,exist_ok=False)
cfg=StrategyConfig.from_file(args.config) if args.config else replace(StrategyConfig(),route_multistart=True,
    shared_observations='opportunistic',verify_near_fraction=.1,recovery_verify=True,
    scan_relocation=True,safe_clear_point=True,step_replan=True)
write_json(out/'effective_config.json',cfg.to_dict())
for error in (-1.005,1.005):
    case=WorldCases([Source(i+1,1800*math.cos((i+.5)*math.pi/3),1800*math.sin((i+.5)*math.pi/3),1000)
        for i in range(6)]+[Source(i+7,800*math.cos(i*math.pi/2),800*math.sin(i*math.pi/2),1000)
        for i in range(4)],1)
    sim=OfflineSimulator(case)
    s=make_strategy(sim,cfg)
    with patch('cumcm_b3.backend_offline._deterministic_error',return_value=error):
        stats=s.run()
    write_json(out/f'error_{error}.json',dict(events=stats.events,trace=sim.trace,
        truth=[vars(x) for x in case.sources],tracks={ch:vars(tr) for ch,tr in s.book.tracks.items()},
        status=s.status(),stop_reason=stats.stop_reason,time_s=stats.virtual_time_s))
    print(error,stats.cleared_count,stats.stop_reason,s.status())
    for ch in s.book.known_channels():
        print('unresolved',ch,s.book[ch].geometry_summary(cfg))
