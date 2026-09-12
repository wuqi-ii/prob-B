"""扫描路线中的清除/验证任务插入代价 A/B。"""
from __future__ import annotations
import argparse, math, statistics, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cumcm_b4.backend_offline import OfflineSimulator,generate_case  # noqa:E402
from cumcm_b4.config import StrategyConfig  # noqa:E402
from cumcm_b4.strategy import DogStrategy  # noqa:E402

class InsertionStrategy(DogStrategy):
    def __init__(self,*args,clear_threshold,verify_threshold,**kwargs):
        super().__init__(*args,**kwargs)
        self.clear_threshold=clear_threshold
        self.verify_threshold=verify_threshold
    def _priority(self,task,cur):
        unvisited=[p for i,p in enumerate(self.scan_points) if not self.visited_scan[i]]
        if not unvisited:
            return math.dist(cur,task.point)
        nxt=min(unvisited,key=lambda p:math.dist(cur,p))
        if task.kind=='scan':
            return 0.0 if task.point==nxt else 1e6+math.dist(cur,task.point)
        detour=(math.dist(cur,task.point)+math.dist(task.point,nxt)-math.dist(cur,nxt))
        threshold=self.clear_threshold if task.kind=='clear' else self.verify_threshold
        return detour-threshold

def run(seed,cfg,p,ct,vt):
    sim=OfflineSimulator(generate_case(seed,directional_prob=p))
    st=(DogStrategy(sim,cfg) if ct is None else InsertionStrategy(sim,cfg,clear_threshold=ct,verify_threshold=vt)).run()
    return sim.virtual_time_s,sim.virtual_time_s/sim.case.total,st.cleared_count,sim.case.total

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cases',type=int,default=50);ap.add_argument('--start-seed',type=int,default=14000);ap.add_argument('--directional-prob',type=float,default=.5);a=ap.parse_args()
    cfg=StrategyConfig(); variants=[('base',None,None),('i50',50,0),('i100',100,0),('i200',200,0),('i400',400,0),('i200v100',200,100)]
    results={}
    for name,ct,vt in variants:
        rows=[run(s,cfg,a.directional_prob,ct,vt) for s in range(a.start_seed,a.start_seed+a.cases)];results[name]=rows
        ps=[x[1] for x in rows];print(name,statistics.fmean(x[0] for x in rows),min(ps),statistics.fmean(ps),max(ps),sum(x[2]==x[3] for x in rows),flush=True)
    b=results['base'];bm=statistics.fmean(x[0] for x in b)
    for name,_,_ in variants[1:]:
        d=[x[0]-y[0] for x,y in zip(results[name],b)];print(name,'paired',statistics.fmean(d),statistics.fmean(d)/bm,statistics.pstdev(d)/math.sqrt(len(d)))
if __name__=='__main__':raise SystemExit(main())
