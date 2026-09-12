"""Optional timing, per-measurement scheduling and certified clearance points."""
import math
from . import geometry
from .forward_strategy import ForwardStrategy


def closest_safe_point(cur, poly, radius=18.):
    """Projection onto intersection of equal-radius vertex discs, or None.

    Candidates: current point, circle projections and pair intersections.
    A convex objective optimum is interior, on one arc, or at an arc junction.
    Every returned candidate is checked against ALL polygon vertices.
    """
    if not poly:
        return None
    def valid(p):
        return all(math.dist(p,v)<=radius+1e-8 for v in poly)
    if valid(cur):
        return cur
    candidates=[geometry.centroid(poly)]
    for v in poly:
        d=math.dist(cur,v)
        if d>0:
            candidates.append(tuple(v[k]+radius*(cur[k]-v[k])/d for k in (0,1)))
    for i,a in enumerate(poly):
        for b in poly[i+1:]:
            d=math.dist(a,b)
            if d<=1e-10 or d>2*radius:
                continue
            h=math.sqrt(max(0.,radius*radius-d*d/4))
            mid=((a[0]+b[0])/2,(a[1]+b[1])/2)
            n=(-(b[1]-a[1])/d,(b[0]-a[0])/d)
            for sign in (-1,1):
                candidates.append((mid[0]+sign*h*n[0],mid[1]+sign*h*n[1]))
    feasible=[p for p in candidates if valid(p)]
    return min(feasible,key=lambda p:math.dist(cur,p)) if feasible else None


class RefinedStrategy(ForwardStrategy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._step_counts={}

    def _route_score(self,cur,route):
        score=super()._route_score(cur,route)
        if not self.cfg.route_time_score:
            return score
        remaining={ch for ch,tr in self.book.tracks.items() if not tr.cleared}
        for task in route:
            if task.kind=='scan':
                score+=30*len(remaining)  # six seconds, in equivalent metres
            elif task.kind in ('clear','direct_clear'):
                remaining.discard(task.channel)  # planning assumption only
        return score

    def _safe_point(self,ch):
        return closest_safe_point(self._pos(),self.book[ch].polygon(self.cfg),
                                  20-self.cfg.clear_safety_margin_m)

    def _service_tasks(self,cur,scans):
        tasks=super()._service_tasks(cur,scans)
        if self.cfg.safe_clear_point:
            for task in tasks:
                if task.kind=='clear':
                    p=self._safe_point(task.channel)
                    if p is not None:
                        task.point=p
        return tasks

    def _approach_and_clear(self,ch):
        if self.cfg.safe_clear_point and not self.book[ch].near_points:
            p=self._safe_point(ch)
            if p is not None:
                self._log('safe_clear_point',channel=ch,at=list(p))
                result=self.backend.clear(*p,ch)
                self.stats.clear_attempts+=1
                if result.get('clear_result')=='success':
                    return True
                self.stats.clear_miss+=1
                return False
        return super()._approach_and_clear(ch)

    def execute(self,task):
        if self.cfg.step_replan and task.kind=='clear':
            ch=task.channel
            tr=self.book[ch]
            summary=tr.geometry_summary(self.cfg)
            if (not tr.near_points and summary['centroid'] is not None
                and summary['centroid_max_error_m']>20-self.cfg.clear_safety_margin_m
                and self._step_counts.get(ch,0)<self.cfg.approach_max_iterations
                and not (self.cfg.safe_clear_point and self._safe_point(ch) is not None)):
                cur=self._pos()
                center=summary['centroid']
                step=max(self.cfg.approach_min_step_m,math.dist(cur,center)*self.cfg.approach_step_ratio)
                p=geometry.point_along(cur,center,step)
                result=self.backend.measure(*p,ch)
                kind=self.book.apply_measure(ch,p,result)
                self._step_counts[ch]=self._step_counts.get(ch,0)+1
                self._log('step_replan',channel=ch,at=list(p),result=kind)
                self._opportunistic_pass(p,exclude=ch)
                return
        return super().execute(task)
