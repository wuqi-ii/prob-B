"""Bounded candidate search using hypothetical geometry, never hidden truth."""
import math
from .config import SOURCE_COUNT_MAX
from . import geometry
from .second_station import candidate_pair
from .shared_strategy import SharedStrategy, polish_route
from .strategy import Task


class ForwardStrategy(SharedStrategy):
    def _upper_bound_complete(self):
        return self.cfg.stop_at_source_upper_bound and self.book.count_cleared() >= SOURCE_COUNT_MAX

    def status(self):
        state=super().status()
        if self._upper_bound_complete():
            state['all_clear']=True
            state['completion_basis']='cleared_source_count_upper_bound'
        return state

    def _do_clear(self,task):
        super()._do_clear(task)
        ch=task.channel
        tr=self.book[ch]
        if not self.cfg.recovery_verify or tr.cleared or not tr.observations:
            return
        counts=getattr(self,'_recovery_counts',{})
        if counts.get(ch,0)>=2:
            return
        counts[ch]=counts.get(ch,0)+1
        self._recovery_counts=counts
        pair=candidate_pair(*tr.observations[0],self.cfg)
        point=min((pair.left,pair.right),key=lambda p:math.dist(self._pos(),p))
        before=len(tr.observations)
        self._log('recovery_verify',channel=ch,at=list(point),attempt=counts[ch])
        self._do_verify(Task('verify',point,ch,note='recovery_standard'))
        if len(tr.observations)>before or tr.near_points:
            self._give_up.discard(ch)

    def _route_score(self, cur, route):
        return self._path_length(cur,route)

    def _best_side(self, cur, scans, track):
        if not self.cfg.adaptive_verify:
            return super()._best_side(cur, scans, track)
        poly, center, _ = self._shape(track.channel)
        if center is None:
            return super()._best_side(cur, scans, track)
        station, _ = track.observations[0]
        pair = candidate_pair(*track.observations[0], self.cfg)
        spine = self._shortest_scan_spine(cur, scans)
        # Interior scenarios plus the centre; not a fitted probability model.
        samples = [center] + [((p[0]+center[0])/2, (p[1]+center[1])/2)
                              for p in poly[::max(1, len(poly)//4)][:4]]
        options = []
        for full in (pair.left, pair.right):
            for fraction in (.05, .1, .2, .4, .7):
                point = tuple(a+fraction*(b-a) for a,b in zip(station, full))
                if any(math.dist(point,p)<1 for p,_ in track.observations):
                    continue
                task = Task('verify', full, track.channel, note='adaptive_verify', near_point=point)
                downstream = []
                for target in samples:
                    distance = math.dist(point, target)
                    if distance > 1000:
                        downstream.append(math.dist(point,full)+math.dist(full,target)+30)
                        continue
                    angle = math.degrees(math.atan2(target[1]-point[1],target[0]-point[0]))
                    for error in (-1.,0.,1.):
                        posterior = geometry.clip_wedge(poly, point, angle+error, 1.)
                        if not posterior:
                            continue
                        c = geometry.centroid(posterior)
                        radius = geometry.max_distance_from(c, posterior)
                        downstream.append(math.dist(point,c)+max(0.,radius-18))
                detour = min(self._insertion_delta(cur,spine,task,i) for i in range(len(spine)+1))
                options.append((detour+sum(downstream)/len(downstream),task))
        return min(options,key=lambda x:x[0])[1] if options else super()._best_side(cur,scans,track)

    def select_task(self):
        if self._upper_bound_complete():
            self._log('upper_bound_complete',cleared=SOURCE_COUNT_MAX)
            return None
        if not self.cfg.route_multistart:
            return super().select_task()
        cur = self._pos()
        scans = self._scan_tasks()
        services = self._service_tasks(cur, scans)
        incumbent = self._plan(cur, scans, services)
        if not incumbent:
            return None
        candidates = [polish_route(cur, incumbent)]
        # Enumerate the next action, complete by nearest insertion, then 2-opt.
        # Replan after feedback; this is NOT a global stochastic optimum.
        tasks = [*scans, *services]
        for first in tasks:
            candidates.append(self._completion_from_first(first,tasks))
        return min(candidates,key=lambda r:self._route_score(cur,r))[0]

    @staticmethod
    def _completion_from_first(first,tasks):
        route=[first]
        pending=[t for t in tasks if t is not first]
        while pending:
            nxt=min(pending,key=lambda t:math.dist(route[-1].execution_point,t.execution_point))
            route.append(nxt)
            pending.remove(nxt)
        return [first,*polish_route(first.execution_point,route[1:])]
