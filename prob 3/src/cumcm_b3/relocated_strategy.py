"""Move next scan toward service routes only after continuous coverage proof."""
import math
from .refined_strategy import RefinedStrategy
from .strategy import Task


class RelocatedStrategy(RefinedStrategy):
    def _scan_index(self, task):
        """把扫描任务解析成扫描点下标。

        基类 ``_scan_tasks`` 生成的 note 形如 ``S3``，但子类改写后可能是别的
        形态；直接 ``int(note[1:])`` 会在 note 不含数字时抛出难以定位的
        ValueError。这里先按约定解析，失败则按点位回退，仍失败就给出明确错误。
        """
        note = task.note or ''
        if note[:1] == 'S' and note[1:].isdigit():
            idx = int(note[1:])
            if 0 <= idx < len(self.scan_points):
                return idx
        for i, point in enumerate(self.scan_points):
            if math.dist(point, task.point) < 1e-9:
                return i
        raise ValueError(f"无法把扫描任务定位到扫描点：note={note!r} point={task.point}")

    def select_task(self):
        task=super().select_task()
        if task is None or task.kind!='scan':
            return task
        cur=self._pos()
        scans=self._scan_tasks()
        rest=[s for s in scans if s.note!=task.note]
        services=self._service_tasks(cur,scans)
        tasks=[*rest,*services]
        def route_for(p):
            first=Task('relocated_scan',p,note=task.note[1:])
            return self._completion_from_first(first,[first,*tasks])
        def score(p):
            return self._route_score(cur,route_for(p))
        candidates=[cur]
        destinations=sorted([t.execution_point for t in services],key=lambda p:math.dist(task.point,p))[:4]
        for dest in [cur,*destinations]:
            for f in (.125,.25,.5,.75,1.):
                candidates.append(tuple(a+f*(b-a) for a,b in zip(task.point,dest)))
        best,best_score=task.point,score(task.point)
        for point in sorted(candidates,key=score):
            if score(point)>=best_score-1e-6:
                continue
            if self._proven_with([point,*(s.point for s in rest)]):
                best,best_score=point,score(point)
                break
        if best==task.point:
            return task
        return route_for(best)[0]

    def execute(self,task):
        if task.kind=='scan':
            self._visit_scan_point(self._scan_index(task))
            return
        if task.kind=='relocated_scan':
            idx=int(task.note)
            remaining=[p for i,p in enumerate(self.scan_points) if not self.visited_scan[i] and i!=idx]
            if not self._proven_with([task.point,*remaining]):
                self._visit_scan_point(idx)
                return
            old=self.scan_points[idx]
            self.scan_points[idx]=task.point
            self._log('relocated_scan',old=list(old),new=list(task.point),
                      remaining=[list(p) for p in remaining],certified=True)
            self._visit_scan_point(idx)
            return
        return super().execute(task)
