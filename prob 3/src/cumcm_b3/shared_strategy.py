"""Shared observation and route polishing experiments. No access to truth.

Future bearings to feasible-region centroids are only planning hypotheses.
Only actual received bearings update tracks or authorize clearance.
"""
import math
from dataclasses import dataclass

from . import geometry
from .config import ARENA_RADIUS_M, BEARING_ERROR_DEG, RECEIVER_MAX_M
from .experimental_strategy import ExperimentalStrategy
from .strategy import Task


@dataclass
class BatchTask(Task):
    channels: tuple = ()

    @property
    def key(self):
        return 'batch:' + ','.join(map(str, self.channels))


def polish_route(start, route):
    """2-opt for a fixed start and free endpoint; accept strict improvement."""
    result = list(route)
    for _ in range(20):
        best_delta, best = -1e-7, None
        for i in range(len(result)-1):
            prev = start if i == 0 else result[i-1].execution_point
            first = result[i].execution_point
            for j in range(i+1, len(result)):
                last = result[j].execution_point
                delta = math.dist(prev, last)-math.dist(prev, first)
                if j+1 < len(result):
                    nxt = result[j+1].execution_point
                    delta += math.dist(first, nxt)-math.dist(last, nxt)
                if delta < best_delta:
                    best_delta, best = delta, (i, j)
        if best is None:
            break
        i, j = best
        result[i:j+1] = reversed(result[i:j+1])
    return result


class SharedStrategy(ExperimentalStrategy):
    def __init__(self, backend, cfg, verbose=False):
        super().__init__(backend, cfg, verbose)
        self._shape_cache = {}

    def _shape(self, ch):
        tr = self.book[ch]
        key = (ch, len(tr.observations))
        if key not in self._shape_cache:
            poly = geometry.locate(tr.observations, BEARING_ERROR_DEG+self.cfg.bearing_rounding_slack_deg,
                                   ARENA_RADIUS_M, max_range_m=RECEIVER_MAX_M)
            if poly:
                centroid = geometry.centroid(poly)
                self._shape_cache[key] = (poly, centroid, geometry.max_distance_from(centroid, poly))
            else:
                self._shape_cache[key] = ([], None, 0.0)
        return self._shape_cache[key]

    def _gain(self, ch, point):
        tr = self.book[ch]
        if tr.cleared or tr.near_points or ch in self._give_up:
            return 0.0
        if any(math.dist(p, point) < 1.0 for p, _ in tr.observations):
            return 0.0
        if any(math.dist(p, point) < 1.0 for p in tr.no_signal_points):
            return 0.0
        poly, centroid, before = self._shape(ch)
        if centroid is None or before <= 18.0 or math.dist(centroid, point) > 1000.0:
            return 0.0
        bearing = math.degrees(math.atan2(centroid[1]-point[1], centroid[0]-point[0]))
        after_poly = geometry.clip_wedge(poly, point, bearing, BEARING_ERROR_DEG+self.cfg.bearing_rounding_slack_deg)
        if not after_poly:
            return 0.0
        after = geometry.max_distance_from(geometry.centroid(after_poly), after_poly)
        # Benefit proxy in metres; six seconds of sensing costs 30 m of travel.
        return max(0.0, before-after-30.0)

    def _opportunistic_pass(self, point, exclude=None):
        if self.cfg.shared_observations == 'off':
            return super()._opportunistic_pass(point, exclude)
        candidates = [(self._gain(ch, point), ch) for ch in self.book.known_channels() if ch != exclude]
        for gain, ch in sorted(candidates, reverse=True):
            if gain <= 0:
                break
            if self._budget_exhausted({}):
                break
            payload = self.backend.measure(*point, ch)
            kind = self.book.apply_measure(ch, point, payload)
            self._log('shared_measure', channel=ch, at=list(point), result=kind,
                      predicted_gain_m=gain)
        if self.cfg.adaptive_search:
            self._search_at_stop(point)

    def _plan(self, cur, scans, services):
        route = self._shortest_scan_spine(cur, scans)
        pending = list(services)
        while pending:
            choices = []
            for index, task in enumerate(pending):
                for pos in range(len(route)+1):
                    delta = self._insertion_delta(cur, route, task, pos)
                    choices.append((delta-self._task_bonus(task), delta, task.key, pos, index))
            _, _, _, pos, index = min(choices)
            route.insert(pos, pending.pop(index))
        return polish_route(cur, route) if self.cfg.route_polish else route

    def select_task(self):
        if self.cfg.shared_observations != 'batch' and not self.cfg.route_polish:
            return super().select_task()
        cur = self._pos()
        scans = self._scan_tasks()
        services = self._service_tasks(cur, scans)
        route = self._plan(cur, scans, services)
        if not route:
            return None
        if self.cfg.shared_observations != 'batch':
            return route[0]
        verifications = [t for t in services if t.kind == 'verify']
        if len(verifications) < 2:
            return route[0]
        points = [t.execution_point for t in verifications]
        # Shared candidates include pair midpoints as well as existing points.
        points += [((a[0]+b[0])/2, (a[1]+b[1])/2)
                   for i, a in enumerate(points) for b in points[i+1:]
                   if math.dist(a, b) < 1000]
        best, best_length = route[0], self._path_length(cur, route)
        for point in points:
            channels = tuple(t.channel for t in verifications if self._gain(t.channel, point) > 0)
            if len(channels) < 2:
                continue
            batch = BatchTask('batch', point, note='shared_observation', channels=channels)
            remaining = [t for t in services if not (t.kind == 'verify' and t.channel in channels)]
            alternative = self._plan(cur, scans, [*remaining, batch])
            if alternative[0] is not batch:
                continue  # only act on shared visits that should happen now
            length = self._path_length(cur, alternative)
            if length < best_length-1e-6:
                best, best_length = batch, length
        return best

    def execute(self, task):
        if not isinstance(task, BatchTask):
            return super().execute(task)
        self._log('batch_start', channels=list(task.channels), at=list(task.point))
        for ch in task.channels:
            if self.book[ch].cleared or self._budget_exhausted({}):
                continue
            payload = self.backend.measure(*task.point, ch)
            kind = self.book.apply_measure(ch, task.point, payload)
            self._log('batch_measure', channel=ch, at=list(task.point), result=kind)
        self._opportunistic_pass(task.point)
