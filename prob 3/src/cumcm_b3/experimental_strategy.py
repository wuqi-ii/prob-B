"""Opt-in adaptive search and joint verification/clearance experiments.

Only ordinary observations are used. The fixed covered layout remains a
fallback. Unknown channels must EACH obtain a continuous coverage certificate.
"""
import math
from dataclasses import dataclass

from . import geometry
from .config import DOG_SPEED_MPS, SOURCE_COUNT_MAX
from .coverage_certificate import certifies_coverage
from .second_station import candidate_pair
from .strategy import DogStrategy, Task


@dataclass
class JointTask(Task):
    predicted_end: tuple = (0.0, 0.0)


class ExperimentalStrategy(DogStrategy):
    def __init__(self, backend, cfg, verbose=False):
        super().__init__(backend, cfg, verbose)
        self._adaptive_scan_count = 0
        self._plan_cache = {}

    def _unknown(self):
        # The problem gives at most 16 distinct sources. Once 16 channels
        # have actually been observed, no additional unknown source exists.
        # Count observed tracks (including cleared ones), never oracle truth.
        if sum(tr.known for tr in self.book.tracks.values()) >= SOURCE_COUNT_MAX:
            return []
        return [ch for ch, tr in self.book.tracks.items() if not tr.known and not tr.cleared]

    def _proven_with(self, planned):
        # Planned points may certify a PLAN but never the actual stopping state.
        histories = {tuple(self.book[ch].no_signal_points) for ch in self._unknown()}
        return all(certifies_coverage((*history, *planned)) for history in histories)

    def _prune_scans(self, scans, extra=()):
        remaining = list(scans)
        # Delete one at a time: independent tests followed by simultaneous
        # deletion could remove two points which cover for each other.
        for task in sorted(scans, key=lambda t: math.dist(self._pos(), t.point), reverse=True):
            proposed = [s for s in remaining if s is not task]
            if self._proven_with([*extra, *(s.point for s in proposed)]):
                remaining = proposed
        return remaining

    def _scan_tasks(self):
        tasks = super()._scan_tasks()
        return self._search_plan()[0] if self.cfg.adaptive_search else tasks

    def _search_plan(self, extra=()):
        histories = tuple(sorted({tuple(tr.no_signal_points) for ch, tr in self.book.tracks.items()
                                  if ch in self._unknown()}))
        key = (self._pos(), histories, tuple(extra), len(self._unknown()))
        if key in self._plan_cache:
            return self._plan_cache[key]
        if not self._unknown() or self._proven_with(extra):
            return [], 0.0
        # Plan remaining search afresh. These are candidate centres, not
        # relaxed radii: every sensing disc is still exactly 1000 m.
        layouts = [super()._scan_tasks()]
        for radius in (1150.0, 1350.0, 1550.0):
            for phase in (0.0, math.pi/6):
                layouts.append([Task('scan', (radius*math.cos(k*math.pi/3+phase),
                                               radius*math.sin(k*math.pi/3+phase)),
                                     note=f'adaptive:{radius:g}:{phase:g}:{k}') for k in range(6)])
        best, best_cost = None, float('inf')
        for layout in layouts:
            if not self._proven_with([*extra, *(t.point for t in layout)]):
                continue
            remaining = self._prune_scans(layout, extra)
            route = self._shortest_scan_spine(self._pos(), remaining)
            cost = self._path_length(self._pos(), route)/DOG_SPEED_MPS
            cost += len(route)*len(self._unknown())*6.0
            if cost < best_cost:
                best, best_cost = route, cost
        # Original complete layout is guaranteed under cfg.validate().
        # If numerical certificates cannot prove a candidate, keep all seven.
        if best is None:
            best = [Task('scan', p, note=f'fallback:{i}') for i, p in enumerate(self.scan_points)]
            best_cost = self._path_length(self._pos(), best)/DOG_SPEED_MPS + len(best)*len(self._unknown())*6
        if len(self._plan_cache) > 128:
            self._plan_cache.clear()
        self._plan_cache[key] = (best, best_cost)
        return best, best_cost

    def status(self):
        state = super().status()
        if self.cfg.adaptive_search:
            uncertified = [ch for ch in self._unknown()
                           if not certifies_coverage(self.book[ch].no_signal_points)]
            state['uncertified_channels'] = uncertified
            state['completion_basis'] = ('observed_source_count_upper_bound' if
                sum(tr.known for tr in self.book.tracks.values()) >= SOURCE_COUNT_MAX else
                'per_channel_continuous_coverage')
            state['all_clear'] = not uncertified and not state['known_not_cleared']
        return state

    def _search_at_stop(self, point):
        channels = self._unknown()
        if not channels:
            return
        scans = self._scan_tasks()
        _, before_cost = self._search_plan()
        proposed, after_cost = self._search_plan(extra=(point,))
        # Include the cost of these ACTUAL additional measurements. No
        # probability-of-detection or empirical information weights needed.
        if before_cost - after_cost <= len(channels)*6.0:
            return
        for ch in channels:
            if certifies_coverage(self.book[ch].no_signal_points):
                continue
            payload = self.backend.measure(*point, ch)
            kind = self.book.apply_measure(ch, point, payload)
            self._log('adaptive_measure', channel=ch, at=list(point), result=kind)
        retained = {s.note for s in self._scan_tasks()}
        self._log('adaptive_replacement', at=list(point),
                  replaced=[s.note for s in scans if s.note not in retained],
                  estimated_saved_s=before_cost-after_cost-len(channels)*6.0)

    def execute(self, task):
        if self.cfg.adaptive_search and task.kind == 'scan':
            channels = [ch for ch, tr in self.book.tracks.items() if not tr.cleared]
            for ch in channels:
                if not self.book[ch].known and certifies_coverage(self.book[ch].no_signal_points):
                    continue
                payload = self.backend.measure(*task.point, ch)
                kind = self.book.apply_measure(ch, task.point, payload)
                if kind != 'no_signal':
                    self._log('discover', channel=ch, at=list(task.point), kind=kind)
            self._adaptive_scan_count += 1
            if task.point in self.scan_points:
                self.visited_scan[self.scan_points.index(task.point)] = True
            self._log('scan_done', point=list(task.point), index=None, adaptive=True,
                      channels=len(channels), note=task.note)
            return
        super().execute(task)

    def run(self):
        stats = super().run()
        if self.cfg.adaptive_search:
            stats.scan_points_visited = 1 + self._adaptive_scan_count
        return stats

    def _opportunistic_pass(self, point, exclude=None):
        super()._opportunistic_pass(point, exclude)
        if self.cfg.adaptive_search:
            self._search_at_stop(point)

    def _predicted_target(self, track):
        # Enclose all plausible sources using received <=1500m and ±1 degree
        # wedge, then use its centroid as a heuristic (not a guarantee).
        poly = track.polygon(self.cfg)
        station, bearing = track.observations[0]
        far = self.cfg.second_station_far_m
        for k in range(32):
            angle = 2*math.pi*k/32
            nx, ny = math.cos(angle), math.sin(angle)
            poly = geometry.clip_halfplane(poly, nx, ny, far+nx*station[0]+ny*station[1])
        return geometry.centroid(poly) if poly else None

    @staticmethod
    def _end(task):
        return task.predicted_end if isinstance(task, JointTask) else task.execution_point

    @classmethod
    def _insertion_delta(cls, cur, route, task, pos):
        start, end = task.execution_point, cls._end(task)
        prev = cur if pos == 0 else cls._end(route[pos-1])
        result = math.dist(prev, start) + math.dist(start, end)
        if pos < len(route):
            nxt = route[pos].execution_point
            result += math.dist(end, nxt) - math.dist(prev, nxt)
        # Additional action costs for verify-then-clear; scanning/ordinary
        # clearance costs are common to inserting the same required tasks.
        if isinstance(task, JointTask):
            result += DOG_SPEED_MPS * (5 + (5 if task.channel is not None else 0))
        return result

    def _best_side(self, cur, scans, track):
        if not self.cfg.joint_service:
            return super()._best_side(cur, scans, track)
        target = self._predicted_target(track)
        if target is None:
            return super()._best_side(cur, scans, track)
        station, bearing = track.observations[0]
        pair = candidate_pair(station, bearing, self.cfg)
        spine = self._shortest_scan_spine(cur, scans)
        options = []
        for label, point in (('left', pair.left), ('right', pair.right)):
            near = None
            f = self.cfg.verify_near_fraction
            if 0 < f < 1:
                near = (station[0]+f*(point[0]-station[0]), station[1]+f*(point[1]-station[1]))
            options.append(JointTask('verify', point, track.channel, note=f'joint:{label}',
                                     near_point=near, predicted_end=target))
        return min(options, key=lambda t: min(self._insertion_delta(cur, spine, t, pos)
                                              for pos in range(len(spine)+1)))

    def _do_verify(self, task):
        super()._do_verify(task)
        # New bearing changes the predicted endpoint; the main loop then
        # replans using the actual posterior rather than forcing an old order.
        if isinstance(task, JointTask):
            self._log('joint_replan', channel=task.channel, predicted_end=list(task.predicted_end))


def make_strategy(backend, cfg, verbose=False):
    if cfg.scan_relocation:
        from .relocated_strategy import RelocatedStrategy
        return RelocatedStrategy(backend,cfg,verbose=verbose)
    if cfg.route_time_score or cfg.step_replan or cfg.safe_clear_point:
        from .refined_strategy import RefinedStrategy
        return RefinedStrategy(backend,cfg,verbose=verbose)
    if cfg.adaptive_verify or cfg.route_multistart or cfg.recovery_verify:
        from .forward_strategy import ForwardStrategy
        return ForwardStrategy(backend, cfg, verbose=verbose)
    if cfg.shared_observations != 'off' or cfg.route_polish:
        from .shared_strategy import SharedStrategy
        return SharedStrategy(backend, cfg, verbose=verbose)
    cls = ExperimentalStrategy if cfg.adaptive_search or cfg.joint_service else DogStrategy
    return cls(backend, cfg, verbose=verbose)
