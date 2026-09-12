"""Opt-in adaptive search and joint verification/clearance experiments.

Only ordinary observations are used. The fixed covered layout remains a
fallback. Unknown channels must EACH obtain a continuous coverage certificate.
"""
import math
from dataclasses import dataclass

from . import geometry
from .config import DOG_SPEED_MPS
from .coverage_certificate import certifies_coverage
from .second_station import candidate_pair
from .strategy import DogStrategy, Task


@dataclass
class JointTask(Task):
    predicted_end: tuple = (0.0, 0.0)


class ExperimentalStrategy(DogStrategy):
    def _unknown(self):
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
        return self._prune_scans(tasks) if self.cfg.adaptive_search else tasks

    def status(self):
        state = super().status()
        if self.cfg.adaptive_search:
            uncertified = [ch for ch in self._unknown()
                           if not certifies_coverage(self.book[ch].no_signal_points)]
            state['uncertified_channels'] = uncertified
            state['completion_basis'] = 'per_channel_continuous_coverage'
            state['all_clear'] = not uncertified and not state['known_not_cleared']
        return state

    def _search_at_stop(self, point):
        channels = self._unknown()
        if not channels:
            return
        scans = self._scan_tasks()
        proposed = self._prune_scans(scans, extra=(point,))
        # Spend measurements only if they can remove at least one scan stop.
        # Each removed scan would measure these unknown channels anyway.
        if len(proposed) >= len(scans):
            return
        for ch in channels:
            if certifies_coverage(self.book[ch].no_signal_points):
                continue
            payload = self.backend.measure(*point, ch)
            kind = self.book.apply_measure(ch, point, payload)
            self._log('adaptive_measure', channel=ch, at=list(point), result=kind)
        retained = {s.note for s in self._scan_tasks()}
        self._log('adaptive_replacement', at=list(point),
                  replaced=[s.note for s in scans if s.note not in retained])

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
        if self.cfg.joint_service and isinstance(task, JointTask):
            track = self.book[task.channel]
            if track.cleared or task.channel in self._give_up:
                return
            if track.near_points or len(track.observations) >= 2:
                if self._budget_exhausted({}):
                    return
                self._log('joint_clear_start', channel=task.channel)
                self._do_clear(Task('clear', self._pos(), task.channel))


def make_strategy(backend, cfg, verbose=False):
    cls = ExperimentalStrategy if cfg.adaptive_search or cfg.joint_service else DogStrategy
    return cls(backend, cfg, verbose=verbose)
