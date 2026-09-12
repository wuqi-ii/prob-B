"""Conservative continuous-disc coverage proof, not a point-sampling test.

The arena is enclosed by a quadtree. A square can be discarded only when
entirely outside the arena or entirely inside one reception disc. Every other
square is subdivided; unresolved leaves mean NOT certified. Floating-point
margins point in the conservative direction. No third-party dependencies.
"""
from functools import lru_cache
import math


def certifies_coverage(centres, arena_radius=1800.0, reception_radius=1000.0):
    centres = tuple(sorted(set((float(x), float(y)) for x, y in centres)))
    return _certify(centres, arena_radius, reception_radius)


@lru_cache(maxsize=4096)
def _certify(centres, arena_radius, reception_radius):
    if not centres:
        return False
    margin = 1e-6
    pending = [(0.0, 0.0, arena_radius, 0)]
    while pending:
        x, y, half, depth = pending.pop()
        # Minimum distance of this square from the arena centre.
        if math.hypot(max(abs(x)-half, 0), max(abs(y)-half, 0)) > arena_radius + margin:
            continue
        # Maximum distance to any point of the square: farthest corner.
        if any(math.hypot(abs(x-cx)+half, abs(y-cy)+half) <= reception_radius-margin
               for cx, cy in centres):
            continue
        # A witness inside the arena and outside every disc disproves coverage.
        if math.hypot(x, y) <= arena_radius and all(
            math.hypot(x-cx, y-cy) > reception_radius for cx, cy in centres
        ):
            return False
        if depth >= 10:
            return False  # unresolved boundary, never claim coverage from samples
        child = half/2
        pending.extend((x+dx*child, y+dy*child, child, depth+1)
                       for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)))
    return True
