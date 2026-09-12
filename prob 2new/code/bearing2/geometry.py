"""Small convex-polygon primitives. Coordinates remain in a common local frame.

Half-plane clipping preserves CCW order. Rotating calipers handles tied support
vertices; the quadratic oracle is retained for independent unit tests.
"""
import math
import numpy as np


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def clean_polygon(vertices, tol=1e-9):
    out = []
    for p in vertices:
        p = (float(p[0]), float(p[1]))
        if not out or math.dist(p, out[-1]) > tol:
            out.append(p)
    if len(out) > 1 and math.dist(out[0], out[-1]) <= tol:
        out.pop()
    return out


def clip_halfplane(poly, nx, ny, rhs, tol=1e-9):
    """Clip a CCW convex polygon against nx*x + ny*y <= rhs."""
    if not poly:
        return []
    result = []
    a = poly[-1]
    da = nx * a[0] + ny * a[1] - rhs
    for b in poly:
        db = nx * b[0] + ny * b[1] - rhs
        ia, ib = da <= tol, db <= tol
        if ia != ib:
            den = da - db
            if abs(den) > 1e-15:
                ratio = min(1.0, max(0.0, da / den))
                result.append((a[0] + ratio * (b[0] - a[0]), a[1] + ratio * (b[1] - a[1])))
        if ib:
            result.append(b)
        a, da = b, db
    return clean_polygon(result)


def clip_wedge(poly, station, bearing, half_angle):
    """The forward wedge, never an unoriented pair of intersecting lines."""
    if not 0 < half_angle < math.pi / 2:
        raise ValueError("Wedge half-angle must be between 0 and pi/2.")
    px, py = station
    lo, hi = bearing - half_angle, bearing + half_angle
    # cross(d_lo, X-P)>=0; cross(d_hi, X-P)<=0.
    nx, ny = math.sin(lo), -math.cos(lo)
    out = clip_halfplane(poly, nx, ny, nx * px + ny * py)
    nx, ny = -math.sin(hi), math.cos(hi)
    return clip_halfplane(out, nx, ny, nx * px + ny * py)


def clip_disk_outer(poly, center, radius, sides):
    """Circumscribed polygon made of supporting half-planes; never lose a source.

Every point retained by all these disk half-planes is at most R*sec(pi/N)
from the disk centre. See MODEL.md for the conservative approximation.
"""
    out = list(poly)
    for k in range(sides):
        angle = 2 * math.pi * k / sides
        nx, ny = math.cos(angle), math.sin(angle)
        out = clip_halfplane(out, nx, ny, radius + nx * center[0] + ny * center[1])
        if not out:
            break
    return out


def diameter_bruteforce(poly):
    if not poly:
        return 0.0, None
    best, pair = 0.0, (poly[0], poly[0])
    for i, a in enumerate(poly):
        for b in poly[i + 1:]:
            d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            if d > best:
                best, pair = d, (a, b)
    return math.sqrt(best), pair


def signed_area(poly):
    total = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def span_scale(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return max(max(xs) - min(xs), max(ys) - min(ys), 1.0)


def diameter(poly):
    """O(m) rotating calipers on an ordered convex polygon (m>=3).

    A zero-area (collinear/degenerate) input is really a segment: the calipers
    advance only on strictly increasing support areas, so such an input would
    miss distant vertex pairs. It is routed to the quadratic oracle instead.
    """
    n = len(poly)
    if n <= 3 or abs(signed_area(poly)) <= 1e-12 * span_scale(poly) ** 2:
        return diameter_bruteforce(poly)
    best, pair, j = 0.0, (poly[0], poly[0]), 1

    def area(i, k):
        a, b, q = poly[i], poly[(i + 1) % n], poly[k % n]
        return abs((b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]))

    for i in range(n):
        moved = 0
        while moved < n and area(i, j + 1) > area(i, j) + 1e-9:
            j = (j + 1) % n
            moved += 1
        js = [j]
        if abs(area(i, j + 1) - area(i, j)) <= 1e-9:
            js.append((j + 1) % n)
        for k in js:
            for ii in (i, (i + 1) % n):
                a, b = poly[ii], poly[k]
                d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                if d > best:
                    best, pair = d, (a, b)
    return math.sqrt(best), pair


def _line_meet(first, second, tol=1e-12):
    a, b, c = first
    d, e, f = second
    det = a * e - b * d
    if abs(det) <= tol:
        return None
    return ((c * e - b * f) / det, (a * f - c * d) / det)


def offset_outward(poly, delta):
    """Push every CCW edge outward by delta and re-intersect the shifted lines.

    Clipping alone can never enlarge a polygon: the polygon already satisfies the
    supporting half-plane of each of its own edges, so a shifted copy of that
    half-plane removes nothing and adds nothing. The vertices must instead be
    rebuilt from the shifted supporting lines (a miter offset), which is what
    this routine does. Nearly parallel neighbours make the miter unbounded, so
    the input polygon is returned unchanged in that case.
    """
    n = len(poly)
    if n < 3 or delta <= 0:
        return list(poly)
    lines = []
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length <= 1e-15:
            continue
        nx, ny = dy / length, -dx / length
        lines.append((nx, ny, nx * ax + ny * ay + delta))
    if len(lines) < 3:
        return list(poly)
    scale = max(span_scale(poly), 1.0)
    out = []
    for i in range(len(lines)):
        point = _line_meet(lines[i - 1], lines[i])
        if point is None or math.hypot(point[0], point[1]) > 1e6 * scale:
            return list(poly)
        out.append(point)
    return clean_polygon(out)


def _distance_to_chord(a, b, point):
    ex, ey = b[0] - a[0], b[1] - a[1]
    length = math.hypot(ex, ey)
    if length <= 1e-15:
        return math.dist(a, point)
    return abs(ex * (point[1] - a[1]) - ey * (point[0] - a[0])) / length


def simplify_polygon(poly, tol=0.02):
    """Merge consecutive vertices while the real deviation stays within tol.

    Removing a vertex whose distance to the chord of its *current* neighbours is
    small does not bound the final error: once neighbours are merged away, the
    surviving chord can be far from the skipped points. This routine merges a
    whole run at once and checks every skipped vertex against the retained
    chord, so the bound holds by construction. The result is then widened by the
    same tol, keeping the input polygon contained.
    """
    n = len(poly)
    if n < 5 or tol <= 0:
        return list(poly)
    cx = sum(p[0] for p in poly) / n
    cy = sum(p[1] for p in poly) / n
    start, best = 0, -1.0
    for i, (px, py) in enumerate(poly):
        distance = math.hypot(px - cx, py - cy)
        if distance > best:
            start, best = i, distance
    seq = poly[start:] + poly[:start] + [poly[start]]
    kept = []
    i = 0
    while i < len(seq) - 1:
        j = i + 1
        while j + 1 < len(seq):
            candidate = seq[j + 1]
            worst = max(_distance_to_chord(seq[i], candidate, seq[k])
                        for k in range(i + 1, j + 1))
            if worst <= tol:
                j += 1
            else:
                break
        kept.append(seq[i])
        i = j
    if len(kept) < 4:
        return list(poly)
    return offset_outward(kept, tol)


def bearing_interval(poly, station):
    """Continuous, unwrapped view-angle interval of a convex polygon outside P."""
    arr = np.asarray(poly, dtype=float) - np.asarray(station)
    ref = math.atan2(float(arr[:, 1].mean()), float(arr[:, 0].mean()))
    angles = np.arctan2(arr[:, 1], arr[:, 0])
    angles = ref + (angles - ref + math.pi) % (2 * math.pi) - math.pi
    lo, hi = float(angles.min()), float(angles.max())
    if hi - lo >= math.pi - 1e-10:
        raise ValueError("Station is inside/too close to the source polygon.")
    return lo, hi


def local_to_global(points, config):
    a = math.radians(config.first_bearing_deg)
    matrix = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return np.asarray(points, dtype=float) @ matrix.T + np.asarray(config.first_station_m)


def global_to_local(points, config):
    a = math.radians(config.first_bearing_deg)
    matrix = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (np.asarray(points, dtype=float) - np.asarray(config.first_station_m)) @ matrix
