"""Q1 geometry. Standard library only; angles in degrees, distances in metres.

hpi_bounded: fast kernel for a bounded, nonempty, two-dimensional intersection.
intersection_reference: independent O(N^3) reference and status classifier.
Floating-point tolerances are engineering tolerances, not certified exact bounds.
"""
import math
from collections import deque
from itertools import combinations

DIST_TOL = 1e-7
ANGLE_TOL = 1e-12


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def sub(a, b):
    return a[0] - b[0], a[1] - b[1]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def bearing_constraints(stations, bearings, error_deg=1.0):
    if len(stations) != len(bearings) or not stations:
        raise ValueError('Require equal, nonzero numbers of stations and bearings')
    if not 0 < error_deg < 90:
        raise ValueError('Require 0 < error_deg < 90')
    lines = []
    for (x, y), theta in zip(stations, bearings):
        lo, hi = map(math.radians, (theta - error_deg, theta + error_deg))
        for a, b in ((-math.sin(lo), math.cos(lo)),
                     (math.sin(hi), -math.cos(hi))):
            lines.append((a, b, a*x + b*y))
    return lines


def normalize(lines):
    out = []
    for a, b, c in lines:
        s = math.hypot(a, b)
        if s == 0:
            raise ValueError('Zero normal')
        out.append((a/s, b/s, c/s))
    return out


def line_intersection(l, k):
    a, b, c = l
    d, e, f = k
    det = a*e - b*d
    if abs(det) <= ANGLE_TOL:
        return None
    return ((c*e - b*f)/det, (a*f - c*d)/det)


def inside(p, line):
    a, b, c = line
    return a*p[0] + b*p[1] >= c - DIST_TOL


def convex_hull(points):
    ordered = sorted(set(points))
    ps = []
    for p in ordered:
        if not ps or math.dist(p, ps[-1]) > DIST_TOL:
            ps.append(p)
    if len(ps) <= 1:
        return ps
    def chain(seq):
        out = []
        for p in seq:
            while len(out) >= 2:
                u, v = sub(out[-1], out[-2]), sub(p, out[-2])
                if cross(u, v) > ANGLE_TOL * max(1.0, math.hypot(*u)*math.hypot(*v)):
                    break
                out.pop()
            out.append(p)
        return out
    return chain(ps)[:-1] + chain(reversed(ps))[:-1]


def intersection_reference(lines):
    """Return ('empty'|'unbounded'|'point'|'segment'|'polygon', vertices).

    Feasibility witnesses include origin, projections onto individual lines and
    pairwise line intersections: the closest point to the origin must be one
    of these types. A nontrivial recession cone contains a boundary ray parallel
    to a constraint line (unless there are no constraints).
    """
    lines = normalize(lines)
    if not lines:
        return 'unbounded', []
    candidates = [(0.0, 0.0)] + [(a*c, b*c) for a, b, c in lines]
    vertices = []
    for l, k in combinations(lines, 2):
        p = line_intersection(l, k)
        if p is not None and all(inside(p, h) for h in lines):
            vertices.append(p)
    candidates.extend(vertices)
    if not any(all(inside(p, h) for h in lines) for p in candidates):
        return 'empty', []
    for a, b, _ in lines:
        for dx, dy in ((b, -a), (-b, a)):
            if all(aa*dx + bb*dy >= -ANGLE_TOL for aa, bb, _ in lines):
                return 'unbounded', []
    hull = convex_hull(vertices)
    if not hull:
        raise ArithmeticError('Bounded intersection but no vertex: check scaling')
    return {1: 'point', 2: 'segment'}.get(len(hull), 'polygon'), hull


def hpi_bounded(lines):
    """Angle-sort + deque, O(N log N), for a bounded full-dimensional P.

    Preconditions must be established by application geometry or separately.
    Degenerate/ill-conditioned inputs should use the reference routine.
    No artificial bounding box is added.
    """
    lines = normalize(lines)
    lines.sort(key=lambda h: math.atan2(-h[0], h[1]) % (2*math.pi))
    unique = []
    for h in lines:
        if unique and abs(cross(h, unique[-1])) <= ANGLE_TOL and dot(h, unique[-1]) > 0:
            if h[2] > unique[-1][2]:
                unique[-1] = h
        else:
            unique.append(h)
    if len(unique) > 1 and abs(cross(unique[0], unique[-1])) <= ANGLE_TOL and dot(unique[0], unique[-1]) > 0:
        if unique[-1][2] > unique[0][2]:
            unique[0] = unique[-1]
        unique.pop()
    q = deque()
    def meet(l, k):
        p = line_intersection(l, k)
        if p is None:
            raise ArithmeticError('Parallel active lines: use reference classifier')
        return p
    for h in unique:
        while len(q) >= 2 and not inside(meet(q[-2], q[-1]), h):
            q.pop()
        while len(q) >= 2 and not inside(meet(q[0], q[1]), h):
            q.popleft()
        q.append(h)
    while len(q) >= 3 and not inside(meet(q[-2], q[-1]), q[0]):
        q.pop()
    while len(q) >= 3 and not inside(meet(q[0], q[1]), q[-1]):
        q.popleft()
    if len(q) < 3:
        raise ArithmeticError('No full-dimensional bounded polygon')
    ls = list(q)
    vertices = [meet(ls[i], ls[(i+1) % len(ls)]) for i in range(len(ls))]
    # Remove exact duplicate consecutive intersections, preserving cyclic order.
    cleaned = []
    for p in vertices:
        if not cleaned or math.dist(cleaned[-1], p) > DIST_TOL:
            cleaned.append(p)
    if len(cleaned) > 1 and math.dist(cleaned[0], cleaned[-1]) <= DIST_TOL:
        cleaned.pop()
    if len(cleaned) < 3:
        raise ArithmeticError('Degenerate polygon: use reference routine')
    return cleaned


def diameter_brute(vertices):
    if not vertices:
        raise ValueError('Empty vertex set')
    return max((math.dist(a, b), a, b) for a in vertices for b in vertices)


def diameter_calipers(vertices):
    """O(m) for CCW convex vertices, without redundant collinear vertices."""
    m = len(vertices)
    if m <= 2:
        return diameter_brute(vertices)
    best2, ends = -1.0, None
    def update(i, j):
        nonlocal best2, ends
        delta = sub(vertices[i % m], vertices[j % m])
        d2 = dot(delta, delta)
        if d2 > best2:
            best2, ends = d2, (vertices[i % m], vertices[j % m])
    j = 1
    for i in range(m):
        edge = sub(vertices[(i+1) % m], vertices[i])
        def area(k):
            return cross(edge, sub(vertices[k % m], vertices[i]))
        # Unwrapped j preserves monotonicity and prevents full-cycle loops.
        while j + 1 < i + m and area(j+1) > area(j):
            j += 1
        update(i, j)
        update(i+1, j)
        if math.isclose(area(j+1), area(j), rel_tol=1e-12, abs_tol=1e-12):
            update(i, j+1)
            update(i+1, j+1)
    return math.sqrt(best2), ends[0], ends[1]


def diameter_disk_test(vertices, result=None):
    d, a, b = result or diameter_calipers(vertices)
    center = ((a[0]+b[0])/2, (a[1]+b[1])/2)
    max_distance = max(math.dist(p, center) for p in vertices)
    return {'covers': max_distance <= d/2 + DIST_TOL,
            'center': center, 'radius': d/2,
            'max_vertex_distance': max_distance,
            'margin': d/2 - max_distance}
