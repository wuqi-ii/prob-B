"""Reproducible self-constructed tests, not official simulator observations."""
import json
import math
import random
from pathlib import Path
from q1_geometry import (bearing_constraints, intersection_reference, hpi_bounded,
                         convex_hull, diameter_brute, diameter_calipers,
                         diameter_disk_test)

ROOT = Path(__file__).resolve().parent


def run():
    rng = random.Random(20260911)
    sqrt3 = math.sqrt(3)
    cases = [
        ('two_stations', [(-1000, 0), (0, -1000)], [0, 90]),
        ('triangle_counterexample', [(-1000, 0), (530, -500*sqrt3),
                                    (515, 515*sqrt3)], [1, 121, 241]),
        ('six_stations', [(1000*math.cos(k*math.pi/3),
                           1000*math.sin(k*math.pi/3)) for k in range(6)],
         [(180+60*k) % 360 for k in range(6)]),
        ('one_station_unbounded', [(-1000, 0)], [0]),
        ('inconsistent_bearings', [(0, 0), (-100, 0)], [0, 180]),
    ]
    report = {'seed': 20260911, 'data_origin': 'self-constructed; not official data',
              'cases': [], 'random_validation': {}}
    for name, stations, angles in cases:
        lines = bearing_constraints(stations, angles)
        status, vertices = intersection_reference(lines)
        row = {'name': name, 'stations': stations, 'bearings_deg': angles,
               'status': status, 'vertices': vertices}
        if status == 'polygon':
            fast = hpi_bounded(lines)
            d, a, b = diameter_calipers(fast)
            db, _, _ = diameter_brute(vertices)
            assert abs(d-db) < 1e-6, (name, d, db)
            row.update({'vertex_count': len(fast), 'diameter_m': d,
                        'diameter_pair': [a, b], 'disk': diameter_disk_test(fast)})
        report['cases'].append(row)
    assert report['cases'][1]['vertex_count'] == 3
    assert abs(report['cases'][1]['diameter_m'] - 30) < 1e-6
    assert report['cases'][1]['disk']['covers'] is False
    assert report['cases'][2]['vertex_count'] > 4
    assert report['cases'][3]['status'] == 'unbounded'
    assert report['cases'][4]['status'] == 'empty'

    edge_cases = {
        'point': [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)],
        'segment': [(1, 0, 0), (-1, 0, -2), (0, 1, 0), (0, -1, 0)],
        'unbounded': [(1, 0, 0), (-1, 0, -2)],
        'empty': [(1, 0, 1), (-1, 0, 0)],
    }
    for expected, lines in edge_cases.items():
        status, vertices = intersection_reference(lines)
        assert status == expected, (expected, status)
        if status in ('point', 'segment'):
            assert diameter_disk_test(vertices)['covers']
    report['special_cases_passed'] = list(edge_cases)

    max_error, max_vertex_error = 0.0, 0.0
    for k in range(500):
        n = rng.randint(3, 16)
        target = (rng.uniform(-200, 200), rng.uniform(-200, 200))
        phase = rng.random()*2*math.pi
        stations, angles = [], []
        for i in range(n):
            t = phase + 2*math.pi*i/n
            radius = rng.uniform(700, 1200)
            p = (target[0]+radius*math.cos(t), target[1]+radius*math.sin(t))
            stations.append(p)
            theta = math.degrees(math.atan2(target[1]-p[1], target[0]-p[0]))
            angles.append((theta+rng.uniform(-0.8, 0.8)) % 360)
        lines = bearing_constraints(stations, angles)
        status, ref = intersection_reference(lines)
        assert status == 'polygon'
        fast = hpi_bounded(lines)
        vertex_error = max(max(min(math.dist(p, q) for q in ref) for p in fast),
                           max(min(math.dist(p, q) for q in fast) for p in ref))
        max_vertex_error = max(max_vertex_error, vertex_error)
        assert vertex_error < 1e-6
        error = abs(diameter_brute(ref)[0] - diameter_calipers(fast)[0])
        max_error = max(max_error, error)
        assert error < 1e-6
    report['random_validation']['angular_cases'] = 500
    report['random_validation']['max_vertex_set_distance_m'] = max_vertex_error
    report['random_validation']['max_diameter_difference_m'] = max_error

    caliper_error = 0.0
    for k in range(1000):
        if k < 100:
            n = k+3
            points = [(math.cos(2*math.pi*i/n), math.sin(2*math.pi*i/n))
                      for i in range(n)]
        else:
            points = [(rng.uniform(-100, 100), rng.uniform(-100, 100))
                      for _ in range(rng.randint(3, 100))]
        vertices = convex_hull(points)
        error = abs(diameter_brute(vertices)[0] - diameter_calipers(vertices)[0])
        caliper_error = max(caliper_error, error)
        assert error < 1e-7
    report['random_validation']['convex_polygon_cases'] = 1000
    report['random_validation']['max_caliper_difference_m'] = caliper_error

    # Similarity invariance catches unit, origin and angle-handling errors.
    base = report['cases'][0]
    translated = [(3*x+120, 3*y-85) for x, y in base['stations']]
    status, vs = intersection_reference(bearing_constraints(translated, base['bearings_deg']))
    assert abs(diameter_calipers(vs)[0] - 3*base['diameter_m']) < 1e-6
    report['similarity_invariance_passed'] = True
    (ROOT/'validation_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
