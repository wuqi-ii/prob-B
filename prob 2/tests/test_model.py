import math
from dataclasses import replace
import numpy as np
import pytest
from bearing2.config import Config
from bearing2.region import (triangle, source_polygon, feasible_mask, physical_checks, uniform_diameter_bound)
from bearing2.risk import certify_risk, sampled_risk
from bearing2.geometry import clip_wedge, diameter_bruteforce
from bearing2.optimize import cost, station_grid


def test_uniform_bound_matches_derivation():
    assert uniform_diameter_bound(Config()) == pytest.approx(168.83479996318715)


@pytest.mark.parametrize("point", [(750,500),(750,-500),(666.8144759063354,526.8109671640481)])
def test_reception_and_crossing_guarantees(point):
    c=Config(); p=np.array(point); k=source_polygon(c)
    checks=physical_checks(p,k,c)
    assert checks['reception_guaranteed'] and checks['angle_guaranteed']
    assert checks['strong_signal_blind_zone_excluded']
    rng=np.random.default_rng(314159)
    r=rng.uniform(5.001,1500,2000); delta=rng.uniform(-c.eps,c.eps,2000)
    g=np.column_stack([r*np.cos(delta),r*np.sin(delta)])
    assert np.max(np.linalg.norm(g-p,axis=1))<=1000+1e-8
    for e in [-c.eps,c.eps]:
        bearings=np.arctan2(g[:,1]-p[1],g[:,0]-p[0])+e
        acute=np.rad2deg(np.arccos(np.clip(np.abs(np.cos(bearings)),0,1)))
        assert acute.min()>=30-1e-8


def test_bad_stations_rejected():
    assert not feasible_mask(np.array([0.,0.]),Config())
    assert not feasible_mask(np.array([750.,100.]),Config())
    assert not feasible_mask(np.array([1300.,500.]),Config())


def test_risk_certificate_brackets_independent_dense_bearings():
    c=Config(certificate_tolerance_m=0.08)
    k=source_polygon(c); p=np.array([750.,500.])
    result=certify_risk(p,k,c)
    dense,_=sampled_risk(p,k,c.eps,3001)
    assert result.converged
    assert dense<=result.upper_m+1e-7
    assert result.lower_m<=result.upper_m
    assert result.upper_m<=uniform_diameter_bound(c)+1e-6
    assert result.lower_m==pytest.approx(diameter_bruteforce(result.polygon)[0])


def test_random_true_sources_stay_in_updated_polygon_constraints():
    c=Config(); k=source_polygon(c); p=np.array([750.,500.]); rng=np.random.default_rng(72)
    # Each actual scenario's clipped-polygon diameter cannot exceed the universal bound.
    for _ in range(80):
        r=rng.uniform(6,1500); delta=rng.uniform(-c.eps,c.eps)
        g=r*np.array([math.cos(delta),math.sin(delta)])
        theta=math.atan2(*(g-p)[::-1])+rng.uniform(-c.eps,c.eps)
        poly=clip_wedge(k,p,theta,c.eps)
        assert poly
        assert diameter_bruteforce(poly)[0] <= uniform_diameter_bound(c)+1e-8


def test_map_boundary_shrinks_source_region_but_not_fixed_candidate_region():
    c=Config(first_station_m=(1200,0),first_bearing_deg=20)
    k=source_polygon(c); no_map=source_polygon(replace(c,use_map_boundary=False))
    assert max(x for x,y in k)<700
    assert max(x for x,y in no_map)>1490
    assert np.array_equal(station_grid(c),station_grid(replace(c,use_map_boundary=False)))


def test_upper_lower_lobe_symmetry():
    c=Config(); k=source_polygon(c)
    up,_=sampled_risk([750,500],k,c.eps,101)
    down,_=sampled_risk([750,-500],k,c.eps,101)
    assert up==pytest.approx(down,abs=1e-7)


def test_cost_units_and_tradeoff():
    c=Config()
    assert cost(900,120,5,c)==pytest.approx(305)
    assert cost(950,100,5,c)<cost(900,120,5,c)


def test_invalid_config_and_empty_map_region():
    with pytest.raises(ValueError): Config(weight_lambda=-1)
    with pytest.raises(ValueError): Config(name='../bad')
    with pytest.raises(ValueError): source_polygon(Config(first_station_m=(5000,0)))
