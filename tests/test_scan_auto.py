"""Automatic Bloch channel grouping, multi RHS and parallel groups."""
from __future__ import annotations
import math
import numpy as np
import pytest
from gratingbem import (Profile, Material, Incidence, Simulation, Numerics,
                        calculate_many, scan, scan_grid)
from gratingbem._scan_bloch import groups

D=.8
WAVELENGTH=248.33

def shift(t,q=1,wavelength_nm=WAVELENGTH):
    sine=math.sin(math.radians(t))+q*(wavelength_nm*.001)/D
    if abs(sine)>=1:raise ValueError('not propagating')
    return math.degrees(math.asin(sine))

def case(theta=10.,wavelength=WAVELENGTH,profile=None,target=0):
    return Simulation(profile or Profile.flat(D), wavelength, Incidence.angle(theta),
                      target,Material.constant(-8+.95j))

def num(modes=155):
    return Numerics(p_order=2,max_panel_wavelengths=2.,modes_air=modes,
                    modes_metal=25,oversample=10,max_nodes=300)

def check_parity(baseline,test,atol=2e-10):
    assert baseline.errors==test.errors
    for first,second in zip(baseline.results,test.results):
        if first is None:
            assert second is None
            continue
        assert first.problem.as_dict()==second.problem.as_dict()
        np.testing.assert_array_equal(first.modes,second.modes)
        for pol in ('TE','TM'):
            np.testing.assert_allclose(first.amplitudes[pol],second.amplitudes[pol],atol=atol,rtol=atol)
            np.testing.assert_allclose(first.efficiencies[pol],second.efficiencies[pol],atol=atol,rtol=atol)

def test_auto_default_groups_nonadjacent_aliases_and_solves_multi_rhs():
    s=[case(10),case(12),case(shift(10)),case(10),case(shift(12))]
    assert groups(s)==[[0,2,3],[1,4]]
    baseline=calculate_many(s,num(),bloch_reuse=False)
    auto=calculate_many(s,num())
    check_parity(baseline,auto)
    assert len(auto.group_stats)==2
    assert sorted(x['status'] for x in auto.group_stats)==['reused','reused']
    assert all(x['multi_rhs'] for x in (r.diagnostics['bloch_group_reuse'] for r in auto.results))
    assert all(r.diagnostics['bloch_group_reuse']['rhs_batches']==2 for r in auto.results)

def test_auto_parallel_groups_process_and_host_progress():
    s=[case(10),case(12),case(shift(10)),case(14),case(shift(12))]
    baseline=calculate_many(s,num(),bloch_reuse=False)
    updates=[]
    out=calculate_many(s,num(),scan_workers=2,scan_backend='process',
                       progress=lambda done,total,index,res,error: updates.append((done,total,index,error)))
    check_parity(baseline,out)
    assert {x['status'] for x in out.group_stats}=={'reused','independent'}
    assert sorted(i for _,_,i,_ in updates)==list(range(len(s)))
    assert sorted(d for d,_,_,_ in updates)==list(range(1,len(s)+1))

def test_auto_negative_low_modes_fallback_even_parallel():
    # A deep echelle with a long period is used as a negative control;
    # 35 modes on a short, shallow flat period might already suffice.
    p=Profile.echelle(1000/79,74.0,90)
    d=p.period_um
    theta0=72.
    theta1=math.degrees(math.asin(math.sin(math.radians(theta0))+.24833/d))
    s=[case(theta0,profile=p),case(theta1,profile=p)]
    n=Numerics(p_order=2,max_panel_wavelengths=3.,modes_air=35,
               modes_metal=10,oversample=8,max_nodes=300)
    baseline=calculate_many(s,n,bloch_reuse=False)
    out=calculate_many(s,n,scan_workers=2,scan_backend='process')
    check_parity(baseline,out)
    assert out.group_stats[0]['status']=='fallback'
    assert all('bloch_group_reuse' not in r.diagnostics for r in out.results)
    assert all('Bloch reuse refused' in ' '.join(r.warning_messages) for r in out.results)

def test_auto_no_wrong_reuse_wavelength_or_geometry_and_per_case_error():
    s=[case(10),case(shift(10)),case(10,wavelength=250.),
       case(10,profile=Profile.echelle(.8,30,90))]
    out=calculate_many(s,num(),scan_workers=2,scan_backend='process',on_error='record')
    assert out.errors==[None]*len(s)
    assert groups(s)==[[0,1],[2],[3]]
    assert len([x for x in out.group_stats if x['status']=='reused'])==1
    # Only the geometrically complex third case must fail the node budget.
    budget=Numerics(p_order=2,max_panel_wavelengths=1.0,modes_air=155,
                    modes_metal=25,oversample=10,max_nodes=80)
    s=[case(10),case(shift(10)),case(10,
                 profile=Profile.echelle(1000/79,74.0,90))]
    out=calculate_many(s,budget,on_error='record',scan_workers=2,scan_backend='process')
    assert out.errors[0] is None and out.errors[1] is None
    assert out.results[0] is not None and out.results[1] is not None
    assert out.results[2] is None and out.errors[2] is not None
    assert 'exceed' in out.errors[2]

def test_single_axis_and_grid_auto_default_and_off():
    c=case(10)
    a=scan(c,'incidence.angle_deg',[10,shift(10),12],num())
    b=scan(c,'incidence.angle_deg',[10,shift(10),12],num(),bloch_reuse=False)
    assert a.errors==b.errors
    assert all(r is not None for r in a.results)
    assert a.results[0].diagnostics['bloch_group_reuse']['size']==2
    for r1,r2 in zip(a.results,b.results):
        for pol in ('TE','TM'):
            np.testing.assert_allclose(r1.amplitudes[pol],r2.amplitudes[pol],rtol=2e-10,atol=2e-10)
    # Wavelength and profile changes create distinct physical operator groups.
    g=scan_grid(c,{'wavelength_nm':[WAVELENGTH,250.],
                   'incidence.angle_deg':[10,shift(10)]},num(),
                scan_workers=2,scan_backend='process')
    assert len(g.results)==4 and g.errors==[None]*4
    assert all(r is not None for r in g.results)

def test_chunked_multi_rhs_and_two_pol_with_repeated_channels(monkeypatch):
    import gratingbem._scan_bloch as bm
    monkeypatch.setattr(bm,'MAX_RHS_COLUMNS',2)
    s=[case(10),case(shift(10)),case(10)]
    r=calculate_many(s,num())
    assert all(x.diagnostics['bloch_group_reuse']['rhs_batches']==4 for x in r.results)
    b=calculate_many(s,num(),bloch_reuse=False)
    check_parity(b,r)

def test_invalid_options_and_non_nesting():
    s=[case(10),case(shift(10))]
    with pytest.raises(ValueError,match='bloch_reuse'):
        calculate_many(s,num(),bloch_reuse='maybe')
    with pytest.raises(ValueError,match='oversubscribe'):
        calculate_many(s,num(),scan_workers=2,workers=2,scan_backend='process')
    off=calculate_many(s,num(),bloch_reuse='off')
    assert all('bloch_group_reuse' not in x.diagnostics for x in off.results)


def test_fast_indexed_grouping_many_wavelengths_preserves_stable_order():
    # 100 distinct wavelengths x 3 angles; each wavelength has exactly one
    # alias pair and one independent incidence channel.
    s=[]
    for i in range(100):
        wl=245.+i*.05
        s.extend([case(10,wavelength=wl),
                  case(12,wavelength=wl),
                  case(shift(10,wavelength_nm=wl),wavelength=wl)])
    blocks=groups(s)
    assert len(blocks)==200
    assert blocks[0]==[0,2] and blocks[1]==[1]
    assert blocks[-2]==[297,299] and blocks[-1]==[298]


def test_group_index_near_integer_bloch_wrap():
    c=case(0.)
    aliases=[c,case(shift(0,q=1)),case(shift(0,q=-1))]
    assert groups(aliases)==[[0,1,2]]

def test_79_near_wood_alias_group_is_not_shared():
    from gratingbem._scan_bloch import shareable
    p=Profile.echelle(1000/79.,74.,90)
    lam=.24833
    t0=math.degrees(math.asin(98*lam/(2*p.period_um)))
    t1=math.degrees(math.asin(math.sin(math.radians(t0))+lam/p.period_um))
    a=case(t0,profile=p)
    b=case(t1,profile=p)
    assert groups([a,b])==[[0,1]]
    can_share,err=shareable([a,b],num())
    assert not can_share
