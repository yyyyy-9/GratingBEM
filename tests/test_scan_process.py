"""Opt-in spawn-process scan scheduler regression; no changes to KM1."""
from __future__ import annotations
import numpy as np
import pytest
from gratingbem import (Profile,Material,Incidence,Simulation,Numerics,
                        calculate_many,scan,scan_grid)


def profile():return Profile.flat(.6)

def case(theta=12., wavelength_nm=248.33):
    return Simulation(profile(),wavelength_nm,Incidence.angle(theta),0,Material.aluminum())

def numerics():
    return Numerics(p_order=4,max_panel_wavelengths=.5,modes_air=35,
                    modes_metal=10,oversample=10,max_nodes=200)

def same_results(serial,other):
    assert serial.errors==other.errors
    for x,y in zip(serial.results,other.results):
        if x is None:
            assert y is None
            continue
        assert x.problem.as_dict()==y.problem.as_dict()
        np.testing.assert_array_equal(x.modes,y.modes)
        for pol in ('TE','TM'):
            np.testing.assert_allclose(x.amplitudes[pol],y.amplitudes[pol],atol=2e-12,rtol=2e-12)
            np.testing.assert_allclose(x.efficiencies[pol],y.efficiencies[pol],atol=2e-12,rtol=2e-12)


def test_process_many_and_input_order_no_files(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    sims=[case(t) for t in [15.,12.,14.]]
    ref=calculate_many(sims,numerics(),scan_workers=1)
    done=[]
    got=calculate_many(sims,numerics(),scan_workers=2,scan_backend='process',
                       progress=lambda count,total,index,result,error:done.append((count,total,index,error)))
    same_results(ref,got)
    assert sorted(k for _,_,k,_ in done)==[0,1,2]
    assert sorted(d for d,_,_,_ in done)==[1,2,3]
    assert list(tmp_path.iterdir())==[]


def test_process_grid_three_axes_preserves_order_and_records_bad_case():
    # Test wavelength, angle and blaze axes with an actual echelle.
    triangle=Simulation(Profile.echelle(1000/79,74.0,90),248.33,
                        Incidence.angle(74.0),-98,Material.aluminum())
    n=Numerics(p_order=2,max_panel_wavelengths=4,modes_air=40,
               modes_metal=12,oversample=8,max_nodes=200)
    a={'wavelength_nm':[248.33], 'incidence.angle_deg':[73.9,95.,74.1],
       'profile.blaze_deg':[74.0,74.20]}
    expected=scan_grid(triangle,a,n,on_error='record')
    got=scan_grid(triangle,a,n,scan_backend='process',scan_workers=2,on_error='record')
    same_results(expected,got)
    assert len(got.results)==6
    assert got.errors[2] is not None and got.errors[3] is not None
    assert got.results[0].problem.profile.as_dict()!=got.results[1].problem.profile.as_dict()


def test_process_scan_one_parameter_and_error_isolation():
    ref=scan(case(),'incidence.angle_deg',[12.,13.,14.],numerics())
    got=scan(case(),'incidence.angle_deg',[12.,13.,14.],numerics(),scan_workers=2,
             scan_backend='process',on_error='record')
    same_results(ref,got)
    bad=Numerics(p_order=4,max_panel_wavelengths=.5,modes_air=35,
                 modes_metal=10,oversample=10,max_nodes=4)
    errors=scan(case(),'incidence.angle_deg',[12.,13.],bad,scan_workers=2,
                scan_backend='process',on_error='record')
    assert all(x is None for x in errors.results)
    assert all('exceed' in x for x in errors.errors)


def test_process_input_guards():
    with pytest.raises(ValueError,match='oversubscribe'):
        calculate_many([case()],numerics(),scan_workers=2,workers=2,scan_backend='process')
    with pytest.raises(ValueError,match='scan_backend'):
        calculate_many([case()],numerics(),scan_workers=2,scan_backend='invalid')
    with pytest.raises(ValueError,match='oversubscribe'):
        scan_grid(case(),{'incidence.angle_deg':[12.,13.]},numerics(),scan_workers=2,
                  workers=2,scan_backend='process')
