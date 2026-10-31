"""Kernel reference, independent batch cases and field consistency."""
import numpy as np
import pytest

from gratingbem import (Profile, Material, Incidence, Simulation, Numerics,
                        calculate, calculate_many, scan, scan_grid)
from gratingbem._core.ewald import EwaldHessian
from gratingbem._core.mesh_cache import clear_geometry_cache, cache_info, make_nodes_cached
from gratingbem._core.solver import make_profile_nodes


def flat(theta=12):
    return Simulation(Profile.flat(.6), 248.33, Incidence.angle(theta), 0, Material.aluminum())


def nums():
    return Numerics(p_order=4,max_panel_wavelengths=.5,modes_air=35,
                    modes_metal=10,oversample=10,max_nodes=200)


def test_green_kernel_matches_outgoing_fourier_solution():
    """Independent PDE reference, rather than equality to another Ewald path.

    Away from z=0 the Rayleigh series converges exponentially. Differentiate
    that series analytically to check the value and all five derivatives.
    """
    k0=2*np.pi/.24833
    alpha0=k0*np.sin(np.deg2rad(74.0))
    z=np.array([.08,-.12,.22])
    for period in (.6,1000/79):
        x=period*np.array([.013,.19,.49])
        for k in (k0,k0+2j,4.264229561097377+71.68977453877558j):
            alpha=alpha0+2*np.pi*np.arange(-3000,3001)/period
            beta=np.sqrt(k*k-alpha*alpha+0j)
            beta=np.where(beta.imag<0,-beta,beta)
            terms=(1j/(2*period))*np.exp(1j*(alpha[:,None]*x+
                    beta[:,None]*np.abs(z)))/beta[:,None]
            reference=(terms.sum(0),(1j*alpha[:,None]*terms).sum(0),
                (1j*beta[:,None]*np.sign(z)*terms).sum(0),
                (-alpha[:,None]**2*terms).sum(0),
                (-alpha[:,None]*beta[:,None]*np.sign(z)*terms).sum(0),
                (-beta[:,None]**2*terms).sum(0))
            actual=EwaldHessian(period,k,alpha0,modes=155).evaluate(x,z)
            for a,b in zip(reference,actual):
                np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-10)


def test_cache_only_geometry_not_angle_or_wavenumber():
    clear_geometry_cache()
    p1=flat(12).to_problem();p2=flat(14).to_problem()
    a=make_nodes_cached(p1,nums(),make_profile_nodes)
    b=make_nodes_cached(p2,nums(),make_profile_nodes)
    assert cache_info()['cached_meshes']==1
    assert a.spec.alpha0!=b.spec.alpha0
    np.testing.assert_array_equal(a.coords,b.coords)
    c=make_nodes_cached(p1.with_parameter('wavelength_um',.26),nums(),make_profile_nodes)
    assert cache_info()['cached_meshes']==2
    assert c.spec.k0!=b.spec.k0


def test_batch_parallel_matches_serial_complex_amplitudes_and_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sims=[flat(t) for t in (12,13,14)]
    serial=calculate_many(sims,nums(),workers=1,scan_workers=1)
    parallel=calculate_many(sims,nums(),workers=1,scan_workers=2)
    assert serial.errors==parallel.errors==[None]*3
    for x,y in zip(serial.results,parallel.results):
        np.testing.assert_array_equal(x.modes,y.modes)
        for pol in ('TE','TM'):
            np.testing.assert_allclose(x.amplitudes[pol],y.amplitudes[pol],atol=1e-12,rtol=1e-12)
    assert list(tmp_path.iterdir())==[]  # memory-only, no automatic output files


def test_nested_parallel_guard():
    with pytest.raises(ValueError,match='oversubscribe'):
        calculate_many([flat()],nums(),workers=2,scan_workers=2)


def test_batch_record_solver_failure():
    bad=Numerics(p_order=4,max_panel_wavelengths=.5,modes_air=35,
                 modes_metal=10,oversample=10,max_nodes=4)
    r=calculate_many([flat(12),flat(13)],bad,scan_workers=2,on_error='record')
    assert r.results==[None,None]
    assert all('exceed' in e for e in r.errors)


def test_scan_grid_records_invalid_input_in_original_order():
    r=scan_grid(flat(),{'incidence.angle_deg':[12,95,13]},nums(),
                scan_workers=2,on_error='record')
    assert len(r.results)==3 and r.results[1] is None
    assert 'angle' in r.errors[1]
    assert r.results[0].problem.incidence_deg==12
    assert r.results[2].problem.incidence_deg==13


def test_numeric_scan_compatibility():
    r=scan(flat(),'numerics.p_order',[3,4],nums(),workers=1)
    assert len(r.results)==2 and r.errors==[None,None]
    assert r.results[0].numerics.p_order==3


def test_field_poynting_flat_fresnel_consistency():
    result=calculate(flat(15),nums(),workers=1,energy_contour=True)
    for pol in ('TE','TM'):
        d=result.diagnostics[pol]
        assert 0 <= d['poynting_air_absorption'] <= 1
        assert abs(d['energy_closure_error']) < 1e-3
        assert d['poynting_offset_variation'] < 1e-3
        assert d['passes_passivity']
        assert result.timings['energy_diagnostic_s'] > 0
