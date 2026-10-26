from __future__ import annotations
import json
import numpy as np
import pytest
from gratingbem import (Profile, Material, Incidence, Simulation, Numerics,
                        calculate,scan,scan_grid,efficiency)


def flat_sim():
    return Simulation(Profile.flat(.6),248.33,Incidence.angle(15),0,Material.aluminum())


def fast_num():
    return Numerics(p_order=4,max_panel_wavelengths=.5,modes_air=45,
                    modes_metal=12,oversample=10,max_nodes=200)


def test_public_names_and_units():
    sim=Simulation(Profile.echelle(1000/79,74,90),248.33,Incidence.littrow(-98),-98)
    assert abs(sim.incidence_deg-74.0049342579)<1e-8
    assert abs(sim.to_problem().wavelength_um-.24833)<1e-12
    assert sim.to_problem().epsilon_value.imag>0
    assert sim.material.model=='aluminum_rakic_ld'


def test_invalid_inputs():
    with pytest.raises(ValueError): Material.constant(-8-1j)
    with pytest.raises(ValueError): Incidence(angle_deg=10,littrow_order=-1)
    with pytest.raises(ValueError): Simulation(Profile.flat(1),248.33,Incidence.littrow(-100),-100)
    with pytest.raises(NotImplementedError): Simulation.from_dict({'profile':{},'layers':[1]})


def test_profiles_and_shapes(tmp_path):
    shapes=(Profile.rectangle(.8,.1,corner_radius_um=.005),
            Profile.trapezoid(.8,.1),Profile.staircase(1.,.15,steps=3),
            Profile.echelle(1000/79,74,90,tip_radius_um=.02,valley_radius_um=.05))
    assert shapes[0].has_arcs and shapes[-1].has_arcs
    assert all(p.lengths_um.sum()>0 for p in shapes)
    path=tmp_path/'s.svg';shapes[-1].save_svg(path)
    assert '<svg' in path.read_text()


@pytest.mark.parametrize('angle,epsilon', [(15,None),(-15,None),(0,2.25)])
def test_flat_fresnel_and_save(tmp_path,angle,epsilon):
    sim=Simulation(Profile.flat(.6),248.33,Incidence.angle(angle),0,
                   Material.aluminum() if epsilon is None else Material.constant(epsilon))
    result=calculate(sim,fast_num(),workers=1)
    assert set(result.efficiencies)=={'TE','TM'}
    assert all(result.diagnostics[pol]['passes_passivity'] for pol in ('TE','TM'))
    theta=np.deg2rad(sim.incidence_deg)
    eps=sim.to_problem().epsilon_value
    a=np.cos(theta);b=np.sqrt(eps+0j-np.sin(theta)**2)
    r_te=(a-b)/(a+b)
    r_tm=(eps*a-b)/(eps*a+b)
    assert abs(result.diagnostics['TE']['reflection_sum']-abs(r_te)**2)<2e-3
    assert abs(result.diagnostics['TM']['reflection_sum']-abs(r_tm)**2)<2e-3
    assert efficiency(result,0,'TE')==pytest.approx(100*result.diagnostics['TE']['target_efficiency'])
    result.save(tmp_path/'r')
    report=json.loads((tmp_path/'r/result.json').read_text())
    assert len(report['orders'])>0
    assert (tmp_path/'r/orders.csv').exists()


def test_scan_uses_nm_and_writes_csv(tmp_path):
    sim=flat_sim()
    r=scan(sim,'wavelength_nm',[248.33,250],fast_num(),workers=1)
    assert len(r.results)==2 and all(v is not None for v in r.results)
    assert r.results[1].problem.wavelength_um==pytest.approx(.250)
    r.save(tmp_path/'scan')
    assert (tmp_path/'scan/scan.csv').exists()


def test_grid_scan_polarizations_and_material(tmp_path):
    sim=flat_sim()
    r=scan_grid(sim,{'wavelength_nm':[248.33,250],'target_order':[0]},fast_num(),polarizations=('TE',))
    assert len(r.results)==2 and all(set(v.efficiencies)=={'TE'} for v in r.results)


def test_old_problem_roundtrip():
    original=flat_sim().to_problem()
    rebuilt=Simulation.from_legacy_problem(original)
    assert rebuilt.incidence_deg==pytest.approx(original.incidence_deg)
    assert rebuilt.to_problem().epsilon_value==original.epsilon_value


def test_internal_legacy_version_not_exposed():
    import gratingbem
    assert not any(x.startswith('echelle_bie_v') for x in gratingbem.__all__)
    assert not hasattr(gratingbem,'Coating')


@pytest.mark.parametrize('make_invalid', [
    lambda: Incidence.littrow(-1.9),
    lambda: Incidence.from_dict({'littrow_order':-1.9}),
    lambda: Simulation(Profile.flat(.6),248.33,Incidence.angle(15),.9),
    lambda: Simulation.from_dict({**flat_sim().as_dict(),'target_order':.9}),
    lambda: flat_sim().with_parameter('target_order',.9),
    lambda: Incidence.from_dict({'angle_deg':10,'littrow_order':-1}),
    lambda: Incidence.from_dict({'angle_deg':10,'theta_deg':20}),
    lambda: Material.from_dict({'model':'aluminum_rakic_ld','epsilon_real':-8,'epsilon_imag':1}),
    lambda: Material.from_dict({'epsilon_imag':1}),
])
def test_ambiguous_or_fractional_physical_inputs_are_rejected(make_invalid):
    """RELEASE-01: malformed input must not silently select a different problem."""
    with pytest.raises(ValueError):
        make_invalid()


def test_efficiency_does_not_truncate_a_fractional_order():
    from types import SimpleNamespace
    result=SimpleNamespace(modes=np.array([0]),efficiencies={'TE':np.array([.5])})
    with pytest.raises(ValueError):
        efficiency(result,.9)


def test_grid_scan_rejects_unknown_error_policy():
    with pytest.raises(ValueError,match='on_error'):
        scan_grid(flat_sim(),{'wavelength_nm':[-1]},fast_num(),on_error='typo')


@pytest.mark.parametrize('order',[-1,1])
def test_signed_littrow_order_is_retroreflected(order):
    """kx(m)=kx(incident)+2*pi*m/d must equal -kx(incident) at Littrow."""
    problem=Simulation(Profile.flat(.6),248.33,Incidence.littrow(order),order).to_problem()
    kx=problem.spec().alpha0
    assert kx+2*np.pi*order/problem.profile.period_um==pytest.approx(-kx,abs=1e-12)


def test_curved_parallel_matches_serial():
    """Exercise the actual curved-row worker on spawn as well as serial assembly."""
    sim=Simulation(Profile.rectangle(.6,.05,corner_radius_um=.005),248.33,
                   Incidence.angle(10),0,Material.aluminum())
    serial=calculate(sim,fast_num(),workers=1)
    parallel=calculate(sim,fast_num(),workers=2)
    np.testing.assert_array_equal(serial.modes,parallel.modes)
    for pol in ('TE','TM'):
        np.testing.assert_allclose(parallel.amplitudes[pol],serial.amplitudes[pol],rtol=1e-11,atol=1e-12)
