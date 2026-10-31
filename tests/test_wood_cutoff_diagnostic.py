import math
import pytest
from gratingbem import Profile,Material,Incidence,Simulation,Numerics,calculate
from gratingbem._core.solver import Problem
from gratingbem._core.wood import rayleigh_cutoff_diagnostic


def test_79_littrow_reports_heuristic_wood_warning():
    sim=Simulation(Profile.echelle(1000/79,74,90),248.33,
                   Incidence.littrow(-98),-98,Material.aluminum())
    d=rayleigh_cutoff_diagnostic(sim.to_problem())
    assert d['status']=='near_cutoff_heuristic'
    assert d['nearest_mode']['order'] in (-100,2)
    assert 0.02 < d['min_abs_beta_over_k0'] < 0.05
    assert d['nearest_cutoff_angles'][0]['angular_distance_deg'] < 0.12


def test_regular_incidence_is_outside_heuristic_warning_band():
    sim=Simulation(Profile.flat(1.0),700,Incidence.angle(0),0,Material.aluminum())
    d=rayleigh_cutoff_diagnostic(sim.to_problem())
    assert d['status']=='outside_warning_band'
    assert d['min_abs_beta_over_k0'] > 0.1


def test_exact_rayleigh_cutoff_fails_before_matrix_assembly():
    theta=math.degrees(math.asin(0.3))  # lambda/d=0.7, m=+1 gives alpha_m/k0=1
    p=Problem(Profile.flat(1.0),0.7,theta_deg=theta,target_order=0)
    d=rayleigh_cutoff_diagnostic(p)
    assert d['status']=='at_or_numerically_on_cutoff'
    with pytest.raises(ValueError,match='Rayleigh/Wood anomaly'):
        calculate(p,Numerics(p_order=2,max_panel_wavelengths=.5,modes_air=12,
                             modes_metal=6,oversample=8,max_nodes=100))


def test_actual_79_calculation_emits_cutoff_warning():
    sim=Simulation(Profile.echelle(1000/79,74,90),248.33,
                   Incidence.littrow(-98),-98,Material.aluminum())
    num=Numerics(p_order=4,max_panel_wavelengths=1.5,modes_air=105,
                 modes_metal=24,oversample=8,max_nodes=400)
    r=calculate(sim,num,workers=1)
    assert r.diagnostics['wood_cutoff']['status']=='near_cutoff_heuristic'
    assert any('Near Rayleigh/Wood cutoff' in w for w in r.warning_messages)


def test_energy_warning_triggers_on_offset_variation(monkeypatch):
    from gratingbem._core import energy as energy_module
    sim=Simulation(Profile.flat(1.0),700,Incidence.angle(10),0,Material.constant(-8+1j))
    num=Numerics(p_order=2,max_panel_wavelengths=.5,modes_air=12,modes_metal=6,oversample=8,max_nodes=100)
    baseline=calculate(sim,num,polarizations=('TE','TM'))
    values=[]
    for pol in ('TE','TM'):
        values.extend([1.0-baseline.diagnostics[pol]['reflection_sum'],
                       1.02-baseline.diagnostics[pol]['reflection_sum']])
    iterator=iter(values)
    monkeypatch.setattr(energy_module,'air_contour_absorption',lambda *args,**kwargs: next(iterator))
    result=calculate(sim,num,polarizations=('TE','TM'),energy_contour=True)
    assert any('contour-offset variation above 1%' in w for w in result.warning_messages)


def test_energy_diagnostic_uses_independent_higher_order_target_rule(monkeypatch):
    from gratingbem._core import energy as energy_module
    sim=Simulation(Profile.flat(1.0),700,Incidence.angle(10),0,Material.constant(-8+1j))
    num=Numerics(p_order=4,max_panel_wavelengths=.5,modes_air=35,modes_metal=10,oversample=8,max_nodes=100)
    calls=[]
    def fake(*args,**kwargs):
        calls.append(kwargs.copy())
        return .1
    monkeypatch.setattr(energy_module,'air_contour_absorption',fake)
    result=calculate(sim,num,polarizations=('TE',),energy_contour=True)
    assert len(calls)==2
    assert all(x['target_order']==10 for x in calls)
    assert all(x['modes_air']==150 for x in calls)
    assert result.diagnostics['TE']['poynting_target_order']==10
    assert result.diagnostics['TE']['poynting_modes_air']==150
