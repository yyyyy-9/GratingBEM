"""Checks for the opt-in experimental coating branch."""
import json
import numpy as np
import pytest
from gratingbem import Profile, Simulation, Material, Incidence, Numerics, calculate as bare_calculate
from gratingbem.coating import (
    Film, CoatedSimulation, calculate, planar_reflection, scan_thickness,
    offset_profile,
)
from gratingbem.coating.geometry import validate_interface_order
from gratingbem.coating.planar import planar_stack_r, solve_planar_bie


def flat(ef=2.25, thickness=70.):
    return CoatedSimulation(
        Simulation(Profile.flat(1.2), 800, Incidence.angle(17), 0,
                   Material.constant(-8+.95j)), Film(thickness,ef))


def flat_numerics():
    return Numerics(p_order=4, panels_per_edge=(4,), modes_air=40,
                    modes_metal=8, oversample=10)


def test_film_units_and_passivity():
    f=Film.from_index(12.5,1.5)
    assert f.epsilon==pytest.approx(2.25)
    assert Film.from_dict(f.as_dict())==f
    for thickness in (0,-5,np.nan):
        with pytest.raises(ValueError):Film(thickness,2.3)
    for eps in (0,-2j, np.nan):
        with pytest.raises(ValueError):Film(10,eps)


def test_fresnel_planar_bie_24():
    for pol in ('TE','TM'):
        for angle in (0,40,74):
            for t in (3.,10.,25.,55.):
                ref=planar_reflection(248.33,angle,2.8,-8+.955j,t,pol)
                got=solve_planar_bie(.24833,angle,2.8,-8+.955j,t/1000,pol)['r_top']
                assert abs(ref-got)<2e-12


def test_absorbing_planar_stack_and_zero_thickness():
    for pol in ('TE','TM'):
        a=planar_reflection(248.33,63,2.4+.03j,-8+.955j,12,pol)
        b=planar_stack_r(.24833,63,[2.4+.03j],[.012],-8+.955j,pol)
        assert abs(a-b)<1e-12
        base=planar_stack_r(.24833,63,[],[],-8+.955j,pol)
        assert abs(planar_reflection(248.33,63,2.4,-8+.955j,0,pol)-base)<1e-12


def test_offset_analytic_and_periodic_seam():
    profile=Profile.echelle(1000/79,74,90,tip_radius_um=.02,valley_radius_um=.05)
    outer=offset_profile(profile,.01)
    assert outer.period_um==pytest.approx(profile.period_um)
    assert any(s.__class__.__name__=='Arc' for s in outer.geometric_segments)
    assert outer.geometric_segments[0].start[0]==pytest.approx(0.,abs=1e-9)
    assert outer.geometric_segments[-1].end[0]==pytest.approx(profile.period_um,abs=1e-9)
    assert validate_interface_order(profile,outer)>0
    with pytest.raises(ValueError,match='thicker|singular'):
        offset_profile(profile,.051)
    with pytest.raises(ValueError,match='non-tangent'):
        offset_profile(Profile.rectangle(1.2,.10),.01)


def test_explicit_crossing_rejected():
    s=flat()
    bad=CoatedSimulation(s.substrate,s.film,outer_profile=Profile.flat(1.2,z_um=-.01))
    with pytest.raises(ValueError,match='touch/cross'):
        calculate(bad,flat_numerics())


def test_periodic_flat_real_and_absorbing():
    for ef in (2.25, 2.25+.09j):
        simulation=flat(ef)
        result=calculate(simulation,flat_numerics())
        assert result.status=='passed_energy_only_mesh_unverified'
        for pol in ('TE','TM'):
            ref=planar_reflection(800,17,ef,-8+.95j,70,pol)
            assert abs(result.efficiency(0,pol,percent=False)-abs(ref)**2)<2e-5
            d=result.raw['polarizations'][pol]
            assert abs(d['energy_balance_error'])<5e-5
            assert d['reflection_sum']<1
            if ef.imag:
                assert d['film_absorption_flux']>0


def test_save_and_thickness_scan(tmp_path):
    s=flat(2.25,70)
    out=calculate(s,flat_numerics())
    output=out.save(tmp_path/'coated')
    assert (output/'orders.csv').is_file()
    assert json.loads((output/'summary.json').read_text())['film_thickness_nm']==70
    scan=scan_thickness(s,[50,70],flat_numerics())
    assert len(scan['records'])==2 and all(x is not None for x in scan['results'])
    assert [r['thickness_nm'] for r in scan['records']]==[50,70]


def test_bare_solver_still_bare():
    s=flat().substrate
    # Prevent auto rerouting bare calculations to the coating branch.
    assert s.as_dict().get('coating') is None
    with pytest.raises((TypeError,AttributeError)):
        calculate(s,flat_numerics())


@pytest.mark.slow
def test_coated_curved_energy_and_orders():
    p=Profile.echelle(1.3,60,90,tip_radius_um=.04,valley_radius_um=.08)
    s=CoatedSimulation(Simulation(p,600,Incidence.angle(20),-1,
                                  Material.constant(-8+.95j)),Film(15,2.25))
    num=Numerics(p_order=4,max_panel_wavelengths=.25,min_panels_per_segment=2,
                 grading=1.0,modes_air=40,modes_metal=8,
                 oversample=12,max_nodes=500)
    r=calculate(s,num)
    for pol in ('TE','TM'):
        d=r.raw['polarizations'][pol]
        assert d['reflection_sum']<1
        assert abs(d['energy_balance_error'])<.002
        assert r.efficiency(-1,pol,percent=False)>0


def test_failure_status_blocks_efficiency_and_summary():
    s=flat()
    result=calculate(s,flat_numerics())
    # Synthetic failed flag, to exercise fail-closed public behavior.
    result.raw['polarizations']['TE']['status']='failed_passivity'
    result.raw['status']='failed_physical_checks'
    with pytest.raises(RuntimeError,match='NOT physical'):
        result.efficiency(0,'TE')
    assert result.summary()['TE_target_pct'] is None
    assert result.efficiency(0,'TE',allow_failed=True)>0


def test_coated_configuration_roundtrip():
    s=flat()
    assert CoatedSimulation.from_dict(s.as_dict())==s
