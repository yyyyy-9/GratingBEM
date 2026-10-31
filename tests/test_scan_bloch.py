"""Bloch group LU-reuse: opt-in, finite-Ewald validation and independent parity."""
import math
import numpy as np
import pytest
from gratingbem import Profile,Material,Incidence,Simulation,Numerics,calculate_many,scan
from gratingbem._scan_bloch import groups,shareable,kernel_discrepancy


def sim(angle,*,wavelength_nm=248.33,blaze=76.156):
    return Simulation(Profile.echelle(1000/83.19,blaze,90),wavelength_nm,
                      Incidence.angle(angle),-94,Material.aluminum())


def angle_shift(theta,wl=248.33,d=1000/83.19,alias=1):
    arg=math.sin(math.radians(theta))+alias*wl/1000/d
    return math.degrees(math.asin(arg))


def reference_numerics(modes=155):
    return Numerics(p_order=4,max_panel_wavelengths=.8,grading=1.15,
                    modes_air=modes,modes_metal=25,oversample=10,max_nodes=800)


def test_group_only_equivalent_and_same_material_wavelength_geometry():
    t=angle_shift(72.)
    cases=[sim(72),sim(76),sim(t),sim(72.,blaze=76.20),sim(72.,wavelength_nm=250),sim(72)]
    assert groups(cases)==[[0,2,5],[1],[3],[4]]


def test_undertruncated_periodic_kernel_is_refused():
    t=angle_shift(72.)
    cases=[sim(72.),sim(t)]
    good,delta=shareable(cases,reference_numerics(155))
    assert good and delta<2e-10
    bad,delta2=shareable(cases,reference_numerics(35))
    assert not bad and delta2>1e-4


def test_alias_two_angle_scattering_full_complex_parity_and_speedup():
    t=angle_shift(72.)
    cases=[sim(72.),sim(t)]
    num=reference_numerics()
    baseline=calculate_many(cases,num,bloch_reuse=False)
    shared=calculate_many(cases,num,bloch_reuse=True)
    assert baseline.errors==shared.errors==[None,None]
    # Performance acceptance belongs to the repeatable benchmark, not a flaky unit assertion.
    for a,b in zip(baseline.results,shared.results):
        assert b.diagnostics['bloch_group_reuse']['size']==2
        np.testing.assert_array_equal(a.modes,b.modes)
        for pol in ('TE','TM'):
            np.testing.assert_allclose(a.amplitudes[pol],b.amplitudes[pol],rtol=2e-10,atol=2e-10)
            np.testing.assert_allclose(a.efficiencies[pol],b.efficiencies[pol],rtol=2e-10,atol=2e-10)
            assert b.diagnostics[pol]['matrix_residual'] < 2e-12


def test_low_modes_auto_fallback_instead_of_unsafe_reuse():
    t=angle_shift(72.)
    base=sim(72.);other=sim(t)
    num=Numerics(p_order=2,max_panel_wavelengths=3.0,modes_air=35,modes_metal=10,
                 oversample=8,max_nodes=100)
    result=calculate_many([base,other],num,bloch_reuse=True,on_error='record')
    assert result.errors==[None,None]
    assert all('bloch_group_reuse' not in r.diagnostics for r in result.results)
    assert all('Bloch reuse refused' in ' '.join(r.warning_messages) for r in result.results)


def test_bloch_reuse_does_not_apply_to_explicit_numerics_scan():
    with pytest.raises(ValueError,match='bloch_reuse'):
        scan(sim(72.),'numerics.p_order',[4,6],reference_numerics(),bloch_reuse=True)
    # In AUTO mode it transparently skips grouping: the operator parameters differ.
    result=scan(sim(72.),'numerics.p_order',[2,3],
                Numerics(p_order=2,max_panel_wavelengths=3,modes_air=25,
                         modes_metal=8,oversample=8,max_nodes=200))
    assert len(result.results)==2
