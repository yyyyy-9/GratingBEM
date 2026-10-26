"""Integration parity fixture from predecessor v16, not an independent physics reference."""
import pytest
from gratingbem import Profile,Material,Incidence,Simulation,Numerics,calculate

@pytest.mark.slow
def test_echelle_v16_numerical_parity():
    # Recorded from predecessor's exact operator with an identical 79-line
    # echelle, 276 nodes, same Ewald and Nyström choices. This pins refactoring.
    p=Profile.echelle(1000/79,blaze_deg=74,apex_deg=90,tip_radius_um=.02,valley_radius_um=.05)
    sim=Simulation(p,248.33,Incidence.littrow(-98),-98,Material.aluminum())
    num=Numerics(p_order=4,max_panel_wavelengths=1.0,
                 modes_air=155,modes_metal=25,oversample=16,max_nodes=1600)
    result=calculate(sim,num,workers=1)
    assert result.diagnostics['n_nodes']==276
    assert result.summary()['TE_target_pct']==pytest.approx(67.2150396872,abs=1e-5)
    assert result.summary()['TM_target_pct']==pytest.approx(63.6508251293,abs=1e-5)
    assert result.diagnostics['TE']['passes_passivity']
    assert result.diagnostics['TM']['passes_passivity']
