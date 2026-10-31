"""End-user main.py: single calculation, angle/wavelength/blaze scan, no forced save."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import main as ui


@pytest.fixture
def demo(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ui,'PROFILE',ui.Profile.flat(.5))
    monkeypatch.setattr(ui,'WAVELENGTH_NM',248.33)
    monkeypatch.setattr(ui,'TARGET_ORDER',0)
    monkeypatch.setattr(ui,'SOURCE',ui.Incidence.angle(20.))
    monkeypatch.setattr(ui,'COATED',False)
    monkeypatch.setattr(ui,'POLARIZATIONS',('TE','TM'))
    monkeypatch.setattr(ui,'NUMERICS',ui.Numerics(p_order=4,max_panel_wavelengths=.6,
        modes_air=25,modes_metal=10,oversample=10,max_nodes=400))
    monkeypatch.setattr(ui,'WORKERS',1)
    monkeypatch.setattr(ui,'SHOW_PLOTS',False)
    yield tmp_path
    ui.plt.close('all')


def test_single_returns_all_orders_and_does_not_export(demo,monkeypatch):
    monkeypatch.setattr(ui,'MODE','single')
    d=ui.run()
    assert 'TE_orders_pct' in d and 'TM_orders_pct' in d
    assert len(d['orders']) == len(d['TE_orders_pct'])
    j=np.flatnonzero(d['orders']==0)[0]
    assert d['TE_orders_pct'][j]==pytest.approx(d['TE_target_pct'][0])
    assert not list(demo.iterdir())


def test_angle_scan_changes_incidence(demo,monkeypatch):
    monkeypatch.setattr(ui,'MODE','angle')
    monkeypatch.setattr(ui,'ANGLE_VALUES',np.array([10.,20.]))
    d=ui.run()
    np.testing.assert_allclose(d['angles_deg'],[10.,20.])
    assert len(d['results'])==2
    assert all(r.problem.wavelength_um==pytest.approx(.24833) for r in d['results'])
    assert not list(demo.iterdir())


def test_wavelength_policy_is_explicit(demo,monkeypatch):
    monkeypatch.setattr(ui,'MODE','wavelength')
    monkeypatch.setattr(ui,'WAVELENGTH_VALUES',np.array([247.,250.]))
    monkeypatch.setattr(ui,'WAVELENGTH_INCIDENCE','fixed')
    fixed=ui.run()
    np.testing.assert_allclose(fixed['angles_deg'],20.)
    monkeypatch.setattr(ui,'SOURCE',ui.Incidence.littrow(-1))
    monkeypatch.setattr(ui,'WAVELENGTH_INCIDENCE','littrow')
    littrow=ui.run()
    assert littrow['angles_deg'][0]!=pytest.approx(littrow['angles_deg'][1])
    assert not list(demo.iterdir())


def test_blaze_scan_changes_geometry_at_fixed_angle(demo,monkeypatch):
    monkeypatch.setattr(ui,'MODE','blaze')
    monkeypatch.setattr(ui,'PROFILE',ui.Profile.echelle(.8333333,28.,90.))
    monkeypatch.setattr(ui,'TARGET_ORDER',-1)
    monkeypatch.setattr(ui,'BLAZE_VALUES',np.array([28.,29.]))
    d=ui.run()
    np.testing.assert_allclose(d['angles_deg'],[20.,20.])
    assert d['results'][0].problem.profile.vertices!=d['results'][1].problem.profile.vertices
    assert not list(demo.iterdir())


def test_bad_blaze_shape_rejected(demo,monkeypatch):
    monkeypatch.setattr(ui,'MODE','blaze')
    with pytest.raises(ValueError,match='echelle'):
        ui.run()


def test_matplotlib_show_when_interactive(demo,monkeypatch):
    monkeypatch.setattr(ui,'MODE','single')
    monkeypatch.setattr(ui,'SHOW_PLOTS',True)
    hits=[]
    monkeypatch.setattr(ui.plt,'get_backend',lambda:'QtAgg')
    monkeypatch.setattr(ui.plt,'show',lambda:hits.append(1))
    ui.run()
    assert hits == [1]


def test_coating_mode_reuses_same_main(demo,monkeypatch):
    monkeypatch.setattr(ui,'MODE','single')
    monkeypatch.setattr(ui,'COATED',True)
    monkeypatch.setattr(ui,'PROFILE',ui.Profile.flat(1.2))
    monkeypatch.setattr(ui,'SOURCE',ui.Incidence.angle(17.))
    monkeypatch.setattr(ui,'WAVELENGTH_NM',800.)
    monkeypatch.setattr(ui,'METAL',ui.Material.constant(-8+.95j))
    monkeypatch.setattr(ui,'FILM_THICKNESS_NM',70.)
    monkeypatch.setattr(ui,'NUMERICS',ui.Numerics(p_order=4,panels_per_edge=(4,),
        modes_air=40,modes_metal=8,oversample=10))
    d=ui.run()
    assert d['last_result'].status=='passed_energy_only_mesh_unverified'
    assert not list(demo.iterdir())
