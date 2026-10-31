"""The single user main.py supports process/grid and optional Bloch reuse."""
from __future__ import annotations
import os,sys,math
os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import main as ui

@pytest.fixture
def settings(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ui,'PROFILE',ui.Profile.echelle(.8,30.,90.))
    monkeypatch.setattr(ui,'SOURCE',ui.Incidence.angle(15.))
    monkeypatch.setattr(ui,'WAVELENGTH_NM',248.33)
    monkeypatch.setattr(ui,'TARGET_ORDER',-1)
    monkeypatch.setattr(ui,'METAL',ui.Material.constant(-8+.95j))
    monkeypatch.setattr(ui,'NUMERICS',ui.Numerics(p_order=3,max_panel_wavelengths=1.8,
        modes_air=40,modes_metal=12,oversample=8,max_nodes=400))
    monkeypatch.setattr(ui,'SHOW_PLOTS',False)
    monkeypatch.setattr(ui,'PRINT_ALL_ORDERS',False)
    monkeypatch.setattr(ui,'POLARIZATIONS',('TE','TM'))
    monkeypatch.setattr(ui,'WORKERS',1)
    monkeypatch.setattr(ui,'SCAN_WORKERS',2)
    monkeypatch.setattr(ui,'SCAN_BACKEND','process')
    monkeypatch.setattr(ui,'BLOCH_REUSE',False)
    monkeypatch.setattr(ui,'COATED',False)
    monkeypatch.setattr(ui,'WAVELENGTH_INCIDENCE','fixed')
    yield tmp_path
    ui.plt.close('all')


def test_main_three_param_grid_process_returns_numpy_only(settings,monkeypatch):
    monkeypatch.setattr(ui,'SCAN_AXES',{'wavelength':[248.33,248.50],
                                       'angle':[14.,15.], 'blaze':[29.8,30.2]})
    got=ui.run('grid')
    assert got['scan_shape']==(2,2,2)
    assert got['results_grid'].shape==(2,2,2)
    assert got['TE_map'].shape==(2,2,2)
    assert got['TM_map'].shape==(2,2,2)
    assert got['scan_backend']=='process'
    assert got['batch_elapsed_s'] is not None
    assert all(r is not None for r in got['results'])
    assert not list(settings.iterdir())


def test_main_eligible_angle_alias_uses_shared_bloch(settings,monkeypatch):
    d=.8;wl=.24833
    theta0=10.
    theta1=math.degrees(math.asin(math.sin(math.radians(theta0))+wl/d))
    monkeypatch.setattr(ui,'PROFILE',ui.Profile.flat(d))
    monkeypatch.setattr(ui,'SOURCE',ui.Incidence.angle(theta0))
    monkeypatch.setattr(ui,'TARGET_ORDER',0)
    monkeypatch.setattr(ui,'BLOCH_REUSE',True)
    monkeypatch.setattr(ui,'SCAN_WORKERS',1)
    monkeypatch.setattr(ui,'SCAN_BACKEND','thread')
    monkeypatch.setattr(ui,'NUMERICS',ui.Numerics(p_order=4,max_panel_wavelengths=.5,
        modes_air=155,modes_metal=25,oversample=10,max_nodes=400))
    monkeypatch.setattr(ui,'ANGLE_VALUES',np.array([theta0,theta1]))
    got=ui.run('angle')
    assert got['bloch_reuse']
    assert all(r.diagnostics.get('bloch_group_reuse',{}).get('size')==2 for r in got['results'])
    assert not list(settings.iterdir())
