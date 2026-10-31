"""User-facing main.py automatically identifies Bloch channels."""
import math,os,sys
from pathlib import Path
import numpy as np
import pytest
os.environ.setdefault('MPLBACKEND','Agg')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import main as ui

def test_main_auto_bloch_with_process_scan_and_numpy_no_file(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    d=.8;lam=248.33;t0=10.;
    t1=math.degrees(math.asin(math.sin(math.radians(t0))+.001*lam/d))
    monkeypatch.setattr(ui,'PROFILE',ui.Profile.flat(d))
    monkeypatch.setattr(ui,'SOURCE',ui.Incidence.angle(t0))
    monkeypatch.setattr(ui,'WAVELENGTH_NM',lam)
    monkeypatch.setattr(ui,'TARGET_ORDER',0)
    monkeypatch.setattr(ui,'METAL',ui.Material.constant(-8+.95j))
    monkeypatch.setattr(ui,'NUMERICS',ui.Numerics(p_order=2,max_panel_wavelengths=2.,
                  modes_air=155,modes_metal=25,oversample=10,max_nodes=200))
    monkeypatch.setattr(ui,'POLARIZATIONS',('TE','TM'))
    monkeypatch.setattr(ui,'SHOW_PLOTS',False)
    monkeypatch.setattr(ui,'PRINT_ALL_ORDERS',False)
    monkeypatch.setattr(ui,'COATED',False)
    monkeypatch.setattr(ui,'SCAN_WORKERS',2)
    monkeypatch.setattr(ui,'SCAN_BACKEND','process')
    monkeypatch.setattr(ui,'BLOCH_REUSE','auto')
    monkeypatch.setattr(ui,'ANGLE_VALUES',np.array([t0,12.,t1]))
    out=ui.run('angle')
    assert out['bloch_reuse']=='auto'
    assert out['scan_backend']=='process'
    assert out['TE_target_pct'].shape==(3,)
    assert out['TM_target_pct'].shape==(3,)
    assert out['results'][0].diagnostics['bloch_group_reuse']['size']==2
    assert out['results'][2].diagnostics['bloch_group_reuse']['size']==2
    assert 'bloch_group_reuse' not in out['results'][1].diagnostics
    assert not list(tmp_path.iterdir())
    ui.plt.close('all')
