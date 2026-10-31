"""User-facing multidimensional scan behavior, no file-based result handoff."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import main as ui


@pytest.fixture
def setup(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ui,'PROFILE',ui.Profile.flat(.5))
    monkeypatch.setattr(ui,'WAVELENGTH_NM',248.33)
    monkeypatch.setattr(ui,'SOURCE',ui.Incidence.angle(20.))
    monkeypatch.setattr(ui,'TARGET_ORDER',0)
    monkeypatch.setattr(ui,'METAL',ui.Material.constant(-8+.95j))
    monkeypatch.setattr(ui,'NUMERICS',ui.Numerics(p_order=4,max_panel_wavelengths=.7,
        modes_air=20,modes_metal=8,oversample=10,max_nodes=300))
    monkeypatch.setattr(ui,'WORKERS',1)
    monkeypatch.setattr(ui,'COATED',False)
    monkeypatch.setattr(ui,'SHOW_PLOTS',False)
    monkeypatch.setattr(ui,'PRINT_ALL_ORDERS',False)
    monkeypatch.setattr(ui,'PLOT_AXES',None)
    monkeypatch.setattr(ui,'PLOT_SLICE',{})
    monkeypatch.setattr(ui,'WAVELENGTH_INCIDENCE','fixed')
    yield tmp_path
    ui.plt.close('all')


def test_grid_two_axes_real_solver_shape_order_and_no_outputs(setup,monkeypatch):
    monkeypatch.setattr(ui,'SCAN_AXES',{'wavelength':[245.,250.], 'angle':[10.,20.]})
    out=ui.run('grid')
    assert out['scan_shape']==(2,2)
    assert out['TE_map'].shape==(2,2)
    assert len(out['results'])==4
    np.testing.assert_allclose(out['angles_deg'],[10,20,10,20])
    np.testing.assert_allclose([r.problem.wavelength_um for r in out['results']], [.245,.245,.25,.25])
    np.testing.assert_allclose(ui.order_efficiency_map(out,0,'TE'),out['TE_map'])
    np.testing.assert_allclose(out['TE_target_pct'],out['TE_map'].reshape(-1))
    assert out['figures'][1] is not None
    assert not list(setup.iterdir())


def test_paired_values_not_cartesian(setup,monkeypatch):
    monkeypatch.setattr(ui,'SCAN_AXES',{'wavelength':[247,248,249], 'angle':[11,12,13]})
    out=ui.run('paired')
    assert len(out['results'])==3
    np.testing.assert_allclose(out['angles_deg'],[11,12,13])
    np.testing.assert_allclose(out['TE_map'],out['TE_target_pct'])
    assert not list(setup.iterdir())


def test_grid_littrow_vs_fixed_wavelength(setup,monkeypatch):
    monkeypatch.setattr(ui,'SOURCE',ui.Incidence.littrow(-1))
    monkeypatch.setattr(ui,'PROFILE',ui.Profile.flat(.75))
    monkeypatch.setattr(ui,'SCAN_AXES',{'wavelength':[248,250], 'period':[.74,.76]})
    fixed=ui.run('grid')
    assert np.ptp(fixed['angles_deg'])<1e-9
    monkeypatch.setattr(ui,'WAVELENGTH_INCIDENCE','littrow')
    littrow=ui.run('grid')
    assert np.ptp(littrow['angles_deg'])>0.01


def test_wrong_littrow_with_explicit_angle_fails_before_solver(setup,monkeypatch):
    monkeypatch.setattr(ui,'SOURCE',ui.Incidence.littrow(-1))
    monkeypatch.setattr(ui,'WAVELENGTH_INCIDENCE','littrow')
    monkeypatch.setattr(ui,'SCAN_AXES',{'wavelength':[247,248], 'angle':[10,11]})
    with pytest.raises(ValueError,match='显式入射角'):
        ui.run('grid')


def test_paired_mismatched_lengths_rejected(setup,monkeypatch):
    monkeypatch.setattr(ui,'SCAN_AXES',{'angle':[10,20], 'wavelength':[247,248,249]})
    with pytest.raises(ValueError,match='长度'):
        ui.run('paired')


def test_three_dimensional_slice_and_results_grid(setup,monkeypatch):
    monkeypatch.setattr(ui,'SOURCE',ui.Incidence.angle(15.))
    monkeypatch.setattr(ui,'PROFILE',ui.Profile.echelle(.8,30,90))
    monkeypatch.setattr(ui,'SCAN_AXES',{'wavelength':[240,245], 'angle':[10,12], 'blaze':[28,29]})
    monkeypatch.setattr(ui,'PLOT_AXES',('wavelength','angle'))
    monkeypatch.setattr(ui,'PLOT_SLICE',{'blaze':29.})
    out=ui.run('grid')
    assert out['scan_shape']==(2,2,2)
    assert len(out['results'])==8
    assert out['results_grid'].shape==(2,2,2)
    assert out['plot_slice']=={'blaze':29.}
    np.testing.assert_allclose(ui.order_efficiency_map(out,100,'TE'),np.zeros((2,2,2)))
    assert not list(setup.iterdir())


def test_invalid_grid_order_and_too_many(setup,monkeypatch):
    monkeypatch.setattr(ui,'SCAN_AXES',{'angle':[20,10,30],'wavelength':[247,249]})
    with pytest.raises(ValueError,match='严格单调'):
        ui.run('grid')
    monkeypatch.setattr(ui,'SCAN_AXES',{'angle':list(range(30)), 'wavelength':list(range(30,60))})
    with pytest.raises(ValueError,match='MAX_CALCULATIONS'):
        ui.run('grid')
