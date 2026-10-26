import json
import pytest
from gratingbem.cli import main


def flat_config(tmp_path):
    config=tmp_path/'config.json'
    assert main(['init','--shape','rectangle','-o',str(config)])==0
    data=json.loads(config.read_text())
    data['simulation']['profile']={'shape':'flat','period_um':.6}
    data['numerics']={'p_order':4,'max_panel_wavelengths':.5,
                      'modes_air':45,'modes_metal':12,'oversample':10,'max_nodes':200}
    data['polarizations']=['te','tm']
    config.write_text(json.dumps(data))
    return config


def test_init_inspect_run(tmp_path,capsys):
    config=flat_config(tmp_path)
    assert main(['inspect',str(config)])==0
    out=tmp_path/'res'
    assert main(['run',str(config),'-o',str(out)])==0
    assert (out/'orders.csv').exists()
    assert (out/'run_info.json').exists()


def test_cli_scan(tmp_path):
    config=flat_config(tmp_path)
    data=json.loads(config.read_text())
    data['scan']={'parameter':'wavelength_nm','values':[248.33,250]}
    config.write_text(json.dumps(data))
    assert main(['scan',str(config),'-o',str(tmp_path/'scan')])==0
    assert (tmp_path/'scan/scan.csv').exists()


@pytest.mark.parametrize('distribution_check,status,exit_code', [
    (True,'distribution_check_incomplete',2),
    (False,'empirically_converged',0),
])
def test_cli_converge_flat(tmp_path,distribution_check,status,exit_code):
    config=flat_config(tmp_path)
    data=json.loads(config.read_text())
    data['adaptive']={'min_levels':3,'max_levels':3,'panel_limit_lambda':.5,
                      'independent_distribution_check':distribution_check}
    config.write_text(json.dumps(data))
    assert main(['converge',str(config),'-o',str(tmp_path/'convergence')])==exit_code
    summary=json.loads((tmp_path/'convergence/adaptive_summary.json').read_text())
    assert summary['status']==status
    assert summary['empirically_converged'] is (exit_code==0)
    assert summary['accuracy_certified'] is False


def test_coating_rejected_in_user_config(tmp_path):
    config=tmp_path/'bad.json'
    config.write_text(json.dumps({'simulation':{'profile':{'shape':'flat','period_um':.6},
        'wavelength_nm':248.33,'incidence':{'angle_deg':0},'material':{'model':'aluminum_rakic_ld'},
        'target_order':0,'coating':{'thickness_nm':5}}}))
    assert main(['inspect',str(config)])==2


@pytest.mark.slow
def test_scan_passivity_failure_returns_nonzero(tmp_path,capsys):
    """A real coarse echelle fails passivity; completed solves are not valid cases."""
    config=tmp_path/'coarse.json'
    assert main(['init','-o',str(config)])==0
    data=json.loads(config.read_text())
    data['numerics']['max_panel_wavelengths']=3.0
    data['polarizations']=['TE']
    data['scan']={'parameter':'wavelength_nm','values':[248.33]}
    config.write_text(json.dumps(data))
    assert main(['scan',str(config),'-o',str(tmp_path/'coarse')])==2
    saved=json.loads((tmp_path/'coarse/run_info.json').read_text())
    assert saved['run_information']['failed_cases']==0
    assert saved['run_information']['passivity_failed_cases']==1
