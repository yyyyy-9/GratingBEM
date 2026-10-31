"""CLI smoke check for independently reviewable coating workflow."""
import json
import os
from pathlib import Path
import subprocess
import sys


def _run(*args, cwd=None):
    env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1'
    return subprocess.run([sys.executable,'-m','gratingbem.coating',*map(str,args)],
                          cwd=cwd,env=env,capture_output=True,text=True)


def test_cli_init_inspect_run(tmp_path):
    config=tmp_path/'film.json';out=tmp_path/'out'
    assert _run('init','-o',config).returncode==0
    assert config.exists()
    proc=_run('inspect',config)
    assert proc.returncode==0,proc.stderr
    assert json.loads(proc.stdout)['estimated_matrix_dimension']==64
    proc=_run('run',config,'-o',out)
    assert proc.returncode==0,proc.stderr
    summary=json.loads((out/'summary.json').read_text())
    assert summary['status']=='passed_energy_only_mesh_unverified'
    for name in ('orders.csv','simulation.json','diagnostics.json',
                 'substrate_profile.svg','outer_profile.svg'):
        assert (out/name).is_file()
