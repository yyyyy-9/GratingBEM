"""Command-line interface: inspect, run, sweep, adaptive and sample configs."""
from __future__ import annotations
import argparse
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import time

from . import __version__
from .model import Simulation, Profile, Incidence, Material
from .api import Numerics, AdaptiveOptions, calculate, calculate_adaptive, scan, scan_grid, convergence


def _demo(shape='echelle') -> dict:
    if shape=='echelle':
        profile={'shape':'echelle','period_um':1000/79,'blaze_deg':74.0,
                 'apex_deg':90.,'tip_radius_um':.02,'valley_radius_um':.05}
        target=-98;order=-98
    elif shape=='rectangle':
        profile={'shape':'rectangle','period_um':.8,'height_um':.08,
                 'duty_cycle':.45,'corner_radius_um':.005}
        target=0;order=None
    elif shape=='staircase':
        profile={'shape':'staircase','period_um':1.2,'height_um':.15,
                 'steps':3,'duty_cycle':.65,'corner_radius_um':.004}
        target=0;order=None
    else:
        raise ValueError('shape echelle, rectangle, staircase')
    return {'simulation':{'profile':profile,'wavelength_nm':248.33,
                          'incidence':{'littrow_order':order} if order is not None else {'angle_deg':12.},
                          'target_order':target,'material':{'model':'aluminum_rakic_ld'}},
            'numerics':{'p_order':4,'max_panel_wavelengths':1.0,
                        'modes_air':155,'modes_metal':25,'oversample':16,'max_nodes':1600},
            'polarizations':['TE','TM'],
            'adaptive':{'target_abs_tol':.001,'max_levels':5},
            'scan':{'parameter':'profile.tip_radius_um' if shape=='echelle' else 'profile.corner_radius_um',
                    'values':[0.,.01,.02]}}


def _load(path: str):
    file=Path(path)
    data=json.loads(file.read_text(encoding='utf-8'))
    if 'simulation' in data:
        sim=Simulation.from_dict(data['simulation'])
    elif 'problem' in data:
        # Forward-compatible reader of older v12-v16 research JSON files.
        from ._core.solver import Problem
        sim=Simulation.from_legacy_problem(Problem.from_dict(data['problem']))
    else:
        raise ValueError('JSON must contain a simulation object')
    num=Numerics.from_dict(data.get('numerics',{}))
    pols=tuple(str(pol).upper() for pol in data.get('polarizations',('TE','TM')))
    if not pols or any(pol not in ('TE','TM') for pol in pols):
        raise ValueError('polarizations must be a nonempty subset of TE,TM')
    return data,sim,num,pols


def _record(out:Path,content:dict,config_path=None):
    import numpy,scipy
    info={'gratingbem_version':__version__, 'python':sys.version.split()[0],
          'numpy':numpy.__version__,'scipy':scipy.__version__,
          'platform':platform.platform(),'source_config':str(config_path) if config_path else None}
    if config_path:
        info['config_sha256']=hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    info['run_information']=content
    out.mkdir(parents=True,exist_ok=True)
    (out/'run_info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2,default=str),encoding='utf8')


def main(argv=None):
    p=argparse.ArgumentParser(prog='gratingbem',description='Periodic TE/TM BEM for bare metal grating profiles')
    p.add_argument('--version',action='version',version=f'gratingbem {__version__}')
    sub=p.add_subparsers(dest='command',required=True)
    init=sub.add_parser('init',help='write a runnable JSON config')
    init.add_argument('-o','--output',default='grating.json')
    init.add_argument('--shape',choices=['echelle','rectangle','staircase'],default='echelle')
    inspect=sub.add_parser('inspect',help='validate and report geometry/material/mesh without solving')
    inspect.add_argument('config')
    run=sub.add_parser('run',help='solve one bare interface')
    run.add_argument('config');run.add_argument('-o','--output',default='results/run')
    run.add_argument('--workers',type=int,default=1)
    sweep=sub.add_parser('scan',help='sweep a parameter in config scan section')
    sweep.add_argument('config');sweep.add_argument('-o','--output',default='results/scan')
    sweep.add_argument('--workers',type=int,default=1)
    conv=sub.add_parser('converge',help='automatic empirical mesh refinement and audits')
    conv.add_argument('config');conv.add_argument('-o','--output',default='results/convergence')
    conv.add_argument('--workers',type=int,default=1)
    args=p.parse_args(argv)
    try:
        if args.command=='init':
            dest=Path(args.output)
            if dest.exists():p.error(f'{dest} already exists; choose a new path')
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(json.dumps(_demo(args.shape),indent=2,ensure_ascii=False),encoding='utf8')
            print(f'Wrote {dest} — inspect it, then gratingbem run {dest}')
            return 0
        data,sim,num,pols=_load(args.config)
        if args.command=='inspect':
            from ._core.solver import make_profile_nodes
            nd=make_profile_nodes(sim.to_problem(),num)
            summary={'period_um':sim.profile.period_um,'wavelength_nm':sim.wavelength_nm,
                     'incidence_deg':sim.incidence_deg,'target_order':sim.target_order,
                     'epsilon_real':sim.to_problem().epsilon_value.real,
                     'epsilon_imag':sim.to_problem().epsilon_value.imag,
                     'n_geometric_segments':len(sim.profile.geometric_segments),
                     'n_nodes':len(nd.coords), 'n_panels':len(nd.panels['ds']),
                     'material':sim.material.model,'coating_supported':False}
            print(json.dumps(summary,indent=2,ensure_ascii=False));return 0
        out=Path(args.output)
        if args.command=='run':
            res=calculate(sim,num,polarizations=pols,workers=args.workers)
            res.save(out)
            summary=res.summary();_record(out,summary,args.config)
            print(json.dumps(summary,indent=2,ensure_ascii=False))
            bad=any(not res.diagnostics[pol]['passes_passivity'] for pol in pols)
            if bad:
                print('WARNING: passivity failed. Do NOT use this efficiency; refine.',file=sys.stderr)
            return 2 if bad else 0
        if args.command=='scan':
            spec=data.get('scan')
            if not spec:raise ValueError('provide a scan object: {"parameter":"profile.tip_radius_um", "values":[...]}')
            if 'axes' in spec:
                res=scan_grid(sim,spec['axes'],num,polarizations=pols,workers=args.workers,on_error=spec.get('on_error','record'))
            else:
                res=scan(sim,spec['parameter'],spec['values'],num,polarizations=pols,workers=args.workers,on_error=spec.get('on_error','record'))
            res.save(out)
            invalid=sum(result is not None and any(not result.diagnostics[pol]['passes_passivity']
                        for pol in result.efficiencies) for result in res.results)
            summary={'n_cases':len(res.results),'failed_cases':sum(v is None for v in res.results),
                     'passivity_failed_cases':invalid,
                     'wall_s':res.timings['total_s'],'saved_to':str(out)}
            _record(out,summary,args.config)
            print(json.dumps(summary,indent=2))
            return 2 if summary['failed_cases'] or invalid else 0
        if args.command=='converge':
            opts=AdaptiveOptions(**data.get('adaptive',{}))
            res=calculate_adaptive(sim,num,opts,polarizations=pols,workers=args.workers,checkpoint_dir=out)
            res.save(out,save_all_runs=False)
            summary=res.summary()
            _record(out,summary,args.config)
            print(json.dumps(summary,indent=2,ensure_ascii=False,default=str))
            return 0 if res.converged else 2
    except (ValueError,TypeError,KeyError,NotImplementedError,OSError) as exc:
        print(f'gratingbem: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
