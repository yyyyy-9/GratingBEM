"""Separate experimental coating CLI. Does not change the bare-metal CLI."""
from __future__ import annotations
import argparse
import csv
import json
import sys
import os
import time
from .assembly import AssemblyOptions,AssemblyTimeout,AssemblyWorkerError
from pathlib import Path
from . import CoatedSimulation, calculate, scan_thickness, offset_profile
from .._core.solver import Numerics, make_profile_nodes
from .geometry import validate_interface_order


def _example():
    return {'simulation': {
        'substrate': {
            'profile': {'shape':'flat','period_um':1.2, 'z_um':0.0},
            'wavelength_nm':800.0, 'incidence':{'angle_deg':17.0},
            'target_order':0,
            'material':{'epsilon_real':-8.0,'epsilon_imag':0.95}},
        'film':{'thickness_nm':70.0,'epsilon_real':2.25,'epsilon_imag':0.0,
                'label':'test dielectric'}},
        'numerics':{'p_order':4,'panels_per_edge':[4],'modes_air':40,
                    'modes_metal':8,'oversample':10},
        'polarizations':['TE','TM'],
        'scan_thickness_nm':[20,70,120]}


def _load(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    sim=CoatedSimulation.from_dict(data['simulation'])
    num=Numerics.from_dict(data.get('numerics',{}))
    return data,sim,num,tuple(data.get('polarizations',['TE','TM']))


def main(argv=None):
    parser=argparse.ArgumentParser(prog='python -m gratingbem.coating',
        description='EXPERIMENTAL single-layer conformal coating BIE (review-gated)')
    commands=parser.add_subparsers(dest='command',required=True)
    init=commands.add_parser('init',help='generate a runnable example JSON')
    init.add_argument('-o','--output',default='coated_example.json')
    for name in ('inspect','run','scan'):
        p=commands.add_parser(name)
        p.add_argument('config')
        if name in ('run','scan'):
            p.add_argument('-o','--output',default='results/coating_'+name)
            p.add_argument('--workers',type=int,default=1,help='isolated assembly processes (1 is safer)')
            p.add_argument('--block-rows',type=int,default=12,help='rows saved per checkpoint')
            p.add_argument('--timeout',type=float,default=120.,help='seconds without block progress before retry')
            p.add_argument('--checkpoint-dir',help='default OUTPUT/assembly_checkpoint')
            p.add_argument('--fresh',action='store_true',help='recompute matrices rather than restore cached blocks')

    args=parser.parse_args(argv)
    try:
        if args.command=='init':
            dst=Path(args.output)
            if dst.exists():raise ValueError('output already exists')
            dst.parent.mkdir(parents=True,exist_ok=True)
            dst.write_text(json.dumps(_example(),indent=2),encoding='utf8')
            print(dst);return 0
        data,sim,num,pols=_load(args.config)
        if args.command=='inspect':
            p=sim.to_problem()
            top=p.top_profile;gap=validate_interface_order(sim.substrate.profile,top)
            nt=len(make_profile_nodes(__import__('dataclasses').replace(p.substrate,profile=top),num).coords)
            nb=len(make_profile_nodes(p.substrate,num).coords)
            print(json.dumps({'film_thickness_nm':sim.film.thickness_nm,
                              'min_sampled_vertical_gap_um':gap,
                              'outer_nodes':nt,'substrate_nodes':nb,
                              'estimated_matrix_dimension':2*(nt+nb),
                              'status':'geometry_checked_not_solved'},indent=2))
            return 0
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else out/'assembly_checkpoint'
        log_path=out/'progress.jsonl'
        def on_progress(evt):
            record=dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**evt)
            with log_path.open('a',encoding='utf8') as f:
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
                f.flush()
                os.fsync(f.fileno())
            if evt['event']=='block_complete':
                print(f"[{evt['operator']}] {evt['completed_rows']}/{evt['total_rows']} rows"
                      f"{' (restored)' if evt.get('restored') else ''}",flush=True)
            elif evt['event'] in ('timeout_retry','operator_complete'):
                print(json.dumps(evt,ensure_ascii=False),flush=True)
        options=AssemblyOptions(checkpoint_dir=checkpoint_dir,workers=args.workers,
            block_rows=args.block_rows,block_timeout_s=args.timeout,
            resume=not args.fresh,progress=on_progress)
        if args.command=='run':
            result=calculate(sim,num,polarizations=pols,assembly_options=options)
            result.save(out)
            print(json.dumps(result.summary(),indent=2))
            if result.status.startswith('failed'):
                print('FAILED physical checks: do not use diffraction efficiencies',file=sys.stderr)
                return 2
            print('NOTE: energy check only; independent mesh refinement required',file=sys.stderr)
            return 0
        if args.command=='scan':
            values=data.get('scan_thickness_nm',[])
            result=scan_thickness(sim,values,num,polarizations=pols,assembly_options=options)
            rows=result['records']
            (out/'scan.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
            keys=list(dict.fromkeys(k for row in rows for k in row))
            with (out/'scan.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(rows)
            print(json.dumps({'cases':len(rows),'wall_seconds':result['total_seconds'],
                              'failed':sum(r.get('status','').startswith(('failed','error')) for r in rows),
                              'output':str(out)},indent=2))
            return 2 if any(r.get('status','').startswith(('failed','error')) for r in rows) else 0
    except KeyboardInterrupt:
        print('Interrupted: completed matrix blocks remain in the checkpoint directory.',file=sys.stderr)
        return 130
    except (AssemblyTimeout,AssemblyWorkerError) as exc:
        print(f'Assembly stopped safely: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 3
    except (ValueError,TypeError,KeyError,NotImplementedError) as e:
        print(f'coating: {type(e).__name__}: {e}',file=sys.stderr)
        return 2
    return 2


if __name__=='__main__':raise SystemExit(main())
