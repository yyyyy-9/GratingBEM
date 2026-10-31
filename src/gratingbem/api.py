"""One-call calculation, scans, and empirical convergence checks."""
from __future__ import annotations
from dataclasses import replace
import itertools
import time
from pathlib import Path
from collections.abc import Iterable
from .model import Simulation
from ._core.solver import Problem, Numerics, ScanResult, Calculation
from ._core.dispatch import Execution, calculate as _calculate, calculate_adaptive as _adaptive
from ._core.adaptive import BalancedOptions as AdaptiveOptions, BalancedRun
from ._scan_batch import calculate_many, BatchResult
from ._core.validation import integer_order


def _problem(simulation):
    if isinstance(simulation,Simulation):return simulation.to_problem()
    if isinstance(simulation,Problem):return simulation
    raise TypeError('simulation must be a Simulation (or a legacy Problem)')


def calculate(simulation, numerics:Numerics|None=None, *, polarizations=('TE','TM'),
              workers:int=1, execution:Execution|None=None, energy_contour=False)->Calculation:
    """Strict 2D periodic, single-interface TE/TM KM1 BIE (not RCIP/coatings).

    Numeric efficiency alone is not an accuracy guarantee; check mesh/order convergence.
    """
    return _calculate(_problem(simulation),numerics,polarizations,
                      execution=execution or Execution(workers=workers),energy_contour=energy_contour)


def calculate_adaptive(simulation,numerics:Numerics|None=None,
                       options:AdaptiveOptions|None=None, *,
                       polarizations=('TE','TM'),workers:int=1,
                       execution:Execution|None=None,checkpoint_dir=None)->BalancedRun:
    return _adaptive(_problem(simulation),numerics,options,polarizations,
                     execution=execution or Execution(workers=workers),
                     checkpoint_dir=checkpoint_dir)


def efficiency(result:Calculation, order:int, polarization:str='TE', *, percent:bool=True) -> float:
    """Diffraction efficiency at a given propagating order.

    Returns percent by default; ``percent=False`` returns a 0--1 fraction.
    Raises ValueError if the order is not propagating.
    """
    pol=polarization.upper()
    if pol not in result.efficiencies:
        raise ValueError(f'polarization {pol!r} not computed; choices: {list(result.efficiencies)}')
    import numpy as np
    order=integer_order(order)
    idx=np.flatnonzero(result.modes==order)
    if len(idx)!=1:
        raise ValueError(f'order {order} is not propagating or not included')
    val=float(result.efficiencies[pol][int(idx[0])])
    return (100.0 if percent else 1.0)*val


def scan(simulation,parameter:str,values:Iterable,numerics:Numerics|None=None,
         *, polarizations=('TE','TM'),workers:int=1,scan_workers:int=1,
         on_error='raise', progress=None,scan_backend='thread',
         bloch_reuse='auto')->ScanResult:
    """Scan profile.tip_radius_um, profile.apex_deg, wavelength_nm, etc.

    For arbitrary custom shapes: parameter='profile', values=[Profile(...),...].
    """
    if on_error not in ('raise','record'):raise ValueError('on_error: raise or record')
    if scan_backend not in ('thread','process'):raise ValueError('scan_backend: thread or process')
    vals=list(values)
    if not vals:raise ValueError('scan values must not be empty')
    if parameter.startswith('numerics.'):
        if bloch_reuse is True or bloch_reuse=='on':
            raise ValueError('explicit bloch_reuse does not apply to numerics scans')
        key=parameter.split('.',1)[1]
        if key not in Numerics.__dataclass_fields__:
            raise ValueError('unknown numerics field: '+key)
        if scan_workers != 1:
            raise ValueError('numeric settings scans use scan_workers=1 for now')
        t0=time.perf_counter();runs=[];errors=[]
        for i,value in enumerate(vals):
            try:
                r=calculate(simulation,replace(numerics or Numerics(),**{key:value}),
                            polarizations=polarizations,workers=workers)
                e=None
            except Exception as exc:
                if on_error=='raise':raise
                r=None;e=f'{type(exc).__name__}: {exc}'
            runs.append(r);errors.append(e)
            if progress: progress(i+1,len(vals),value,r,e)
        return ScanResult(parameter,vals,runs,errors,{'total_s':time.perf_counter()-t0})
    t0=time.perf_counter()
    prepared=[]; original_indices=[]
    runs=[None]*len(vals);errors=[None]*len(vals)
    for i,value in enumerate(vals):
        try:
            case=simulation.with_parameter(parameter,value)
        except Exception as exc:
            if on_error=='raise': raise
            errors[i]=f'{type(exc).__name__}: {exc}'
            if progress: progress(sum(e is not None for e in errors),len(vals),value,None,errors[i])
        else:
            prepared.append(case);original_indices.append(i)
    if prepared:
        invalid_count=sum(e is not None for e in errors)
        def cb(done,total,index,result,error):
            if progress:
                i=original_indices[index]
                progress(invalid_count+done,len(vals),vals[i],result,error)
        batch=calculate_many(prepared,numerics,polarizations=polarizations,
                             workers=workers,scan_workers=scan_workers,on_error=on_error,
                             progress=cb,scan_backend=scan_backend,bloch_reuse=bloch_reuse)
        for j,i in enumerate(original_indices):
            runs[i]=batch.results[j]; errors[i]=batch.errors[j]
    return ScanResult(parameter,vals,runs,errors,{'total_s':time.perf_counter()-t0})


def scan_grid(simulation,axes:dict,numerics:Numerics|None=None,*,polarizations=('TE','TM'),
              workers:int=1,scan_workers:int=1,on_error='raise',progress=None,
              scan_backend='thread',bloch_reuse='auto'):
    """Parameter grid scan; all grid keys must be Simulation properties, not numerics."""
    if on_error not in ('raise','record'):raise ValueError('on_error: raise or record')
    if not axes:raise ValueError('axes may not be empty')
    cases=[dict(zip(axes,v)) for v in itertools.product(*(list(values) for values in axes.values()))]
    if not cases:raise ValueError('empty grid')
    t0=time.perf_counter()
    prepared=[];orig_indices=[]
    runs=[None]*len(cases);errors=[None]*len(cases)
    for i,params in enumerate(cases):
        try:
            sim=simulation
            for key,value in params.items(): sim=sim.with_parameter(key,value)
        except Exception as exc:
            if on_error=='raise':raise
            errors[i]=f'{type(exc).__name__}: {exc}'
            if progress:progress(sum(e is not None for e in errors),len(cases),params,None,errors[i])
        else:
            prepared.append(sim);orig_indices.append(i)
    if prepared:
        invalid_count=sum(e is not None for e in errors)
        def cb(done,total,index,result,error):
            if progress:
                i=orig_indices[index]
                progress(invalid_count+done,len(cases),cases[i],result,error)
        batch=calculate_many(prepared,numerics,polarizations=polarizations,
                             workers=workers,scan_workers=scan_workers,on_error=on_error,
                             progress=cb,scan_backend=scan_backend,bloch_reuse=bloch_reuse)
        for j,i in enumerate(orig_indices):
            runs[i]=batch.results[j];errors[i]=batch.errors[j]
    return ScanResult('grid:'+','.join(axes),cases,runs,errors,{'total_s':time.perf_counter()-t0})


def convergence(simulation, panel_wavelengths=(1.5,1.1,0.8),
                numerics:Numerics|None=None,*,polarizations=('TE','TM'),workers:int=1):
    """Simple mesh study. Difference between successive grids is not a rigorous error bound."""
    values=list(panel_wavelengths)
    if not values: raise ValueError('panel_wavelengths may not be empty')
    outputs=[]; t0=time.perf_counter()
    for panel_w in values:
        out=calculate(simulation,replace(numerics or Numerics(),max_panel_wavelengths=panel_w),
                      polarizations=polarizations,workers=workers)
        outputs.append(out)
    return ScanResult('numerics.max_panel_wavelengths',values,outputs,
                      [None]*len(values),{'total_s':time.perf_counter()-t0})

__all__=['Numerics','Execution','AdaptiveOptions','Calculation','BalancedRun',
         'calculate','calculate_adaptive','calculate_many','BatchResult','scan','scan_grid','convergence','efficiency']
