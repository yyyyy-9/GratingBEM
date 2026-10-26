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
from ._core.validation import integer_order


def _problem(simulation):
    if isinstance(simulation,Simulation):return simulation.to_problem()
    if isinstance(simulation,Problem):return simulation
    raise TypeError('simulation must be a Simulation (or a legacy Problem)')


def calculate(simulation, numerics:Numerics|None=None, *, polarizations=('TE','TM'),
              workers:int=1, execution:Execution|None=None)->Calculation:
    """Strict 2D periodic, single-interface TE/TM KM1 BIE (not RCIP/coatings).

    Numeric efficiency alone is not an accuracy guarantee; check mesh/order convergence.
    """
    return _calculate(_problem(simulation),numerics,polarizations,
                      execution=execution or Execution(workers=workers))


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
         *, polarizations=('TE','TM'),workers:int=1,on_error='raise',
         progress=None)->ScanResult:
    """Scan profile.tip_radius_um, profile.apex_deg, wavelength_nm, etc.

    For arbitrary custom shapes: parameter='profile', values=[Profile(...),...].
    """
    if on_error not in ('raise','record'):raise ValueError('on_error: raise or record')
    vals=list(values)
    if not vals:raise ValueError('scan values must not be empty')
    start=time.perf_counter();runs=[];errors=[]
    for i,value in enumerate(vals):
        result=None;err=None
        try:
            if parameter.startswith('numerics.'):
                key=parameter.split('.',1)[1]
                if key not in Numerics.__dataclass_fields__:raise ValueError('unknown numerics field: '+key)
                sim=simulation;num=replace(numerics or Numerics(),**{key:value})
            else:
                sim=simulation.with_parameter(parameter,value);num=numerics
            result=calculate(sim,num,polarizations=polarizations,workers=workers)
        except Exception as exc:
            if on_error=='raise':raise
            err=f'{type(exc).__name__}: {exc}'
        runs.append(result);errors.append(err)
        if progress:progress(i+1,len(vals),value,result,err)
    return ScanResult(parameter,vals,runs,errors,{'total_s':time.perf_counter()-start})


def scan_grid(simulation,axes:dict,numerics:Numerics|None=None,*,polarizations=('TE','TM'),
              workers:int=1,on_error='raise',progress=None):
    """Parameter grid scan; all grid keys must be Simulation properties, not numerics."""
    if on_error not in ('raise','record'):raise ValueError('on_error: raise or record')
    if not axes:raise ValueError('axes may not be empty')
    cases=[dict(zip(axes,v)) for v in itertools.product(*(list(values) for values in axes.values()))]
    if not cases:raise ValueError('empty grid')
    start=time.perf_counter();runs=[];errors=[]
    for i,params in enumerate(cases):
        result=None;err=None
        try:
            sim=simulation
            for key,value in params.items():sim=sim.with_parameter(key,value)
            result=calculate(sim,numerics,polarizations=polarizations,workers=workers)
        except Exception as exc:
            if on_error=='raise':raise
            err=f'{type(exc).__name__}: {exc}'
        runs.append(result);errors.append(err)
        if progress:progress(i+1,len(cases),params,result,err)
    return ScanResult('grid:'+','.join(axes),cases,runs,errors,{'total_s':time.perf_counter()-start})


def convergence(simulation, panel_wavelengths=(1.5,1.1,0.8),
                numerics:Numerics|None=None,*,polarizations=('TE','TM'),workers:int=1):
    """Simple mesh study. Difference between successive grids is not a rigorous error bound."""
    return scan(simulation,'numerics.max_panel_wavelengths',panel_wavelengths,numerics,
                polarizations=polarizations,workers=workers)

__all__=['Numerics','Execution','AdaptiveOptions','Calculation','BalancedRun',
         'calculate','calculate_adaptive','scan','scan_grid','convergence','efficiency']
