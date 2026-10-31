"""v16 user-facing API: v13 Maxwell physics, v15 audited adaptivity,
process-parallel curved matrix assembly and goal-oriented refinement scores.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from gratingbem._core.solver import calculate as _calculate
from gratingbem._core.adaptive import BalancedOptions, calculate_balanced
from .parallel import build_matrices_parallel
from .multigoal import multigoal_segment_indicator


@dataclass(frozen=True)
class Execution:
    workers: int = 4
    chunk_rows: int = 16
    multigoal: bool = True
    indicator_memory: float = 0.25

    def __post_init__(self):
        if not isinstance(self.workers,int) or not 1<=self.workers<=32:
            raise ValueError('workers in [1,32]')
        if not isinstance(self.chunk_rows,int) or self.chunk_rows<1:
            raise ValueError('positive integer chunk_rows required')
        if not 0<=self.indicator_memory<1:raise ValueError('indicator_memory in [0,1)')


def calculate(problem,numerics=None,polarizations=('TE','TM'),*,execution=None,_capture_internal=False,energy_contour=False):
    """Direct TE/TM periodic KM1; identical equations to v13.

    For line-only profiles, default v13 assembler is still used.
    """
    ex=execution or Execution()
    assembler=None
    if problem.profile.has_arcs:
        assembler=partial(build_matrices_parallel,workers=ex.workers,chunk_rows=ex.chunk_rows)
    return _calculate(problem,numerics,polarizations,_capture_internal=_capture_internal,
                      _assembly_function=assembler,energy_contour=energy_contour)


def calculate_adaptive(problem,numerics=None,options=None,polarizations=('TE','TM'),
                       *,execution=None,checkpoint_dir=None):
    ex=execution or Execution()
    fn=partial(calculate,execution=ex)
    return calculate_balanced(problem,numerics,options or BalancedOptions(),polarizations,
                             checkpoint_dir,calculator=fn,
                             indicator_function=(multigoal_segment_indicator if ex.multigoal else None),
                             indicator_memory=ex.indicator_memory if ex.multigoal else 0)


def scan(problem,parameter,values,numerics=None,polarizations=('TE','TM'),*,
         execution=None,update=None,on_error='raise',progress=None):
    """Parameter scan using the accelerated single-run solver when arcs exist."""
    from dataclasses import replace
    import time
    from gratingbem._core.solver import ScanResult,Numerics,Problem
    if on_error not in ('raise','record'):raise ValueError('on_error must be raise or record')
    t=time.perf_counter();vals=list(values);runs=[];errors=[]
    for i,val in enumerate(vals):
        out=None;err=None
        try:
            if parameter.startswith('numerics.') and update is None:
                key=parameter.split('.',1)[1]
                if key not in Numerics.__dataclass_fields__:raise ValueError(f'Unknown numerical field {key}')
                p=problem;n=replace(numerics or Numerics(),**{key:val})
            else:
                p=update(problem,val) if update else problem.with_parameter(parameter,val)
                n=numerics
            if not isinstance(p,Problem):raise TypeError('update must return Problem')
            out=calculate(p,n,polarizations,execution=execution)
        except Exception as ex:
            if on_error=='raise':raise
            err=f'{type(ex).__name__}: {ex}'
        runs.append(out);errors.append(err)
        if progress:progress(i+1,len(vals),val,out,err)
    return ScanResult(parameter,vals,runs,errors,dict(total_s=time.perf_counter()-t))


def scan_grid(problem,axes,numerics=None,polarizations=('TE','TM'),*,
              execution=None,on_error='raise',progress=None):
    """Cartesian geometry/material scan using the same accelerated backend."""
    import itertools
    if not axes:raise ValueError('axes may not be empty')
    keys=list(axes)
    if any(k=='profile' or k.startswith('numerics.') for k in keys):
        raise ValueError('grid axes must be Problem parameters, not numerics/profile objects')
    cases=[dict(zip(keys,v)) for v in itertools.product(*(list(axes[k]) for k in keys))]
    if not cases:raise ValueError('empty scan grid')
    def modify(base,case):
        for k,v in case.items():base=base.with_parameter(k,v)
        return base
    return scan(problem,'grid:'+','.join(keys),cases,numerics,polarizations,
                execution=execution,update=modify,on_error=on_error,progress=progress)
