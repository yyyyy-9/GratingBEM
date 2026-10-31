"""Independent parameter-point scheduling; no alteration of Maxwell discretization.

Never nests scan-level threads and assembly-level processes, avoiding CPU oversubscription.
Returns results in INPUT ORDER, even when worker completion order differs.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from dataclasses import dataclass, field
import time
from collections.abc import Sequence, Callable

from ._core.solver import Calculation, Numerics
from .model import Simulation


@dataclass
class BatchResult:
    results: list[Calculation | None]
    errors: list[str | None]
    elapsed_s: float
    group_stats: list[dict] = field(default_factory=list)


def _run_one(simulation, numerics, polarizations, workers):
    # Resolved inside the worker; avoid sending expensive kernel instances over IPC.
    from .api import calculate
    return calculate(simulation, numerics, polarizations=polarizations, workers=workers)


def _policy(value):
    """Backward compatible bools, plus automatic gate and explicit off switch."""
    if value is True or value == 'on':
        return 'auto'
    if value is False or value == 'off':
        return 'off'
    if value == 'auto':
        return 'auto'
    raise ValueError("bloch_reuse must be 'auto', 'off', True, or False")


def _run_block(block, numerics, polarizations, workers, try_reuse, on_error):
    """Execute one Bloch equivalence class inside ONE scheduling worker.

    Gate and assembly stay in the worker, so distinct Bloch classes execute
    concurrently without moving LU matrices/densities between processes.
    Falls back to individual full solves when the sampled finite-Ewald kernel
    fails equivalence. Errors are isolated by original input case.
    """
    n=len(block)
    stats={'size':n, 'status':'independent', 'kernel_sample_rel_error':None}
    gate_error=None
    if try_reuse and n>1 and workers==1:
        from ._scan_bloch import shareable, calculate_group
        can_share=False
        try:
            can_share,delta=shareable(block,numerics)
            stats['kernel_sample_rel_error']=float(delta)
        except Exception as exc:
            gate_error=f'{type(exc).__name__}: {exc}'
        if can_share:
            try:
                group=calculate_group(block,numerics,polarizations,
                    kernel_discrepancy_value=stats['kernel_sample_rel_error'])
                stats['status']='reused'
                stats['shared_lu']=True
                return list(zip(group,[None]*n)),stats
            except Exception as exc:
                # The group-specific optimization is optional. An unexpected
                # failure must never silently lose an otherwise valid case.
                gate_error=f'{type(exc).__name__}: {exc}'
        stats['status']='fallback'
        if gate_error is not None:
            stats['gate_error']=gate_error
    elif try_reuse and n>1:
        stats['status']='disabled_due_to_assembly_workers'

    answer=[]
    for simulation in block:
        try:
            result=_run_one(simulation,numerics,polarizations,workers)
            if stats['status']=='fallback':
                msg='Bloch reuse refused; independent full KM1 solve used'
                if gate_error is not None:
                    msg+=f' (gate: {gate_error})'
                result.warning_messages.append(msg)
            answer.append((result,None))
        except Exception as exc:
            if on_error=='raise':
                raise
            answer.append((None,f'{type(exc).__name__}: {exc}'))
    return answer,stats


def calculate_many(simulations: Sequence[Simulation], numerics: Numerics | None = None,
                   *, polarizations=('TE','TM'), scan_workers=1, workers=1,
                   on_error='raise', progress: Callable | None = None,
                   scan_backend: str = 'thread', bloch_reuse: bool | str = 'auto'):
    """Run scan with automatic Bloch grouping and safe finite-kernel fallback.

    `bloch_reuse='auto'` (default): identical wavelength/profile/material and
    reciprocal-lattice-equivalent incidence angles share a KM1 matrix and LU
    only if a finite-Ewald kernel check passes. Independent equivalence groups
    can also run in different processes via scan_backend='process'.

    `bloch_reuse=False` uses the independent-point scheduler.
    `scan_workers=1` does not launch a pool; `workers` selects single-case
    assembly workers. Both worker levels >1 are deliberately rejected.
    Completed results and errors are in INPUT ORDER; progress callbacks are
    run only in the calling process, not inside worker processes.
    """
    if scan_backend not in ('thread','process'):
        raise ValueError("scan_backend must be 'thread' or 'process'")
    if not isinstance(scan_workers,int) or not 1<=scan_workers<=64:
        raise ValueError('scan_workers must be an integer in 1..64')
    if not isinstance(workers,int) or not 1<=workers<=32:
        raise ValueError('workers must be an integer in 1..32')
    policy=_policy(bloch_reuse)
    if scan_workers>1 and workers>1:
        raise ValueError('Nested scan_workers>1 and workers>1 would oversubscribe CPUs; choose one level')
    if on_error not in ('raise','record'):
        raise ValueError("on_error must be 'raise' or 'record'")
    cases=list(simulations)
    if not cases:
        raise ValueError('simulations must be nonempty')
    if any(not isinstance(s,Simulation) for s in cases):
        raise TypeError('calculate_many needs Simulation objects')
    results=[None]*len(cases)
    errors=[None]*len(cases)
    stats=[]
    t0=time.perf_counter()
    if policy=='auto':
        from ._scan_bloch import groups
        blocks=groups(cases)
    else:
        blocks=[[i] for i in range(len(cases))]
    num=numerics or Numerics()
    pols=tuple(str(p).upper() for p in polarizations)
    completed=0

    def handle_block(indices,payload,stat):
        nonlocal completed
        stat['input_indices']=list(indices)
        stats.append(stat)
        for i,(result,error) in zip(indices,payload):
            results[i]=result;errors[i]=error
            completed+=1
            if progress:
                progress(completed,len(cases),i,result,error)

    if scan_workers==1 or len(blocks)==1:
        for indices in blocks:
            payload,stat=_run_block([cases[i] for i in indices],num,pols,
                                    workers,policy=='auto',on_error)
            handle_block(indices,payload,stat)
    else:
        executor_type=ProcessPoolExecutor if scan_backend=='process' else ThreadPoolExecutor
        options=({'mp_context':get_context('spawn')} if scan_backend=='process' else {})
        # In the process variant each independent block owns its mesh cache,
        # KM1 assembly, LU factorization and temporary arrays.
        with executor_type(max_workers=scan_workers,**options) as executor:
            futures={executor.submit(_run_block,[cases[i] for i in indices],
                                     num,pols,1,policy=='auto',on_error):indices
                     for indices in blocks}
            for future in as_completed(futures):
                indices=futures[future]
                try:
                    payload,stat=future.result()
                except Exception:
                    if on_error=='raise':
                        for other in futures:other.cancel()
                        raise
                    # The worker itself failed (e.g. failed process), not just
                    # a simulation: record per-case errors without reordering.
                    why=f'{type(future.exception()).__name__}: {future.exception()}'
                    payload=[(None,why)]*len(indices)
                    stat={'size':len(indices),'status':'worker_error'}
                handle_block(indices,payload,stat)
    stats.sort(key=lambda x: min(x['input_indices']))
    return BatchResult(results,errors,time.perf_counter()-t0,stats)
