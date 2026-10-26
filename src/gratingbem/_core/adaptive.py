"""Balanced refinement, independent discretization audits, and time logs.

The result is an *empirical* convergence assessment, never a certified
Maxwell error bound. No tuning against a target RCWA efficiency is performed.

Numerical kernels are delegated to the v13 curved KM1/Ewald solver.  The
v15 work is an independently testable convergence/mesh-management layer.
"""
from __future__ import annotations
from dataclasses import dataclass, replace, field
from pathlib import Path
import csv
import json
import math
import time
import numpy as np

from gratingbem._core.solver import Problem, Numerics, calculate
from gratingbem._core.solver import make_profile_nodes
from gratingbem._core.adaptive_util import _get_metrics, _meets
from gratingbem._core.modal import modal_change_indicators


def _integer_apportion(weights, total):
    """Nonnegative Hamilton largest-remainder apportionment (exact total)."""
    w = np.asarray(weights, float)
    if not np.all(np.isfinite(w)) or np.any(w < 0) or w.ndim != 1 or len(w) == 0:
        raise ValueError('invalid nonnegative weights')
    if int(total) != total or total < 0:
        raise ValueError('total must be a nonnegative integer')
    if total == 0:
        return np.zeros(len(w), dtype=int)
    if np.sum(w) == 0:
        w = np.ones_like(w)
    z = float(total) * w / np.sum(w)
    n = np.floor(z).astype(int)
    remaining = int(total - np.sum(n))
    if remaining:
        ids = np.argsort(-(z - n), kind='stable')[:remaining]
        n[ids] += 1
    return n


def max_graded_panel_lengths(lengths, counts, grade):
    """Physical maximum panel arclength per analytic segment, not mean length."""
    out = []
    for length, cnt in zip(lengths, counts):
        cnt = int(cnt)
        if cnt < 1: raise ValueError('counts must be >= 1')
        s = np.linspace(0, 1, cnt + 1)
        if grade > 1:
            a = s ** grade
            b = (1 - s) ** grade
            s = a / (a + b)
        out.append(float(length * np.max(np.diff(s))))
    return np.array(out)


def length_resolution_floor(lengths, wavelength, grading, panel_limit_lambda, min_counts=1):
    """Smallest integer panel counts meeting a graded *maximum* length cap."""
    lens = np.asarray(lengths, float)
    if wavelength <= 0 or panel_limit_lambda <= 0 or min_counts < 1:
        raise ValueError('invalid length limit')
    out = np.full(len(lens), int(min_counts), dtype=int)
    for j, length in enumerate(lens):
        while max_graded_panel_lengths([length], [out[j]], grading)[0] > wavelength * panel_limit_lambda:
            out[j] += 1
            if out[j] > 100_000: raise ValueError('unachievable mesh length requirement')
    return out


def balanced_refinement_counts(counts, indicators, lengths, grow=1.35,
                               global_fraction=.75, floors=None):
    """Grow ALL segments with a global backbone, add modal-guided surplus.

    This is intentionally conservative: local indicators are not adjoint or
    residual error estimators and are not allowed to starve long segments.
    """
    c = np.asarray(counts, int)
    length = np.asarray(lengths, float)
    scores = np.asarray(indicators, float)
    if c.ndim != 1 or len(c) == 0 or len(length) != len(c) or len(scores) != len(c):
        raise ValueError('inconsistent segment counts/lengths/scores')
    if np.any(c < 1) or np.any(length <= 0) or np.any(scores < 0) or not np.all(np.isfinite(scores)):
        raise ValueError('invalid counts, lengths, or indicators')
    if not 1 < grow <= 3 or not 0 <= global_fraction <= 1:
        raise ValueError('invalid growth/balance')
    target = max(int(np.ceil(c.sum() * grow)), int(c.sum()) + 1)
    extra = target - int(c.sum())
    # Length and existing DOF each account for half of the global backbone.
    baseline = .5 * (length / length.sum()) + .5 * (c / c.sum())
    if scores.sum():
        # Root-compress very spiky signals, then mix a small physical floor.
        raw = np.sqrt(scores / np.max(scores))
        loc = .92 * raw / raw.sum() + .08 * baseline
    else:
        loc = baseline
    n_global = int(round(extra * global_fraction))
    additions = _integer_apportion(baseline, n_global) + _integer_apportion(loc, extra - n_global)
    proposed = c + additions
    if floors is not None:
        minc = np.asarray(floors, int)
        if len(minc) != len(c) or np.any(minc < 1):
            raise ValueError('invalid physical floors')
        proposed = np.maximum(proposed, minc)
    if not np.all(proposed > c):
        # For a handful of faces, keep every segment improving, even if this
        # slightly exceeds the requested total growth budget.
        proposed = np.maximum(proposed, c + 1)
    return tuple(map(int, proposed))


@dataclass(frozen=True)
class BalancedOptions:
    target_abs_tol: float = 1e-3
    reflection_abs_tol: float = 2e-3
    order_l1_tol: float | None = None
    min_levels: int = 3
    max_levels: int = 5
    growth: float = 1.35
    global_fraction: float = .75
    panel_limit_lambda: float = 1.0
    # A p-check changes the density representation without changing geometry.
    polynomial_order_check: bool = True
    polynomial_increase: int = 2
    independent_kernel_check: bool = False
    kernel_modes_multiplier: float = 1.25
    kernel_oversample_increase: int = 4
    # A different mesh distribution at similar DOF probes allocation artifacts.
    independent_distribution_check: bool = True
    passivity_tolerance: float = 5e-4
    residual_tolerance: float = 1e-8
    max_wall_s: float | None = None
    # Audits on an unconverged final level are costly; enable for diagnostics.
    audit_unconverged: bool = False

    def __post_init__(self):
        if not (np.isfinite(self.target_abs_tol) and self.target_abs_tol > 0):raise ValueError('invalid target tolerance')
        if not (np.isfinite(self.reflection_abs_tol) and self.reflection_abs_tol > 0):raise ValueError('invalid reflection tolerance')
        if self.order_l1_tol is not None and self.order_l1_tol <= 0:raise ValueError('invalid order tolerance')
        if not 3 <= self.min_levels <= self.max_levels <= 30:raise ValueError('3 <= min_levels <= max_levels <= 30')
        if not 1 < self.growth <= 3:raise ValueError('invalid mesh growth')
        if not 0 <= self.global_fraction <= 1:raise ValueError('invalid global_fraction')
        if self.panel_limit_lambda <= 0:raise ValueError('invalid panel_limit_lambda')
        if self.polynomial_increase < 1:raise ValueError('invalid p-order increase')
        if self.max_wall_s is not None and self.max_wall_s <= 0:raise ValueError('invalid wall limit')
        if self.passivity_tolerance < 0 or self.residual_tolerance <= 0:raise ValueError('invalid validation tolerance')
        if self.kernel_modes_multiplier <= 1 or self.kernel_oversample_increase < 1:raise ValueError('invalid kernel audit')

    def as_dict(self):return dict(self.__dict__)


@dataclass
class BalancedRun:
    problem: Problem
    settings: BalancedOptions
    runs: list
    steps: list[dict]
    status: str
    audit: dict
    total_wall_s: float
    notes: list[str]
    checkpoint_dir: str | None = None

    @property
    def result(self):return self.runs[-1] if self.runs else None
    @property
    def converged(self):return self.status == 'empirically_converged'

    def summary(self):
        obj = dict(status=self.status, empirically_converged=self.converged,
                   accuracy_certified=False, levels=len(self.runs),
                   total_wall_s=self.total_wall_s, audit=self.audit,
                   options=self.settings.as_dict(), notes=self.notes)
        if self.result is not None:obj['latest_result'] = self.result.summary()
        return obj

    def save(self, path, *, save_all_runs=True):
        dest = Path(path);dest.mkdir(parents=True,exist_ok=True)
        if save_all_runs:
            for i, result in enumerate(self.runs):result.save(dest / f'level_{i:02d}')
        if self.steps:
            names = list(dict.fromkeys(k for row in self.steps for k in row))
            with (dest/'adaptive_history.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(self.steps)
        (dest/'adaptive_summary.json').write_text(json.dumps(self.summary(),ensure_ascii=False,indent=2),encoding='utf-8')
        return dest


def _validation_differences(original, check, pols):
    row={'wall_s':check.timings['total_s'], 'n_nodes':check.diagnostics['n_nodes']}
    passing=True
    for pol in pols:
        eo=original.diagnostics[pol]['target_efficiency']; ec=check.diagnostics[pol]['target_efficiency']
        dtarget = float('inf') if eo is None or ec is None else 100 * abs(ec - eo)
        dreflect = 100 * abs(check.diagnostics[pol]['reflection_sum'] - original.diagnostics[pol]['reflection_sum'])
        row[pol+'_target_delta_pp']=dtarget
        row[pol+'_reflect_delta_pp']=dreflect
        if check.diagnostics[pol]['reflection_sum']>1.0005:passing=False
    return row,passing


def calculate_balanced(problem:Problem,numerics:Numerics|None=None,
                       options:BalancedOptions|None=None,polarizations=('TE','TM'),
                       checkpoint_dir: str|Path|None=None,*,
                       calculator=None,indicator_function=None,
                       indicator_memory:float=0.0)->BalancedRun:
    """Adaptive periodic KM1 with growth safeguards and independent audits.

    "Empirically converged" means independently altered numerical settings
    agree within chosen efficiency thresholds. It does NOT imply a rigorous
    error bound or that RCWA/surface-coating uncertainty has been eliminated.
    """
    num=numerics or Numerics(); opts=options or BalancedOptions()
    calc=calculate if calculator is None else calculator
    if not 0.0<=indicator_memory<1.0:raise ValueError("indicator_memory must be in [0,1)")
    pols=tuple(str(p).upper() for p in polarizations)
    if not pols or len(set(pols)) != len(pols) or any(p not in ('TE','TM') for p in pols):
        raise ValueError('polarizations must be unique TE/TM')
    if num.panels_per_edge is not None:raise ValueError('v15 manages panels_per_edge internally')
    t0=time.perf_counter();notes=[];runs=[];steps=[];audit={};status='not_converged'
    try:
        base=make_profile_nodes(problem,num)
    except (ValueError,MemoryError) as ex:
        return BalancedRun(problem,opts,[],[],'resource_limited',{},time.perf_counter()-t0,
                           [f'Baseline mesh: {type(ex).__name__}: {ex}'],str(checkpoint_dir) if checkpoint_dir else None)
    counts=tuple(map(int,base.panels['counts_by_edge']))
    lengths=np.asarray(problem.profile.lengths_um,float)
    if len(counts)!=len(lengths):raise RuntimeError('profile count mismatch')
    floors=length_resolution_floor(lengths,problem.wavelength_um,num.grading,
                                    opts.panel_limit_lambda,min_counts=num.min_panels_per_segment)
    counts=tuple(map(int,np.maximum(counts,floors)))
    pending_indicator=np.zeros(len(counts))
    for level in range(opts.max_levels):
        if level>0 and opts.max_wall_s is not None and time.perf_counter()-t0>opts.max_wall_s:
            status='time_limit'; notes.append('wall budget reached before next level');break
        if level>0:
            counts=balanced_refinement_counts(counts,pending_indicator,lengths,
                                              grow=opts.growth,global_fraction=opts.global_fraction,
                                              floors=floors)
        current_num=replace(num,panels_per_edge=counts)
        try:
            current=calc(problem,current_num,pols,_capture_internal=True)
        except (MemoryError,ValueError) as ex:
            status='resource_limited';notes.append(f'level {level}: {type(ex).__name__}: {ex}');break
        previous=runs[-1] if runs else None
        row=dict(level=level,p_order=num.p_order,counts=json.dumps(counts),
                 n_nodes=current.diagnostics['n_nodes'],n_panels=current.diagnostics['n_panels'],
                 max_panel_lambda=float(np.max(max_graded_panel_lengths(lengths,counts,num.grading))/problem.wavelength_um),
                 total_s=current.timings['total_s'],assembly_s=current.timings['assembly_s'],
                 solve_s=current.timings['solve_s'])
        metrics, errors=_get_metrics(current,previous,pols)
        row.update(metrics)
        runs.append(current)
        if previous is not None:
            new_indicator=(indicator_function(previous,current,pols) if indicator_function is not None
                           else modal_change_indicators(previous,current,pols))
            pending_indicator=indicator_memory*pending_indicator + (1-indicator_memory)*new_indicator
            row['segment_modal_indicators']=json.dumps([float(x) for x in pending_indicator])
        steps.append(row)
        if checkpoint_dir is not None:
            path=Path(checkpoint_dir);path.mkdir(parents=True,exist_ok=True)
            current.save(path/f'level_{level:02d}')
            # A complete checkpoint even if a later run is interrupted.
            with (path/'adaptive_history.csv').open('w',newline='',encoding='utf-8') as f:
                cols=list(dict.fromkeys(k for r in steps for k in r))
                writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows(steps)
        if errors:
            status='numerical_invalid';notes.extend(errors);break
        if _meets(runs,steps,opts,pols):
            status='empirically_converged';break
    if status=='not_converged' and runs:
        notes.append('Maximum refinement levels reached: differences not yet within tolerance')
    if not runs and status=='not_converged':status='no_result'
    if runs:
        for pol in pols:
            chk=runs[-1].diagnostics[pol]
            if chk['reflection_sum']>1+opts.passivity_tolerance:
                status='physical_check_failed';notes.append(f'{pol}: total reflection violates passivity')
            if chk['matrix_residual']>opts.residual_tolerance:
                status='numerical_invalid';notes.append(f'{pol}: linear system residual exceeds tolerance')
    # Optional audits run for converged grids, or with audit_unconverged=True
    # for transparency. An audit cannot turn a nonconverged run into a pass.
    if runs and (status=='empirically_converged' or opts.audit_unconverged):
        baseline=runs[-1];ns=baseline.numerics
        if opts.polynomial_order_check:
            if ns.p_order+opts.polynomial_increase>12:
                audit['polynomial_order']={'passed':False,'reason':'p_order maximum is 12'}
                if status=='empirically_converged':status='polynomial_check_failed'
            else:
                changed=replace(ns,p_order=ns.p_order+opts.polynomial_increase)
                try:
                    order_run=calc(problem,changed,pols)
                    entry,physical=_validation_differences(baseline,order_run,pols)
                    passed=physical and all(entry[f'{p}_target_delta_pp']<=100*opts.target_abs_tol and
                                            entry[f'{p}_reflect_delta_pp']<=100*opts.reflection_abs_tol for p in pols)
                    audit['polynomial_order']={**entry,'tested_p_order':changed.p_order,'passed':passed}
                    if not passed and status=='empirically_converged':status='polynomial_check_failed'
                except (ValueError,MemoryError) as ex:
                    audit['polynomial_order']={'passed':False,'error':f'{type(ex).__name__}: {ex}'}
                    if status=='empirically_converged':status='polynomial_check_incomplete'
        if opts.independent_kernel_check:
            changed=replace(ns,modes_air=max(ns.modes_air+1,math.ceil(ns.modes_air*opts.kernel_modes_multiplier)),
                            modes_metal=max(ns.modes_metal+1,math.ceil(ns.modes_metal*opts.kernel_modes_multiplier)),
                            oversample=ns.oversample+opts.kernel_oversample_increase)
            krun=calc(problem,changed,pols)
            entry,physical=_validation_differences(baseline,krun,pols)
            passed=physical and all(entry[f'{p}_target_delta_pp']<=100*opts.target_abs_tol and
                                    entry[f'{p}_reflect_delta_pp']<=100*opts.reflection_abs_tol for p in pols)
            audit['kernel']={**entry,'modes_air':changed.modes_air,'oversample':changed.oversample,'passed':passed}
            if not passed and status=='empirically_converged':status='kernel_check_failed'
        if opts.independent_distribution_check:
            total_panels=sum(ns.panels_per_edge)
            min_keep=length_resolution_floor(lengths,problem.wavelength_um,ns.grading,
                opts.panel_limit_lambda,min_counts=num.min_panels_per_segment)
            if total_panels < int(np.sum(min_keep)):
                min_keep=np.ones_like(min_keep)
            base_uniform=_integer_apportion(lengths,total_panels-int(np.sum(min_keep))) + min_keep
            try:
                # Match node count while changing spatial allocation; if the
                # partition is unchanged, skip rather than claim independence.
                if tuple(base_uniform)==tuple(ns.panels_per_edge):
                    audit['distribution']={'passed':False,'reason':'independent distribution identical'}
                    if status=='empirically_converged':status='distribution_check_incomplete'
                else:
                    independent=replace(ns,panels_per_edge=tuple(map(int,base_uniform)))
                    urun=calc(problem,independent,pols)
                    entry,physical=_validation_differences(baseline,urun,pols)
                    passed=physical and all(entry[f'{p}_target_delta_pp']<=100*opts.target_abs_tol and
                                            entry[f'{p}_reflect_delta_pp']<=100*opts.reflection_abs_tol for p in pols)
                    audit['distribution']={**entry,'counts':list(map(int,base_uniform)),'passed':passed}
                    if not passed and status=='empirically_converged':status='distribution_check_failed'
            except (ValueError,MemoryError) as ex:
                audit['distribution']={'passed':False,'error':f'{type(ex).__name__}: {ex}'}
                if status=='empirically_converged':status='distribution_check_incomplete'
    notes.insert(0,'Experimental stability checks only. Not a rigorous error bound, independent RCWA comparison, or absorption-flux validation.')
    result=BalancedRun(problem,opts,runs,steps,status,audit,time.perf_counter()-t0,notes,
                       str(checkpoint_dir) if checkpoint_dir is not None else None)
    if checkpoint_dir is not None:
        result.save(checkpoint_dir,save_all_runs=False)
    return result
