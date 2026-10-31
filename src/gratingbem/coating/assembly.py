"""Resumable, bounded-time matrix assembly for *experimental* coated BIE.

No changes to the electromagnetic kernel or Nyström discretization are made by
this module. Every completed row block is durably checkpointed with a hash of
its geometry, quadrature and kernel parameters. A watchdog supervises workers;
a timed-out block is bisected, so a single expensive near-singular row cannot
silently hang the complete run. Errors are reported, never fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable
import json
import multiprocessing as mp
import os
import time
import uuid

import numpy as np
from .geometry import validate_interface_order
from .value_gradient import EwaldValueGradient
from .._core.quadrature import _poly_basis_data, basis_at, log_product_weights
from .._core.curved import nearest_periodic_chord, curved_panel_quadrature


class AssemblyTimeout(TimeoutError):
    """A bounded worker timed out after recursive splitting to one row."""


class AssemblyWorkerError(RuntimeError):
    """A worker crashed; any completed blocks remain checkpointed."""


@dataclass(frozen=True)
class AssemblyOptions:
    """Assembly management ONLY, not Maxwell numerical settings.

    workers: number of simultaneous isolated worker processes (1 by default).
    block_rows: checkpoint granularity in Nyström collocation rows.
    block_timeout_s: hard per-block wall-time limit, excluding process startup.
    min_block_rows: floor for timeout-triggered block splitting, usually 1.
    checkpoint_dir: None means no persistent checkpoint; still isolates workers.
    progress: optional on-progress event handler. Callback exceptions propagate
      AFTER the current block has been saved, so they simulate interruption.
    """
    checkpoint_dir: str | Path | None = None
    workers: int = 1
    block_rows: int = 8
    block_timeout_s: float = 90.0
    min_block_rows: int = 1
    resume: bool = True
    progress: Callable[[dict], None] | None = None

    def __post_init__(self):
        if not (1 <= self.workers <= 8):
            raise ValueError('workers must be between 1 and 8')
        if self.block_rows < 1 or self.min_block_rows < 1:
            raise ValueError('block_rows and min_block_rows must be positive')
        if not (np.isfinite(self.block_timeout_s) and self.block_timeout_s > 0):
            raise ValueError('block_timeout_s must be positive and finite')


def _matrix_rows(target, source, kernel, oversample, *, same, start, stop,
                 near_cutoff=2.5):
    """Exactly the legacy `_SD` quadrature, restricted to [start,stop) rows."""
    nt = len(target.coords); ns = len(source.coords)
    if not (0 <= start < stop <= nt):
        raise ValueError('invalid matrix row range')
    p = source.p_order
    coefficients = _poly_basis_data(p)[2]
    log_weights = log_product_weights(p)
    wGL = _poly_basis_data(p)[1]
    segments = source.panels['curves']
    period = source.spec.period_um
    S = np.empty((stop-start, ns), dtype=np.complex128)
    D = np.empty_like(S)
    Y = source.coords.copy()
    for local_i, i in enumerate(range(start,stop)):
        x = target.coords[i]
        normal = target.normals[i]
        if same:
            Y[i] = x + np.array([.0004731, .0008712])
        # Retain the verified legacy Ewald implementation / exact conventions.
        g,gx,gz = kernel.evaluate_vg(x[0]-Y[:,0], x[1]-Y[:,1])
        S[local_i,:] = g * source.weights
        D[local_i,:] = (normal[0]*gx+normal[1]*gz)*source.weights
        if same:
            Y[i] = source.coords[i]
        for j, segment in enumerate(segments):
            onself = same and j == target.panel_idx[i]
            distance, cut = nearest_periodic_chord(x, segment, period)
            if not (onself or distance <= near_cutoff * segment.length):
                continue
            if onself:
                cut = _poly_basis_data(p)[0][target.local_idx[i]]
                subdivisions = 1
            else:
                subdivisions = int(np.clip(np.ceil(np.log2(segment.length / max(
                    distance, .0001*segment.length))) + 2, 1, 15))
            t, weights, coords, _ = curved_panel_quadrature(
                segment, cut, oversample, subdivisions)
            g,gx,gz = kernel.evaluate_vg(x[0]-coords[:,0],x[1]-coords[:,1])
            if onself:
                r = np.linalg.norm(x - coords,axis=1)
                g += (1 / (2*np.pi)) * np.log(r)
            B = basis_at(t, coefficients)
            sl = slice(j*p,(j+1)*p)
            S[local_i,sl] = (weights*g)@B
            D[local_i,sl] = (weights*(normal[0]*gx+normal[1]*gz))@B
            if onself:
                half_length = segment.length/2
                analytic = half_length * (np.log(half_length)*wGL +
                                          log_weights[target.local_idx[i]])
                S[local_i,sl] -= analytic / (2*np.pi)
    return S,D


def _persistent_worker(worker_id, tasks, target, source, kernel, oversample,
                       same, checkpoint_root, fingerprint, output_queue):
    """One spawned process per worker, sequentially checkpointing its row tasks."""
    try:
        for a,b in tasks:
            start=time.monotonic()
            s,d=_matrix_rows(target,source,kernel,oversample,same=same,start=a,stop=b)
            _atomic_checkpoint(Path(checkpoint_root)/f'rows_{a:06d}_{b:06d}.npz',
                               s,d,fingerprint,a,b)
            output_queue.put(('block',worker_id,a,b,time.monotonic()-start))
        output_queue.put(('done',worker_id))
    except BaseException as exc:
        import traceback
        output_queue.put(('error',worker_id,type(exc).__name__,str(exc),
                          traceback.format_exc()))


def _fingerprint(target,source,kernel,oversample,same):
    digest=sha256()
    digest.update(b'gratingbem-coating-rowmatrix-legacy-quadrature-v1')
    for node in (target,source):
        for field in (node.coords,node.normals,node.weights,node.panel_idx,node.local_idx):
            arr=np.ascontiguousarray(field)
            digest.update(str(arr.shape).encode()); digest.update(str(arr.dtype).encode())
            digest.update(arr.tobytes())
        digest.update(repr(node.panels['curves']).encode())
        digest.update(str(node.spec.period_um).encode())
        digest.update(str(node.spec.alpha0).encode())
        digest.update(str(node.p_order).encode())
    opts={'k':str(kernel.k),'alpha0':kernel.alpha0,'period':kernel.period,
          'eta':kernel.eta,'images':kernel.images,'nterms':kernel.nterms,
          'method':kernel.method,'modes':(len(kernel.a)-1)//2,
          'oversample':oversample,'same':same,'near_cutoff':2.5}
    digest.update(json.dumps(opts,sort_keys=True).encode())
    return digest.hexdigest()


def _read_checkpoint(path,start,stop,ns,fp):
    if not path.is_file():
        return None
    try:
        with np.load(path,allow_pickle=False) as z:
            if (str(z['fingerprint']) != fp or int(z['start'])!=start
                or int(z['stop'])!=stop):
                return None
            s=np.array(z['S']);d=np.array(z['D'])
        if s.shape!=(stop-start,ns) or d.shape!=s.shape:
            return None
        if not (np.isfinite(s).all() and np.isfinite(d).all()):
            return None
        return s,d
    except (OSError,ValueError,KeyError,EOFError):
        return None


def _atomic_checkpoint(path, s,d,fp,start,stop):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.stem+'.tmp-'+uuid.uuid4().hex+'.npz')
    try:
        with tmp.open('wb') as f:
            np.savez(f,S=s,D=d,fingerprint=np.asarray(fp),start=np.asarray(start),stop=np.asarray(stop))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def _emit(options,event):
    if options.progress is not None:
        options.progress(event)


def _matrix_job(job,target,source,kernel,oversample,same,options:AssemblyOptions):
    nt=len(target.coords);ns=len(source.coords)
    fp=_fingerprint(target,source,kernel,oversample,same)
    # Persistent staging is needed even if no reusable checkpoint was requested,
    # because workers send only metadata through the queue, not huge arrays.
    import tempfile
    temp_context=(tempfile.TemporaryDirectory(prefix='gratingbem-assembly-')
                  if options.checkpoint_dir is None else None)
    root=(Path(temp_context.name) if temp_context else
          Path(options.checkpoint_dir)/(job+'_'+fp[:16]))
    root.mkdir(parents=True,exist_ok=True)
    config={'operator':job,'fingerprint':fp,'rows':nt,'cols':ns,
            'block_rows':options.block_rows,'format_version':1}
    _atomic_json(root/'job.json',config)
    S=np.empty((nt,ns),complex);D=np.empty_like(S)
    stats={'operator':job,'fingerprint':fp[:16],
           'restored_blocks':0,'computed_blocks':0,'worker_seconds':0.,
           'timeouts':0,'rows':nt,'cols':ns}
    finished=np.zeros(nt,bool)
    started=time.perf_counter()
    try:
        # Reuse ANY complete, valid contiguous row chunk from an earlier run.
        # This also handles automatic bisection following a timeout.
        if options.resume and options.checkpoint_dir is not None:
            found=[]
            for cp in root.glob('rows_*.npz'):
                try:
                    parts=cp.stem.split('_')
                    a,b=map(int,parts[-2:])
                except (ValueError,IndexError):
                    continue
                if 0 <= a < b <= nt:
                    found.append((a,b,cp))
            found.sort(key=lambda item: (-(item[1]-item[0]),item[0]))
            for a,b,cp in found:
                if finished[a:b].any():continue
                restored=_read_checkpoint(cp,a,b,ns,fp)
                if restored is None:continue
                S[a:b]=restored[0];D[a:b]=restored[1]
                finished[a:b]=True
                stats['restored_blocks']+=1
                _emit(options,dict(event='block_complete',operator=job,start=a,stop=b,
                                   restored=True,completed_rows=int(finished.sum()),
                                   total_rows=nt))

        def make_tasks(block_size):
            tasks=[]
            for a in range(0,nt):
                if finished[a]:continue
                if a>0 and not finished[a-1]:continue
                b=a
                while b<nt and not finished[b]: b+=1
                for st in range(a,b,block_size):
                    tasks.append((st,min(st+block_size,b)))
            return tasks

        size=options.block_rows
        current_workers=options.workers
        while not finished.all():
            tasks=make_tasks(size)
            nworkers=min(current_workers,len(tasks))
            ctx=mp.get_context('spawn')
            q=ctx.Queue()
            assignments=[tasks[i::nworkers] for i in range(nworkers)]
            procs=[ctx.Process(target=_persistent_worker,
                    args=(i,assignments[i],target,source,kernel,oversample,same,
                          str(root),fp,q)) for i in range(nworkers)]
            last_event={i:time.monotonic() for i in range(nworkers)}
            done=set()
            failure=None
            try:
                for pr in procs: pr.start()
                # Progress is sent after each block is fsync/atomic-replaced.
                while len(done)<nworkers:
                    import queue
                    try:
                        event=q.get(timeout=.25)
                    except queue.Empty:
                        event=None
                    now=time.monotonic()
                    if event is not None:
                        kind,wid,*rest=event
                        last_event[wid]=now
                        if kind=='block':
                            a,b,elapsed=rest
                            cp=root/f'rows_{a:06d}_{b:06d}.npz'
                            values=_read_checkpoint(cp,a,b,ns,fp)
                            if values is None:
                                raise AssemblyWorkerError(f'checkpoint corrupted for {job} [{a},{b})')
                            S[a:b],D[a:b]=values
                            finished[a:b]=True
                            stats['computed_blocks']+=1
                            stats['worker_seconds']+=elapsed
                            _emit(options,dict(event='block_complete',operator=job,
                                    start=a,stop=b,restored=False,block_seconds=elapsed,
                                    completed_rows=int(finished.sum()),total_rows=nt))
                        elif kind=='done':
                            done.add(wid)
                        elif kind=='error':
                            raise AssemblyWorkerError(f'{job} worker {wid}: {rest[0]}: '
                                                      f'{rest[1]}\n{rest[2]}')
                    for wid,pr in enumerate(procs):
                        if wid in done:continue
                        if pr.exitcode is not None and pr.exitcode != 0:
                            raise AssemblyWorkerError(f'{job} worker {wid} exited {pr.exitcode}; '
                                                      'finished blocks remain checkpointed')
                        if pr.exitcode == 0 and now-last_event[wid]>min(options.block_timeout_s,5.):
                            raise AssemblyWorkerError(f'{job} worker {wid} exited without completion acknowledgement')
                        if now-last_event[wid]>options.block_timeout_s:
                            failure=AssemblyTimeout(f'{job} worker {wid} unresponsive for '
                                    f'{options.block_timeout_s:g}s; completed blocks saved in {root}')
                            break
                    if failure is not None:break
            finally:
                for pr in procs:
                    if pr.is_alive():
                        pr.terminate()
                        pr.join(timeout=2)
                        if pr.is_alive():
                            pr.kill()
                    pr.join(timeout=2)
                q.close()
                q.join_thread()
            if failure is not None:
                stats['timeouts']+=1
                if size<=options.min_block_rows:
                    raise failure
                size=max(options.min_block_rows,size//2)
                current_workers=1  # reliability fallback after a parallel stall
                _emit(options,dict(event='timeout_retry',operator=job,
                       block_rows=size,completed_rows=int(finished.sum()),
                       total_rows=nt))
            elif not finished.all():
                raise AssemblyWorkerError(f'{job}: unfinished rows despite worker completion')
        _emit(options,dict(event='operator_complete',operator=job,rows=nt,
                           blocks=stats['computed_blocks'],restored=stats['restored_blocks'],
                           elapsed_seconds=time.perf_counter()-started))
        return (S,D),stats
    finally:
        if temp_context is not None:temp_context.cleanup()


def _atomic_json(path,data):
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        tmp.write_text(json.dumps(data,indent=2),encoding='utf-8')
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def assemble_coated(problem,numerics,options:AssemblyOptions,oversample=None):
    """Six physical operator pairs with row checkpoints and hard worker timeouts."""
    from dataclasses import replace
    from .._core.curved import make_curve_nodes
    t0=time.perf_counter()
    upper=problem.top_profile
    validate_interface_order(problem.substrate.profile,upper)
    outer=replace(problem.substrate,profile=upper)
    top=make_curve_nodes(outer,numerics)
    bottom=make_curve_nodes(problem.substrate,numerics)
    spec=problem.substrate.spec();k0=spec.k0
    ea=complex(problem.film_epsilon)
    es=complex(problem.substrate.epsilon_value)
    q=numerics.oversample if oversample is None else oversample
    kernels={
        'air':EwaldValueGradient(spec.period_um,k0,spec.alpha0,eta=numerics.eta,modes=numerics.modes_air),
        'film':EwaldValueGradient(spec.period_um,k0*np.sqrt(ea),spec.alpha0,
                            eta=numerics.eta,modes=numerics.modes_air),
        'substrate':EwaldValueGradient(spec.period_um,k0*np.sqrt(es),spec.alpha0,
                                 eta=numerics.eta,modes=numerics.modes_metal)
    }
    jobs=[('att',top,top,'air',True),('ftt',top,top,'film',True),
          ('ftb',top,bottom,'film',False),('fbt',bottom,top,'film',False),
          ('fbb',bottom,bottom,'film',True),('sbb',bottom,bottom,'substrate',True)]
    mats={};statistics=[]
    for key,t,s,material,same in jobs:
        _emit(options,dict(event='operator_start',operator=key,total_rows=len(t.coords)))
        mats[key],stat=_matrix_job(key,t,s,kernels[material],q,same,options)
        statistics.append(stat)
    return top,bottom,mats,time.perf_counter()-t0,statistics
