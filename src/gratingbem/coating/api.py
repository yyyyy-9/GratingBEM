"""Opt-in coated-grating calculations; not yet an accuracy-certified solver."""
from __future__ import annotations
from dataclasses import replace
import time
from .model import CoatedSimulation, CoatedResult
from .periodic import calculate_coated as _run_raw
from .._core.solver import Numerics
from .planar import thin_film_r


def calculate(simulation:CoatedSimulation, numerics:Numerics|None=None,
              *, polarizations=('TE','TM'), assembly_options=None) -> CoatedResult:
    """Two-interface periodic single-layer BIE; reports validation status.

    No general convergence guarantee. Never rely on a single mesh, especially
    for metal TM, very thin films, sharp corners or deep/high-order gratings.
    """
    if not isinstance(simulation,CoatedSimulation):
        raise TypeError('requires CoatedSimulation, not bare Simulation')
    raw=_run_raw(simulation.to_problem(),numerics or Numerics(),pols=polarizations,
                 assembly_options=assembly_options)
    return CoatedResult(simulation,numerics or Numerics(),raw)


def planar_reflection(wavelength_nm:float, angle_deg:float, film_epsilon:complex,
                      substrate_epsilon:complex, thickness_nm:float, polarization='TE'):
    """Analytical single-layer reflection referred to the upper interface."""
    return thin_film_r(wavelength_nm/1000,angle_deg,film_epsilon,substrate_epsilon,
                       thickness_nm/1000,polarization.upper())


def scan_thickness(simulation:CoatedSimulation, thicknesses_nm,
                   numerics:Numerics|None=None,*,polarizations=('TE','TM'),
                   on_error='record', assembly_options=None):
    """Sequential film-thickness scan; saves each model and audit status."""
    if on_error not in ('record','raise'):
        raise ValueError('on_error must be record or raise')
    vals=list(thicknesses_nm)
    if not vals:raise ValueError('thicknesses_nm is empty')
    start=time.perf_counter();records=[];outputs=[]
    for t in vals:
        try:
            s=simulation.with_thickness(t)
            options=assembly_options
            if assembly_options is not None and assembly_options.checkpoint_dir is not None:
                options=replace(assembly_options,checkpoint_dir=str(assembly_options.checkpoint_dir)+'/'+f'case_{len(records):04d}')
            r=calculate(s,numerics,polarizations=polarizations,assembly_options=options)
            records.append(dict(thickness_nm=float(t),**r.summary()))
            outputs.append(r)
        except Exception as ex:
            if on_error=='raise':raise
            records.append({'thickness_nm':float(t),'status':'error','message':str(ex)})
            outputs.append(None)
    return {'records':records,'results':outputs,'total_seconds':time.perf_counter()-start}
