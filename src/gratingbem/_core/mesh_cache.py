"""Conservative LRU for geometry-only Nyström nodes across parameter scans.

A mesh may be reused only when the profile, wavelength and mesh numerical
parameters are identical. Changes to Bloch alpha, TE/TM or metal epsilon do
not alter the quadrature mesh; a new per-call ``spec`` is always attached.
No Green operator or factorization is ever reused across different angles.
"""
from __future__ import annotations
from collections import OrderedDict
import json
import os
from threading import RLock
from .quadrature import Nodes

_CACHE = OrderedDict()
_MAX_ENTRIES = 8
_LOCK = RLock()


def cache_info():
    with _LOCK:
        return {'cached_meshes':len(_CACHE), 'max_entries':_MAX_ENTRIES}


def clear_geometry_cache():
    with _LOCK:
        _CACHE.clear()


def make_nodes_cached(problem,numerics,make_function):
    if os.environ.get('GRATINGBEM_GEOMETRY_CACHE','1').lower() in ('0','false','no'):
        return make_function(problem,numerics)
    key=json.dumps((problem.profile.as_dict(),problem.wavelength_um,
                    numerics.p_order,numerics.max_panel_wavelengths,
                    numerics.min_panels_per_segment,numerics.grading,
                    numerics.panels_per_edge,numerics.max_nodes),
                   sort_keys=True,default=str)
    # Geometry creation is short compared with kernel assembly. One lock keeps
    # a concurrent hit/eviction atomic, without locking electromagnetic solves.
    with _LOCK:
        old=_CACHE.get(key)
        if old is None:
            item=make_function(problem,numerics)
            _CACHE[key]=item
            if len(_CACHE)>_MAX_ENTRIES:
                _CACHE.popitem(last=False)
            return item
        _CACHE.move_to_end(key)
    sp=problem.spec()
    panels=dict(old.panels)
    panels['spec']=sp
    return Nodes(sp,panels,old.p_order,old.coords,old.normals,old.weights,
                 old.panel_idx,old.local_idx)
