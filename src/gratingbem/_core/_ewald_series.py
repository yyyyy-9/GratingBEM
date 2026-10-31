"""Vectorized real-space Ewald recurrence, with an optional Numba accelerator.

No change of quadrature, physical Green function, summation cutoff, or truncation.
NUMBA is an OPTIONAL speed-up and is disabled with GRATINGBEM_NUMBA=0.
"""
from __future__ import annotations
import os
import threading
import numpy as np


def _series_numpy(t, et, e1, coeff):
    """Original verified NumPy recurrence, kept as the dependency-free path."""
    terms = [np.asarray(e1)]
    total = coeff[0]*terms[0]
    d1 = coeff[0]*et/t
    for order in range(1,len(coeff)):
        en=(et-t*terms[-1])/order
        terms.append(en)
        total += coeff[order]*en
        d1 += coeff[order]*terms[-2]
    d2 = coeff[0]*et*(1/t+1/(t*t))
    if len(coeff)>1:
        d2 += coeff[1]*et/t
    for order in range(2,len(coeff)):
        d2 += coeff[order]*terms[order-2]
    return total,d1,d2


# Do not eagerly import a heavy optional compiler when importing gratingbem.
_JIT = None
_ATTEMPTED = False
_JIT_LOCK = threading.Lock()


def _compiled():
    global _JIT, _ATTEMPTED
    if _ATTEMPTED:
        return _JIT
    with _JIT_LOCK:
        if _ATTEMPTED:
            return _JIT
        impl = _build_jit()
        _JIT = impl
        _ATTEMPTED = True
        return _JIT


def _build_jit():
    if os.environ.get('GRATINGBEM_NUMBA', '1').lower() in ('0','false','no','off'):
        return None
    try:
        from numba import njit
    except (ImportError, OSError):
        return None

    @njit(cache=True)
    def inner(t, et, e1, coeff):
        total = np.empty(len(t), dtype=np.complex128)
        d1 = np.empty(len(t), dtype=np.complex128)
        d2 = np.empty(len(t), dtype=np.complex128)
        for i in range(len(t)):
            ti = t[i]; eti = et[i]; prev = e1[i]
            prevprev = 0.
            acc = coeff[0] * prev
            der1 = coeff[0] * eti / ti
            der2 = coeff[0] * eti * (1. / ti + 1. / (ti*ti))
            for order in range(1, len(coeff)):
                nxt = (eti - ti * prev) / order
                acc += coeff[order] * nxt
                der1 += coeff[order] * prev
                if order == 1:
                    der2 += coeff[order] * eti / ti
                else:
                    der2 += coeff[order] * prevprev
                prevprev = prev
                prev = nxt
            total[i] = acc
            d1[i] = der1
            d2[i] = der2
        return total, d1, d2
    return inner

def ewald_series(t, et, e1, coeff):
    """Compute Ewald recurrence and its first/twice differentiated sums."""
    compiled = _compiled()
    if compiled is None:
        return _series_numpy(t, et, e1, coeff)
    return compiled(np.asarray(t, np.float64), np.asarray(et, np.float64),
                    np.asarray(e1, np.float64), np.asarray(coeff, np.complex128))
