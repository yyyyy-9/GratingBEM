"""Segment-resolved Rayleigh contributions for a posteriori *heuristic* refinement.

Uses field densities from a solved boundary integral system, not an imposed
corner singular exponent. Segment contributions add coherently to a mode's
complex amplitude; their intermesh change provides a refinement *indicator*,
not a mathematical estimator of absolute efficiency error.
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from gratingbem._core.segments import Line
from gratingbem._core.quadrature import _poly_basis_data,basis_at


def rayleigh_segment_contributions(calculation,polarizations=('TE','TM'),p_quad=32):
    """Return {pol: complex[n_original_edges, n_propagating_orders]}.

    Can operate on a Calculation made with calculate(...,_capture_internal=True).
    The sum over original edges must equal the solver's Rayleigh amplitudes
    up to quadrature tolerance. This is explicitly checked in unit tests.
    """
    data=calculation.internal
    if not data or not data.get('densities'):
        raise ValueError('requires calculate(...,_capture_internal=True)')
    nodes=data['nodes'];den=data['densities'];sp=nodes.spec
    d=sp.period_um;k0=sp.k0;modes=calculation.modes
    alpha=sp.alpha0+2*np.pi*modes/d
    beta=np.sqrt(k0*k0-alpha*alpha+0j)
    nsegments=len(calculation.problem.profile.geometric_segments)
    q,w=leggauss(p_quad)
    _,_,coeff=_poly_basis_data(nodes.p_order)
    basis=basis_at(q,coeff)
    panels=nodes.panels
    if 'curves' in panels:curves=panels['curves']
    else:curves=[Line(tuple(a),tuple(b)) for a,b in zip(panels['p'],panels['q'])]
    contributions={p:np.zeros((nsegments,len(modes)),complex) for p in polarizations}
    for j,curve in enumerate(curves):
        y=curve.eval(q);nn=curve.normal(q)
        phase=np.exp(-1j*(alpha[:,None]*y[None,:,0]+beta[:,None]*y[None,:,1]))
        dot=alpha[:,None]*nn[None,:,0]+beta[:,None]*nn[None,:,1]
        sl=slice(j*nodes.p_order,(j+1)*nodes.p_order)
        seg_idx=int(panels['facets'][j])
        for pol in polarizations:
            mu,rho=den[pol]
            muv=basis@mu[sl];rhov=basis@rho[sl]
            integrand=phase*(rhov[None,:]-1j*dot*muv[None,:])
            c=1j/(2*d*beta)*np.sum(integrand*w[None,:],axis=1)*(curve.length/2)
            contributions[pol][seg_idx,:]+=c
    return contributions


def modal_change_indicators(prev,curr,polarizations=('TE','TM')):
    """Segment-weighted change in complex target-mode contributions.

    Sum of segment differences can cancel, so the sum of absolute changes is
    deliberately used to flag potentially unresolved local contributions.
    """
    a=rayleigh_segment_contributions(prev,polarizations)
    b=rayleigh_segment_contributions(curr,polarizations)
    scores=np.zeros(len(curr.problem.profile.geometric_segments))
    for pol in polarizations:
        prev_modes=prev.modes;new_modes=curr.modes
        mode=curr.problem.target_order
        if mode in prev_modes and mode in new_modes:
            da=b[pol][:,np.where(new_modes==mode)[0][0]] - a[pol][:,np.where(prev_modes==mode)[0][0]]
            scores+=np.abs(da)
        # Include a small contribution from other propagating modes.
        common=np.intersect1d(prev_modes,new_modes)
        if len(common):
            delta=(b[pol][:,np.searchsorted(new_modes,common)]-
                   a[pol][:,np.searchsorted(prev_modes,common)])
            scores+=0.08*np.mean(np.abs(delta),axis=1)
    return scores


def allocate_refinement_panels(counts,scores,refine_factor):
    """Allocate a total panel-growth budget according to the edge indicators.

    The existing per-edge mesh is never coarsened; local nonzero floor avoids
    permanently starving edges whose contributions are temporarily small.
    """
    c=np.asarray(counts,int)
    e=np.asarray(scores,float)
    if len(e)!=len(c) or not np.all(np.isfinite(e)) or np.any(e<0):
        raise ValueError('invalid per-segment indicators')
    add=max(1,int(np.ceil(sum(c)/refine_factor))-sum(c))
    w=e+0.08*max(float(np.max(e)),1e-16)
    x=add*w/np.sum(w)
    more=np.floor(x).astype(int)
    k=add-int(sum(more))
    if k:more[np.argsort(-(x-np.floor(x)))[:k]]+=1
    assert sum(more)==add
    return tuple(map(int,c+more))
