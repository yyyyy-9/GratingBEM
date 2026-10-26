"""Output-oriented, multi-order *heuristic* panel refinement scores.

Not a residual estimator, a posteriori error bound, or an adjoint solve.
Changes in each segment's *complex* Rayleigh amplitude over two meshes
are projected onto the total complex amplitude to capture their effect on
intensity (and a complex-amplitude safety component prevents phase blindness).
"""
from __future__ import annotations
import numpy as np
from gratingbem._core.modal import rayleigh_segment_contributions


def multigoal_segment_indicator(prev, curr, polarizations=('TE','TM'),
                                target_weight=.6,reflection_weight=.3,phase_weight=.1):
    """Goal-oriented inter-grid differences per analytic segment.

    Scales the response by beta_m/beta_inc and by the *total* complex field;
    therefore a strong order and an evanescent order are not treated equally.
    Scores remain positive despite cancellation of complex segment amplitudes.
    """
    if not np.isclose(target_weight+reflection_weight+phase_weight,1.):
        raise ValueError('weights must sum to one')
    weights=np.array([target_weight,reflection_weight,phase_weight],float)
    if np.any(weights<0):raise ValueError('weights must be nonnegative')
    pols=tuple(polarizations)
    a=rayleigh_segment_contributions(prev,pols)
    b=rayleigh_segment_contributions(curr,pols)
    common,ai,bi=np.intersect1d(prev.modes,curr.modes,return_indices=True)
    result=np.zeros(len(curr.problem.profile.geometric_segments),float)
    if not len(common):return np.ones_like(result)
    sp=curr.problem.spec()
    alpha=sp.alpha0+2*np.pi*common/sp.period_um
    beta=np.sqrt(sp.k0**2-alpha**2+0j)
    power=np.real(beta)/sp.beta0
    power=np.maximum(0,power)
    target_indices=np.flatnonzero(common==curr.problem.target_order)
    for pol in pols:
        old=a[pol][:,ai];new=b[pol][:,bi]
        delta=new-old
        middle=.5*(curr.amplitudes[pol][bi]+prev.amplitudes[pol][ai])
        # Signed first-order contribution to modal diffraction efficiencies.
        linear=power[None,:]*2*np.real(np.conj(middle)[None,:]*delta)
        phase=power[None,:]*2*np.abs(middle)[None,:]*np.abs(delta)
        if len(target_indices):
            result+=target_weight*np.abs(linear[:,target_indices[0]])
        result+=reflection_weight*np.mean(np.abs(linear),axis=1)
        # A pure phase change may be orthogonal to amplitude and should not
        # disappear from the refinement metric.
        result+=phase_weight*np.mean(phase,axis=1)
    if not np.all(np.isfinite(result)):raise ArithmeticError('nonfinite multigoal indicator')
    return result


def modal_intensity_change_by_segment(prev,curr,polarizations=('TE','TM')):
    """Return signed segment terms that sum to exact inter-grid power changes.

    The identity eta_new-eta_old = w_m*2 Re(conj((a_new+a_old)/2)
    *(a_new-a_old)) is algebraically exact.  It is NOT a true error bound,
    because both grids can share a discretization bias.
    """
    a=rayleigh_segment_contributions(prev,polarizations)
    b=rayleigh_segment_contributions(curr,polarizations)
    common,ia,ib=np.intersect1d(prev.modes,curr.modes,return_indices=True)
    sp=curr.problem.spec()
    beta=np.sqrt(sp.k0**2-(sp.alpha0+2*np.pi*common/sp.period_um)**2+0j)
    power=np.maximum(0.,np.real(beta)/sp.beta0)
    out={}
    for pol in polarizations:
        middle=(prev.amplitudes[pol][ia]+curr.amplitudes[pol][ib])/2
        out[pol]=power[None,:]*2*np.real(np.conj(middle)[None,:]*(b[pol][:,ib]-a[pol][:,ia]))
    return common,out
