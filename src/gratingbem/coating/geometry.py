"""Conformal offsets of v13 analytic line/arc periodic profiles.

Analytic arcs remain true circles, and line offsets remain true lines.
A periodic cell is cut at x=0 and x=d after normal offset, preserving
physical horizontal registration; no artificial lateral profile translation.
Only tangent-connected x-monotone profiles are accepted in auto mode.
Sharp corners require an explicit upper profile (or a rounding recipe).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from .._core.segments import Line, Arc
from .._core.profiles import Profile


def _offset(seg, t):
    if isinstance(seg,Line):
        n=seg.normal(0.)
        return Line(tuple(np.asarray(seg.start)+t*n),tuple(np.asarray(seg.end)+t*n))
    if isinstance(seg,Arc):
        rad=seg.radius-np.sign(seg.sweep)*t
        if rad<=0:raise ValueError('film thicker than local concave arc radius: conformal offset singular')
        return Arc(seg.center,float(rad),seg.theta0,seg.sweep)
    raise TypeError('expected Line or Arc')


def _cut_x(s,x):
    if abs(s.start[0]-x)<1e-11:return 0.
    if abs(s.end[0]-x)<1e-11:return 1.
    return float(brentq(lambda u:s.eval(2*u-1)[0]-x,0.,1.,xtol=1e-13))


def offset_profile(profile: Profile, thickness_um: float):
    """Normal offset into air, periodic re-cut at x=0..period.

    For a sharp corner a parallel offset needs geometric corner rules, so
    reject automatic construction rather than quietly create holes/overlap.
    """
    t=float(thickness_um)
    if not np.isfinite(t) or t<=0:raise ValueError('film thickness must be > 0')
    segments=profile.geometric_segments
    tol=1e-7*max(1.,profile.period_um)
    for a,b in zip(segments,segments[1:]):
        if np.linalg.norm(a.normal(1.)-b.normal(-1.))>5e-7:
            raise ValueError('non-tangent interface; apply circular fillets or supply explicit upper_profile')
    if np.linalg.norm(segments[-1].normal(1.)-segments[0].normal(-1.))>5e-7:
        raise ValueError('non-tangent periodic seam; fillet seam or supply explicit upper_profile')
    initial=[_offset(s,t) for s in segments];d=profile.period_um
    pieces=[]
    for shift in (-d,0.,d):
        for seg in initial:
            shifted=seg.shift_x(shift)
            if shifted.end[0]<=0+1e-12 or shifted.start[0]>=d-1e-12:continue
            left=max(shifted.start[0],0.)
            right=min(shifted.end[0],d)
            if right-left<=1e-13:continue
            u=_cut_x(shifted,left);v=_cut_x(shifted,right)
            if v-u<1e-12:continue
            pieces.append(shifted.sub(u,v))
    pieces.sort(key=lambda p:p.start[0])
    if not pieces:raise ValueError('offset lost whole period')
    if abs(pieces[0].start[0])>tol or abs(pieces[-1].end[0]-d)>tol:
        raise ValueError('periodic normal offset does not cover [0,d]')
    for a,b in zip(pieces,pieces[1:]):
        if np.linalg.norm(np.asarray(a.end)-b.start)>tol:
            raise ValueError('offset panels disconnected/self-intersecting; reduce thickness')
    if abs(pieces[-1].end[1]-pieces[0].start[1])>tol:
        raise ValueError('periodic offset violates seam height')
    return Profile.from_segments(d,pieces,'conformal_coating_outer')


def validate_interface_order(lower:Profile, upper:Profile, samples_per_segment:int=32) -> float:
    """Reject crossed/touching two-interface graphs; return sampled min vertical gap.

    This is a conservative *geometric precheck*, not a proof that arbitrarily
    close curved interfaces cannot cross between sample points. For this dev
    stage coated profiles must be strict x-monotone graphs (no vertical walls).
    """
    d=lower.period_um
    if not np.isclose(d,upper.period_um,rtol=0,atol=1e-9*max(1,d)):
        raise ValueError('film interfaces must have the same period')
    def sampled(profile):
        chunks=[]
        for segment in profile.geometric_segments:
            q=segment.eval(np.linspace(-1.,1.,samples_per_segment+1))
            if chunks: q=q[1:]
            chunks.append(q)
        xy=np.concatenate(chunks)
        delta=np.diff(xy[:,0]);tol=1e-12*max(1.,d)
        if np.any(delta<=tol):
            raise ValueError('coating requires strictly x-monotone line/arc interfaces; vertical walls unsupported')
        return xy
    a=sampled(lower);b=sampled(upper)
    x=np.unique(np.r_[np.linspace(0,d,2049),a[:,0],b[:,0]])
    ga=np.interp(x,a[:,0],a[:,1]);gb=np.interp(x,b[:,0],b[:,1])
    gap=float(np.min(gb-ga))
    if not np.isfinite(gap) or gap <= 1e-10*max(1,d):
        raise ValueError('film interfaces touch/cross, or thickness is below geometric tolerance')
    return gap
