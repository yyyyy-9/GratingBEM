"""Exact oriented straight and circular-arc segments, in micrometres.

Every segment is parameterized on t in [-1,1] at *constant arc-length speed*.
For this reason the on-panel logarithmic product-integration rule retains its
usual log(L/2) term even for circular arcs.  Normal is left normal of tangent
for a monotone left-to-right profile (metal below, air above).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import brentq


def _vec(x):
    out=np.asarray(x,dtype=float)
    if out.shape!=(2,) or not np.all(np.isfinite(out)):
        raise ValueError('points must be finite length-2 coordinates')
    return out


def _rotl(v):return np.stack((-v[...,1],v[...,0]),axis=-1)


@dataclass(frozen=True)
class Line:
    start:tuple[float,float]
    end:tuple[float,float]

    def __post_init__(self):
        a=_vec(self.start);b=_vec(self.end)
        if np.linalg.norm(b-a)<=1e-14:raise ValueError('zero-length line')
        object.__setattr__(self,'start',tuple(a))
        object.__setattr__(self,'end',tuple(b))

    @property
    def length(self):return float(np.linalg.norm(_vec(self.end)-_vec(self.start)))

    def eval(self,t):
        tt=np.asarray(t,dtype=float)
        return (1-tt[...,None])/2*_vec(self.start)+(1+tt[...,None])/2*_vec(self.end)

    def normal(self,t):
        tt=np.asarray(t,dtype=float)
        v=_rotl((_vec(self.end)-_vec(self.start))/self.length)
        return np.broadcast_to(v,tt.shape+(2,)).copy()

    def sub(self,a,b):
        return Line(tuple(self.eval(2*a-1)),tuple(self.eval(2*b-1)))

    def shift_x(self,d):return Line((self.start[0]+d,self.start[1]),(self.end[0]+d,self.end[1]))

    @property
    def sagitta(self):return 0.

    def as_dict(self):return dict(type='line',start=list(self.start),end=list(self.end))


@dataclass(frozen=True)
class Arc:
    center:tuple[float,float]
    radius:float
    theta0:float
    sweep:float

    def __post_init__(self):
        _vec(self.center)
        if not (np.isfinite(self.radius) and self.radius>0 and
                np.isfinite(self.theta0) and np.isfinite(self.sweep) and
                0<abs(self.sweep)<2*np.pi-1e-10):
            raise ValueError('arc needs radius>0 and sweep in (-2pi,2pi) excluding zero')
        if not self.x_monotone():
            raise ValueError('arc has decreasing x / overhang; unsupported periodic graph')

    @property
    def length(self):return float(abs(self.sweep)*self.radius)

    @property
    def sagitta(self):return float(self.radius*(1-math.cos(abs(self.sweep)/2)))

    def eval(self,t):
        t=np.asarray(t,float)
        a=self.theta0+(1+t)*self.sweep/2
        return _vec(self.center)+self.radius*np.stack((np.cos(a),np.sin(a)),axis=-1)

    def normal(self,t):
        t=np.asarray(t,float)
        a=self.theta0+(1+t)*self.sweep/2
        # x'(t)=-r sin a * sweep/2 ; y'(t)=r cos a * sweep/2.
        # left normal=(-y', x')/speed.
        return -np.sign(self.sweep)*np.stack((np.cos(a),np.sin(a)),axis=-1)

    @property
    def start(self):return tuple(float(x) for x in self.eval(-1.))

    @property
    def end(self):return tuple(float(x) for x in self.eval(1.))

    def x_monotone(self):
        # dx/dtheta * dtheta/dt = -r sin(theta) * sweep >= -tol,
        # including extrema of sin in the traversed interval.
        aa=np.linspace(self.theta0,self.theta0+self.sweep,max(8,int(abs(self.sweep)*24)))
        return bool(np.all(-np.sin(aa)*self.sweep>=-1e-11))

    def sub(self,a,b):
        if not (0<=a<b<=1):raise ValueError('subarc 0<=a<b<=1')
        return Arc(self.center,self.radius,self.theta0+a*self.sweep,(b-a)*self.sweep)

    def shift_x(self,d):return Arc((self.center[0]+d,self.center[1]),self.radius,self.theta0,self.sweep)

    def as_dict(self):return dict(type='arc',center=list(self.center),radius=self.radius,
                                  theta0=self.theta0,sweep=self.sweep)


def from_dict(d):
    if d['type']=='line':return Line(d['start'],d['end'])
    if d['type']=='arc':return Arc(tuple(d['center']),float(d['radius']),
                                   float(d['theta0']),float(d['sweep']))
    raise ValueError('unknown segment type')


def fillet_polyline(vertices,radii):
    """Fillet all corners, including periodic seam. No corner exponent assumed.

    Radii specified per vertex (including first=last periodic seam).  Fillets
    are true circular arcs tangent to both adjacent straight edges. All edges
    must remain non-overlapping after trimming. Returns line/arc sequence.
    """
    v=np.asarray(vertices,float);n=len(v)-1
    if n<2:raise ValueError('need at least two edges for fillets')
    rr=np.asarray(radii,float)
    if rr.shape!=(n+1,) or (rr<0).any() or not np.isfinite(rr).all():
        raise ValueError('radii must be nonnegative and provided at every vertex')
    if not np.isclose(rr[0],rr[-1],rtol=1e-10,atol=1e-13):
        raise ValueError('periodic seam first and last radii must be equal')
    d=v[-1,0]-v[0,0]
    dirs=np.diff(v,axis=0);ll=np.linalg.norm(dirs,axis=1)
    dirs/=ll[:,None]
    cut=np.zeros(n+1);corners=[None]*(n+1)
    for j in range(n):
        before=dirs[j-1] if j else dirs[-1]
        after=dirs[j]
        turn=float(np.arctan2(np.linalg.det(np.stack((before,after))), before@after))
        if rr[j]<1e-15 or abs(turn)<1e-12:continue
        if abs(turn)>np.pi-1e-7:raise ValueError('180-degree reverse turn cannot be filleted')
        t=rr[j]*np.tan(abs(turn)/2)
        cut[j]=t
        C=v[j]
        A=C-t*before;B=C+t*after
        cent=A+np.sign(turn)*rr[j]*_rotl(before)
        theta=float(np.arctan2(A[1]-cent[1],A[0]-cent[0]))
        arc=Arc(tuple(cent),float(rr[j]),theta,turn)
        if np.linalg.norm(np.array(arc.end)-B)>1e-9*max(1.,d):
            raise RuntimeError('fillet construction bug')
        corners[j]=(A,B,arc)
    cut[-1]=cut[0];corners[-1]=corners[0]
    for j in range(n):
        if cut[j]+cut[j+1]>=ll[j]-1e-12:
            raise ValueError(f'fillet radii overlap on edge {j}; reduce radii')

    pieces=[]
    seam=corners[0]
    if seam is not None:
        aa,bb,arc=seam
        if not aa[0]<v[0,0]+1e-10 or not bb[0]>v[0,0]-1e-10:
            raise ValueError('seam fillet requires a crossing, x-monotone periodic surface')
        s=brentq(lambda u: arc.eval(2*u-1)[0]-v[0,0],0.,1.,xtol=1e-14)
        after_seam=arc.sub(s,1)
        before_seam=arc.sub(0,s).shift_x(d)
        pieces.append(after_seam)
        point_start=np.array(after_seam.end)
    else:
        point_start=v[0]
    for j in range(n):
        if j==0:a=point_start
        else:a=np.asarray(corners[j][1] if corners[j] is not None else v[j])
        if j==n-1:
            b=np.array(before_seam.start) if seam is not None else v[-1]
        else:b=np.asarray(corners[j+1][0] if corners[j+1] is not None else v[j+1])
        if np.linalg.norm(b-a)>1e-12:pieces.append(Line(tuple(a),tuple(b)))
        if j<n-1 and corners[j+1] is not None:pieces.append(corners[j+1][2])
    if seam is not None:pieces.append(before_seam)
    return tuple(pieces)


def arc_through_radius(a,b,radius,turn='ccw',large_arc=False):
    """One of the two circles through endpoints, selected by sweep/large-arc.

    Endpoint tangent is NOT implied. Use fillet_polyline for tangent rounding.
    """
    a=_vec(a);b=_vec(b);v=b-a;chord=float(np.linalg.norm(v))
    if chord<=1e-14 or radius<chord/2-1e-13:raise ValueError('radius too small for arc endpoints')
    if turn not in ('ccw','cw'):raise ValueError('turn must be ccw or cw')
    midpoint=(a+b)/2
    s=math.sqrt(max(0.,radius**2-(chord/2)**2))
    opts=[]
    for sign in (-1,1):
        center=midpoint+sign*s*_rotl(v/chord)
        t0=math.atan2(a[1]-center[1],a[0]-center[0])
        t1=math.atan2(b[1]-center[1],b[0]-center[0])
        if turn=='ccw':sweep=(t1-t0)%(2*np.pi)
        else:sweep=-((t0-t1)%(2*np.pi))
        opts.append((center,t0,sweep))
    sorted_opts=sorted(opts,key=lambda q:abs(q[2]),reverse=large_arc)
    center,t0,sweep=sorted_opts[0]
    return Arc(tuple(center),float(radius),t0,sweep)
