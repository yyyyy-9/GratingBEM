"""Exact circular-arc Nyström integration for the 2-density periodic KM1 BIE.

Curved panels use a constant arc-length parameterization, exact normals and
Jacobian. Self-panel S and (T2-T1) logarithms use analytical Gauss product
weights. Adjacent near kernels are recomputed via split/adaptive Gauss rules.
These are *geometric* high-order Nyström rules; no material-specific corner
exponents and no claim of full RCIP validation.
"""
from __future__ import annotations
import math
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from gratingbem._core.quadrature import Nodes,_poly_basis_data,basis_at,log_product_weights
from gratingbem._core.ewald import EwaldHessian
from gratingbem._core.periodic import kernel_components
from .segments import Line,Arc


def make_curve_nodes(problem,num):
    sp=problem.spec()
    segments=problem.profile.geometric_segments
    counts=[];curves=[];which=[];p0=[];p1=[];ds=[];nn=[]
    if num.panels_per_edge is not None and len(num.panels_per_edge)!=len(segments):
        raise ValueError('panels_per_edge must have one entry per analytic line/arc segment')
    hmax=num.max_panel_wavelengths*problem.wavelength_um
    for j,seg in enumerate(segments):
        n=int(num.panels_per_edge[j]) if num.panels_per_edge is not None else max(
            num.min_panels_per_segment,math.ceil(seg.length/hmax))
        counts.append(n)
        ss=np.linspace(0,1,n+1)
        if num.grading>1:
            ss=ss**num.grading/(ss**num.grading+(1-ss)**num.grading)
        for a,b in zip(ss[:-1],ss[1:]):
            curve=seg.sub(float(a),float(b))
            curves.append(curve);which.append(j)
            p0.append(curve.start);p1.append(curve.end)
            ds.append(curve.length);nn.append(curve.normal(0.))
    lens=np.asarray(ds)
    if (lens<=1e-13).any():raise ValueError('degenerate micro-panel; reduce grading')
    p=np.asarray(p0);q=np.asarray(p1)
    gl,ww,_=_poly_basis_data(num.p_order)
    xx=np.concatenate([c.eval(gl) for c in curves],axis=0)
    norms=np.concatenate([c.normal(gl) for c in curves],axis=0)
    weights=np.repeat(lens,num.p_order)*np.tile(ww/2,len(curves))
    nd=len(xx)
    if nd>num.max_nodes:
        raise ValueError(f'{nd} unknown nodes exceed max_nodes={num.max_nodes}')
    panel=dict(p=p,q=q,mid=np.asarray([c.eval(0.) for c in curves]),ds=lens,
               n=np.asarray(nn),curves=tuple(curves),facets=np.asarray(which,int),
               counts_by_edge=np.asarray(counts,int),flat=False,spec=sp)
    return Nodes(sp,panel,num.p_order,xx,norms,weights,
                 np.repeat(np.arange(len(curves)),num.p_order),
                 np.tile(np.arange(num.p_order),len(curves)))


def _distance_to_chord(point,a,b):
    v=b-a;length2=max(v@v,1e-28)
    t=np.clip((point-a)@v/length2,0,1)
    return np.linalg.norm(point-a-t*v),2*t-1


def nearest_periodic_chord(x,curve,period):
    a=np.asarray(curve.start);b=np.asarray(curve.end)
    closest=(np.inf,0.)
    for shift in (-period,0.,period):
        dist,tt=_distance_to_chord(x,np.array([shift,0])+a,np.array([shift,0])+b)
        if dist<closest[0]:closest=(dist,tt)
    return max(0.,closest[0]-curve.sagitta),closest[1]


@lru_cache(maxsize=16)
def _leggauss_cached(nq):
    """Reuse immutable Gaussian rules: the old code recalculated them per pair."""
    x,w=leggauss(nq)
    x.setflags(write=False);w.setflags(write=False)
    return x,w


def curved_panel_quadrature(curve,cut,nq,subdivisions):
    """Geometrically graded Gauss subintervals toward the near-target cut.

    This replaces expensive uniform 1/d subdivisions; the smallest interval
    is roughly 2**(-subdivisions) of the source half panel.
    """
    a=np.clip(float(cut),-1.,1.)
    bounds=[-1.,a,1.] if abs(a)<.999999999 else [-1.,1.]
    g,w=_leggauss_cached(nq)
    ts=[];ws=[]
    for l,r in zip(bounds[:-1],bounds[1:]):
        if subdivisions<=1:
            steps=[l,r]
        elif abs(a-l)<1e-12: # left end is the point near target
            steps=[l]+[l+(r-l)*2.**(-j) for j in range(subdivisions,0,-1)]+[r]
        elif abs(a-r)<1e-12: # right end is near target
            steps=[l]+[r-(r-l)*2.**(-j) for j in range(1,subdivisions+1)]+[r]
        else:
            steps=[l,r]
        steps=np.asarray(sorted(set(steps)))
        for lo,hi in zip(steps[:-1],steps[1:]):
            if hi-lo<1e-14:continue
            ts.append(.5*(hi-lo)*g+.5*(lo+hi))
            ws.append(.5*(hi-lo)*w)
    t=np.concatenate(ts);w=np.concatenate(ws)*(curve.length/2)
    return t,w,curve.eval(t),curve.normal(t)


def build_matrices_curved(nodes,eps,mode_air=155,mode_metal=25,oversample=16,eta=5.,
                          near_cutoff=1.7,verbose=False):
    sp=nodes.spec;k1=sp.k0;k2=k1*np.sqrt(eps+0j)
    air=EwaldHessian(sp.period_um,k1,sp.alpha0,eta=eta,modes=mode_air)
    metal=EwaldHessian(sp.period_um,k2,sp.alpha0,eta=eta,modes=mode_metal)
    N=len(nodes.coords);p=nodes.p_order;crvs=nodes.panels['curves'];npan=len(crvs)
    _,wGL,co=_poly_basis_data(p)
    logw=log_product_weights(p)
    keys=('S1','S2','K1','K2','Ka1','Ka2','Td')
    mats={k:np.zeros((N,N),complex) for k in keys}
    cT=-(k2*k2-k1*k1)/(2*np.pi)
    for i in range(N):
        x=nodes.coords[i];normal=nodes.normals[i];ip=int(nodes.panel_idx[i])
        Y=nodes.coords.copy();Y[i]=x+np.array([.00053,.00081])
        raw=kernel_components(k1,k2,air,metal,x,normal,Y,nodes.normals)
        for c,k in enumerate(keys):mats[k][i,:]=raw[:,c]*nodes.weights
        for j,curve in enumerate(crvs):
            same=(j==ip)
            dist,cut=nearest_periodic_chord(x,curve,sp.period_um)
            if not (same or dist<=near_cutoff*curve.length):continue
            # For self the exact node parameter is used to split the singular
            # integrand, NOT a projected chord parameter.
            if same:cut=_poly_basis_data(p)[0][nodes.local_idx[i]];sub=1
            else:sub=int(np.clip(math.ceil(math.log2(curve.length/max(dist,.0005*curve.length)))+1,1,12))
            t,wq,yq,nq=curved_panel_quadrature(curve,cut,oversample,sub)
            vals=kernel_components(k1,k2,air,metal,x,normal,yq,nq)
            if same:
                r=np.linalg.norm(x-yq,axis=1)
                vals[:,0:2]+=(1/np.pi)*np.log(r)[:,None]
                vals[:,6]-=cT*np.log(r)
            B=basis_at(t,co)
            dense=(wq[:,None]*vals).T@B
            if same:
                # Exact log|t-t_i| moment; arc-length parametrization gives
                # J=L/2. Shape-dependent regular remainder is integrated above.
                lg=(curve.length/2)*(np.log(curve.length/2)*wGL+logw[nodes.local_idx[i]])
                dense[0,:]+=-lg/np.pi
                dense[1,:]+=-lg/np.pi
                dense[6,:]+=cT*lg
            sl=slice(j*p,(j+1)*p)
            for c,k in enumerate(keys):mats[k][i,sl]=dense[c,:]
        if verbose and (i+1)%max(1,N//5)==0:print(f'curved KM1 {i+1}/{N}',flush=True)
    return mats


def rayleigh_curved(nodes,mu,rho,p_quad=32):
    sp=nodes.spec;d=sp.period_um;k1=sp.k0
    mmin=int(np.ceil((-k1-sp.alpha0)*d/(2*np.pi)+1e-9))
    mmax=int(np.floor((k1-sp.alpha0)*d/(2*np.pi)-1e-9))
    modes=np.arange(mmin,mmax+1)
    alpha=sp.alpha0+2*np.pi*modes/d
    beta=np.sqrt(k1*k1-alpha*alpha+0j)
    t,w=_leggauss_cached(p_quad)
    _,_,co=_poly_basis_data(nodes.p_order);B=basis_at(t,co)
    summ=np.zeros(len(modes),complex)
    for j,curve in enumerate(nodes.panels['curves']):
        y=curve.eval(t);nq=curve.normal(t)
        sl=slice(j*nodes.p_order,(j+1)*nodes.p_order)
        muv=B@mu[sl];rhov=B@rho[sl]
        ph=np.exp(-1j*(alpha[:,None]*y[None,:,0]+beta[:,None]*y[None,:,1]))
        s=(rhov[None,:]-1j*(alpha[:,None]*nq[None,:,0]+beta[:,None]*nq[None,:,1])*muv[None,:])
        summ+=(curve.length/2)*np.sum(ph*s*w[None,:],axis=1)
    amp=1j/(2*d*beta)*summ
    eff=abs(amp)**2*np.real(beta)/sp.beta0
    return modes,amp,eff
