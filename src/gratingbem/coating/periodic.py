"""Experimental 2-interface periodic single-layer BIE for coated profiles.

Air / conformal dielectric film / absorbing substrate.  Retains physical
Helmholtz and material transmission conditions, but is a *first-kind* 4-density
formulation, NOT a periodic KM1+RCIP solver. Near-contact films and sharp
corners require additional quadrature/preconditioning checks.
"""
from __future__ import annotations
from dataclasses import dataclass,replace
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve
from .._core.ewald import EwaldHessian
from .._core.quadrature import _poly_basis_data, basis_at, log_product_weights
from .._core.solver import Problem, Numerics
from .._core.profiles import Profile
from .._core.curved import (make_curve_nodes,nearest_periodic_chord,
                                        curved_panel_quadrature)
from .geometry import offset_profile, validate_interface_order

@dataclass(frozen=True)
class CoatedProblem:
    substrate:Problem
    film_epsilon:complex
    thickness_um:float
    upper_profile:Profile|None=None

    def __post_init__(self):
        ep=complex(self.film_epsilon)
        if ep==0 or ep.imag < 0 or not np.isfinite(ep.real) or not np.isfinite(ep.imag):
            raise ValueError('film epsilon must be finite/passive')
        if self.thickness_um<=0:raise ValueError('finite coating thickness required')
        if self.upper_profile and abs(self.upper_profile.period_um-self.substrate.profile.period_um)>1e-9:
            raise ValueError('upper/lower profiles must have equal period')

    @property
    def top_profile(self):
        return self.upper_profile or offset_profile(self.substrate.profile,self.thickness_um)


def _SD(target,source,kernel,oversample:int,near_cutoff=2.5,same=False):
    """Periodic physical single layer and its target-normal derivative K'.
    The +/-0.5 jumps are NOT included here (they enter transmission rows).
    """
    nt=len(target.coords);ns=len(source.coords);p=source.p_order
    S=np.zeros((nt,ns),complex);D=np.zeros((nt,ns),complex)
    coeff=_poly_basis_data(p)[2];logw=log_product_weights(p)
    wGL=_poly_basis_data(p)[1]
    crvs=source.panels['curves'];d=source.spec.period_um
    for i,(x,n) in enumerate(zip(target.coords,target.normals)):
        coords=source.coords.copy()
        if same:coords[i]=x+np.array([.0004731,.0008712])
        g,gx,gz,*_=kernel.evaluate(x[0]-coords[:,0],x[1]-coords[:,1])
        S[i,:]=g*source.weights
        D[i,:]=(n[0]*gx+n[1]*gz)*source.weights
        for j,seg in enumerate(crvs):
            onself=same and j==target.panel_idx[i]
            distance,cut=nearest_periodic_chord(x,seg,d)
            if not (onself or distance <= near_cutoff*seg.length):continue
            if onself:
                cut=_poly_basis_data(p)[0][target.local_idx[i]]
                sub=1
            else:
                sub=int(np.clip(np.ceil(np.log2(seg.length/max(distance,.0001*seg.length)))+2,1,15))
            ts,w,y,n_y=curved_panel_quadrature(seg,cut,oversample,sub)
            g,gx,gz,*_=kernel.evaluate(x[0]-y[:,0],x[1]-y[:,1])
            if onself:
                r=np.linalg.norm(x-y,axis=1)
                g+=(1/(2*np.pi))*np.log(r)
            B=basis_at(ts,coeff)
            sl=slice(j*p,(j+1)*p)
            S[i,sl]=(w*g)@B
            D[i,sl]=(w*(n[0]*gx+n[1]*gz))@B
            if onself:
                l=seg.length/2
                lg=l*(np.log(l)*wGL+logw[target.local_idx[i]])
                S[i,sl]-=lg/(2*np.pi)
    return S,D


def assemble(problem:CoatedProblem,numerics:Numerics,oversample=None):
    """Return layered trace operators; assume real-axis x-monotone profiles."""
    start=time.perf_counter()
    top_profile=problem.top_profile
    validate_interface_order(problem.substrate.profile,top_profile)
    outer_problem=replace(problem.substrate,profile=top_profile)
    top=make_curve_nodes(outer_problem,numerics)
    bottom=make_curve_nodes(problem.substrate,numerics)
    spec=problem.substrate.spec()
    k0=spec.k0
    eps_f=complex(problem.film_epsilon)
    eps_s=complex(problem.substrate.epsilon_value)
    f_kernel=EwaldHessian(spec.period_um,k0*np.sqrt(eps_f),spec.alpha0,
                          modes=numerics.modes_air,eta=numerics.eta)
    a_kernel=EwaldHessian(spec.period_um,k0,spec.alpha0,
                          modes=numerics.modes_air,eta=numerics.eta)
    s_kernel=EwaldHessian(spec.period_um,k0*np.sqrt(eps_s),spec.alpha0,
                          modes=numerics.modes_metal,eta=numerics.eta)
    q=numerics.oversample if oversample is None else oversample
    mats={}
    mats['att']=_SD(top,top,a_kernel,q,same=True)
    mats['ftt']=_SD(top,top,f_kernel,q,same=True)
    mats['ftb']=_SD(top,bottom,f_kernel,q,same=False)
    mats['fbt']=_SD(bottom,top,f_kernel,q,same=False)
    mats['fbb']=_SD(bottom,bottom,f_kernel,q,same=True)
    mats['sbb']=_SD(bottom,bottom,s_kernel,q,same=True)
    return top,bottom,mats, time.perf_counter()-start


def solve_assembled(problem:CoatedProblem,top,bottom,op,pol='TE'):
    n=len(top.coords);m=len(bottom.coords)
    Znm=np.zeros((n,m),complex)
    Zmn=np.zeros((m,n),complex)
    Znn=np.zeros((n,n),complex)
    Zmm=np.zeros((m,m),complex)
    I_n=np.eye(n);I_m=np.eye(m)
    efilm=complex(problem.film_epsilon)
    emetal=complex(problem.substrate.epsilon_value)
    ga=1.; gf=1 if pol=='TE' else 1/efilm;gm=1 if pol=='TE' else 1/emetal
    if pol not in ('TE','TM'):raise ValueError('pol must be TE/TM')
    sa,da=op['att'];sft,dft=op['ftt'];sfb,dfb=op['ftb']
    sbf,dbf=op['fbt'];sbb,dbb=op['fbb'];sm,dm=op['sbb']
    # unknowns (air_top, film_top, film_bottom, substrate_bottom)
    A=np.block([
        [sa,       -sft,              -sfb,            Znm],
        [Zmn,       sbf,               sbb,             -sm],
        [ga*(-I_n/2+da), -gf*(I_n/2+dft), -gf*dfb,     Znm],
        [Zmn,      gf*dbf,     gf*(-I_m/2+dbb), -gm*(I_m/2+dm)]
    ])
    sp=problem.substrate.spec()
    xt=top.coords; nt=top.normals
    inc=np.exp(1j*(sp.alpha0*xt[:,0]-sp.beta0*xt[:,1]))
    deriv=1j*(sp.alpha0*nt[:,0]-sp.beta0*nt[:,1])*inc
    rhs=np.r_[-inc,np.zeros(m,complex),-ga*deriv,np.zeros(m,complex)]
    # Differential rows scale like k0 times field rows for the chosen density basis
    scale=np.r_[np.ones(n+m),np.ones(n+m)/sp.k0]
    A=A*scale[:,None];rhs=rhs*scale
    start=time.perf_counter()
    dens=solve(A,rhs,check_finite=False)
    duration=time.perf_counter()-start
    err=float(np.linalg.norm(A@dens-rhs)/np.linalg.norm(rhs))
    # Independent Poynting-flux check through the metal interface.
    trace_m=sm@dens[2*n+m:]
    normal_m=(I_m/2+dm)@dens[2*n+m:]
    flux_m=-np.imag(np.vdot(trace_m, bottom.weights*(gm*normal_m))) / (sp.period_um*sp.beta0)
    # Film absorption from flux difference through the upper/lower contours.
    film_top=sft@dens[n:2*n]+sfb@dens[2*n:2*n+m]
    film_top_deriv=(I_n/2+dft)@dens[n:2*n]+dfb@dens[2*n:2*n+m]
    film_bottom=sbf@dens[n:2*n]+sbb@dens[2*n:2*n+m]
    film_bottom_deriv=dbf@dens[n:2*n]+(-I_m/2+dbb)@dens[2*n:2*n+m]
    flow_top=-np.imag(np.vdot(film_top,top.weights*(gf*film_top_deriv)))/(sp.period_um*sp.beta0)
    flow_bot=-np.imag(np.vdot(film_bottom,bottom.weights*(gf*film_bottom_deriv)))/(sp.period_um*sp.beta0)
    film_abs=float(flow_top-flow_bot)
    return {'densities':dens,'residual':err,'solve_seconds':duration,
            'pol':pol,'matrix_shape':list(A.shape), 'metal_absorption_flux':float(flux_m),
            'film_absorption_flux':film_abs, 'flux_through_film_top':float(flow_top),
            'flux_through_film_bottom':float(flow_bot)}


def rayleigh_top(top,density,p_quad=32):
    """Expand the air-side scattered single layer into outgoing modes."""
    sp=top.spec;k=sp.k0;d=sp.period_um
    low=int(np.ceil((-k-sp.alpha0)*d/(2*np.pi)+1e-9))
    high=int(np.floor((k-sp.alpha0)*d/(2*np.pi)-1e-9))
    mm=np.arange(low,high+1)
    alpha=sp.alpha0+2*np.pi*mm/d
    beta=np.sqrt(k*k-alpha*alpha+0j)
    t,w=leggauss(p_quad)
    p=top.p_order;co=_poly_basis_data(p)[2];B=basis_at(t,co)
    amp=np.zeros(len(mm),complex)
    for j,c in enumerate(top.panels['curves']):
        pts=c.eval(t);dens=B@density[j*p:(j+1)*p]
        phi=np.exp(-1j*(alpha[:,None]*pts[None,:,0]+beta[:,None]*pts[None,:,1]))
        amp+=(c.length/2)*(phi@(w*dens))
    amp=1j*amp/(2*d*beta)
    eta=abs(amp)**2*np.real(beta)/sp.beta0
    return mm,amp,eta


def calculate_coated(problem:CoatedProblem,numerics:Numerics,pols=('TE','TM'),oversample=None,
                     assembly_options=None):
    t0=time.perf_counter()
    if assembly_options is None:
        top,bottom,mats,tasm=assemble(problem,numerics,oversample)
        stats=None
    else:
        from .assembly import assemble_coated
        top,bottom,mats,tasm,stats=assemble_coated(problem,numerics,assembly_options,oversample)
    result={'model':'2-interface-periodic-single-layer-BIE','status':'research_unvalidated_for_general_curvature',
            'film_thickness_um':float(problem.thickness_um),
            'film_epsilon':str(problem.film_epsilon),
            'substrate_epsilon':str(problem.substrate.epsilon_value),
            'n_top':len(top.coords),'n_bottom':len(bottom.coords),
            'assembly_seconds':tasm,'polarizations':{}}
    for pol in pols:
        s=solve_assembled(problem,top,bottom,mats,pol)
        n=len(top.coords)
        m,amp,eta=rayleigh_top(top,s['densities'][:n])
        target=problem.substrate.target_order
        wh=np.flatnonzero(m==target)
        result['polarizations'][pol]={'target_efficiency':None if not len(wh) else float(eta[wh[0]]),
                 'reflection_sum':float(np.sum(eta)),
                 'matrix_residual':s['residual'],
                 'metal_absorption_flux':s['metal_absorption_flux'],
                 'film_absorption_flux':s['film_absorption_flux'],
                 'energy_balance_error':float(np.sum(eta)+s['metal_absorption_flux']+s['film_absorption_flux']-1),
                 'solve_seconds':s['solve_seconds'],
                 'modes':m.tolist(),'efficiencies':eta.tolist(),
                 'amplitude_real':amp.real.tolist(),'amplitude_imag':amp.imag.tolist()}
    result['total_seconds']=time.perf_counter()-t0
    if stats is not None:
        result['assembly_checkpoint_stats']=stats
    # Energy conservation alone is necessary but never proof of mesh convergence.
    statuses={}
    for pol,v in result['polarizations'].items():
        if v['reflection_sum']>1.001 or v['metal_absorption_flux']<-.001 or v['film_absorption_flux']<-.001:
            statuses[pol]='failed_passivity'
        elif abs(v['energy_balance_error'])>.005:
            statuses[pol]='failed_energy_balance'
        else:
            statuses[pol]='passed_energy_only_mesh_unverified'
        v['status']=statuses[pol]
    result['status']= ('failed_physical_checks' if any(x.startswith('failed') for x in statuses.values())
                       else 'passed_energy_only_mesh_unverified')
    return result
