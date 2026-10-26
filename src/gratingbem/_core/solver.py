"""v13 public calculate / scan API over the v11 periodic two-density KM1 kernel.

Research solver, not a certified design tool. Units: micrometres, degrees.
Material is one homogeneous halfspace beneath the periodic profile: NO coating.
RCIP remains experimental and is deliberately not implied by calculate().
"""
from __future__ import annotations
from .validation import integer_order
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence
import csv
import json
import math
import itertools
import time
import warnings
import numpy as np
from scipy.linalg import solve
from numpy.polynomial.legendre import leggauss

from gratingbem._core.quadrature import Nodes, _poly_basis_data
from gratingbem._core.materials import eps_al_rakic_ld
from gratingbem._core.periodic import build_matrices, rayleigh_mu_rho
from .profiles import Profile, from_dict as profile_from_dict
from .curved import make_curve_nodes, build_matrices_curved, rayleigh_curved


@dataclass(frozen=True)
class Problem:
    profile: Profile
    wavelength_um: float
    target_order: int = 0
    theta_deg: float | None = None
    littrow_order: int | None = None
    epsilon: complex | None = None  # None: Rakić-Lorentz Drude Al

    def __post_init__(self):
        object.__setattr__(self,'target_order',integer_order(self.target_order,name='target_order'))
        if self.littrow_order is not None:
            object.__setattr__(self,'littrow_order',integer_order(self.littrow_order,name='Littrow order',nonzero=True))
        if self.wavelength_um<=0 or not np.isfinite(self.wavelength_um):
            raise ValueError('wavelength_um must be positive')
        if (self.theta_deg is None)==(self.littrow_order is None):
            raise ValueError('specify exactly one of theta_deg or littrow_order')
        if self.theta_deg is not None and not (-89.99<float(self.theta_deg)<89.99):
            raise ValueError('theta_deg must lie in (-89.99,89.99)')
        if self.littrow_order is not None:
            x=abs(self.littrow_order)*self.wavelength_um/(2*self.profile.period_um)
            if x >= 1:raise ValueError('Littrow order is not propagating')
        if self.epsilon is not None:
            e=complex(self.epsilon)
            if not np.isfinite(e.real) or not np.isfinite(e.imag) or e.imag < 0:
                raise ValueError('passive epsilon with Im(epsilon)>=0 required')
            if e==0:raise ValueError('epsilon=0 unsupported')

    @property
    def incidence_deg(self):
        if self.theta_deg is not None:return float(self.theta_deg)
        return float(np.rad2deg(np.arcsin(-self.littrow_order*self.wavelength_um/
                                     (2*self.profile.period_um))))

    @property
    def epsilon_value(self):
        return complex(eps_al_rakic_ld(self.wavelength_um) if self.epsilon is None else self.epsilon)

    def spec(self):
        return _Spec(self)

    def with_parameter(self,parameter:str,value):
        if parameter=='profile':
            if not isinstance(value,Profile):raise ValueError('profile scan values must be Profile objects')
            return replace(self,profile=value)
        if parameter.startswith('profile.'):
            return replace(self,profile=self.profile.with_parameter(parameter.split('.',1)[1],value))
        if parameter in ('wavelength_um','target_order','theta_deg','littrow_order','epsilon'):
            if parameter=='theta_deg':return replace(self,theta_deg=float(value),littrow_order=None)
            if parameter=='littrow_order':return replace(self,littrow_order=value,theta_deg=None)
            if parameter=='epsilon':
                if isinstance(value,dict):value=complex(value['epsilon_real'],value.get('epsilon_imag',0))
                elif isinstance(value,(list,tuple)) and len(value)==2:value=complex(*value)
            return replace(self,**{parameter:value})
        raise ValueError(f'unknown parameter {parameter!r}')

    def as_dict(self):
        return dict(profile=self.profile.as_dict(),wavelength_um=self.wavelength_um,
                    theta_deg=self.theta_deg,littrow_order=self.littrow_order,
                    incidence_deg=self.incidence_deg,target_order=self.target_order,
                    material=('Al_Rakic_LD' if self.epsilon is None else
                              dict(epsilon_real=complex(self.epsilon).real,
                                   epsilon_imag=complex(self.epsilon).imag)),
                    epsilon_real=self.epsilon_value.real,epsilon_imag=self.epsilon_value.imag)

    @classmethod
    def from_dict(cls,config):
        data=dict(config)
        inc=data.pop('incidence',{})
        mat=data.pop('material',{'model':'Al_Rakic_LD'})
        if isinstance(mat,str):mat={'model':mat}
        if 'epsilon_real' in mat:
            eps=complex(mat['epsilon_real'],mat.get('epsilon_imag',0))
        elif mat.get('model') in ('Al_Rakic_LD','aluminum_rakic_ld'):
            eps=None
        else:raise ValueError('material must specify model=Al_Rakic_LD or epsilon_real/epsilon_imag')
        shape=data.pop('profile')
        if 'layers' in data or 'coating' in data:
            raise NotImplementedError('coating / multilayer not implemented in v12')
        return cls(profile=profile_from_dict(shape),
                   wavelength_um=float(data.pop('wavelength_um')),
                   target_order=data.pop('target_order',0),
                   theta_deg=inc.get('theta_deg',data.pop('theta_deg',None)),
                   littrow_order=inc.get('littrow_order',data.pop('littrow_order',None)),
                   epsilon=eps)


@dataclass(frozen=True)
class _Spec:
    problem: Problem

    @property
    def period_um(self):return self.problem.profile.period_um
    @property
    def wavelength_um(self):return self.problem.wavelength_um
    @property
    def k0(self):return 2*np.pi/self.wavelength_um
    @property
    def alpha0(self):return self.k0*np.sin(np.deg2rad(self.problem.incidence_deg))
    @property
    def beta0(self):return self.k0*np.cos(np.deg2rad(self.problem.incidence_deg))
    @property
    def order(self):return self.problem.target_order


@dataclass(frozen=True)
class Numerics:
    p_order: int = 4
    max_panel_wavelengths: float = .80
    min_panels_per_segment: int = 2
    grading: float = 1.15
    modes_air: int = 155
    modes_metal: int = 25
    oversample: int = 16
    eta: float = 5.
    max_nodes: int = 2000
    panels_per_edge: tuple[int,...] | None = None
    rcip: bool = False

    def __post_init__(self):
        if self.rcip:raise NotImplementedError('general-profile RCIP is not validated; v12 KM1 runs without RCIP')
        if not 2<=self.p_order<=12:raise ValueError('p_order must be in 2..12')
        if self.max_panel_wavelengths<=0 or self.min_panels_per_segment<1 or self.grading<1:
            raise ValueError('invalid mesh parameters')
        if self.modes_air<1 or self.modes_metal<1 or self.oversample<8 or self.max_nodes<4:
            raise ValueError('invalid kernel/size parameters')
        if self.panels_per_edge is not None and (not self.panels_per_edge or
                any(int(x)!=x or x<1 for x in self.panels_per_edge)):
            raise ValueError('panels_per_edge must be positive integers')

    def as_dict(self):return dict(self.__dict__)

    @classmethod
    def from_dict(cls,d):return cls(**d)


def make_profile_nodes(problem: Problem,numerics: Numerics):
    if problem.profile.has_arcs:
        return make_curve_nodes(problem,numerics)
    """Length-limited graded Gauss panels on *any* x-monotone polyline.

    Each original edge is independently graded toward its endpoints.  Thus
    arbitrary vertices (including vertical rectangular walls) are corners.
    """
    sp=problem.spec();v=np.asarray(problem.profile.vertices,float)
    limit=numerics.max_panel_wavelengths*problem.wavelength_um
    starts=[];ends=[];facets=[];counts=[]
    if numerics.panels_per_edge is not None and len(numerics.panels_per_edge)!=len(v)-1:
        raise ValueError("panels_per_edge must have one entry per profile edge")
    for face,(a,b) in enumerate(zip(v[:-1],v[1:])):
        length=np.linalg.norm(b-a)
        n=(int(numerics.panels_per_edge[face]) if numerics.panels_per_edge is not None
           else max(numerics.min_panels_per_segment,math.ceil(length/limit)))
        counts.append(n)
        s=np.linspace(0,1,n+1)
        grade=numerics.grading
        t=s**grade/(s**grade+(1-s)**grade)
        for lo,hi in zip(t[:-1],t[1:]):
            starts.append(a+(b-a)*lo);ends.append(a+(b-a)*hi);facets.append(face)
    p=np.asarray(starts);q=np.asarray(ends)
    ds=np.linalg.norm(q-p,axis=1)
    if np.any(ds<=0):raise RuntimeError('degenerate mesh under float precision')
    d=q-p;normals=np.column_stack((-d[:,1]/ds,d[:,0]/ds))
    n_pan=len(ds);p_order=numerics.p_order;nd=n_pan*p_order
    if nd>numerics.max_nodes:
        raise ValueError(f'{nd} unknown nodes exceed max_nodes={numerics.max_nodes}; increase limit explicitly')
    panels=dict(p=p,q=q,mid=(p+q)/2,ds=ds,n=normals,
                facets=np.asarray(facets,int),counts_by_edge=np.asarray(counts,int),
                flat=len(v)==2 and np.isclose(v[0,1],v[1,1]),spec=sp)
    t,w,_=_poly_basis_data(p_order)
    xx=((p[:,None,:]+q[:,None,:])/2+t[None,:,None]*d[:,None,:]/2).reshape(-1,2)
    nn=np.repeat(normals,p_order,axis=0)
    wt=(ds[:,None]*w[None,:]/2).reshape(-1)
    return Nodes(sp,panels,p_order,xx,nn,wt,
                 np.repeat(np.arange(n_pan),p_order),np.tile(np.arange(p_order),n_pan))


@dataclass
class Calculation:
    problem: Problem
    numerics: Numerics
    modes: np.ndarray
    amplitudes: dict[str,np.ndarray]
    efficiencies: dict[str,np.ndarray]
    diagnostics: dict
    timings: dict
    warning_messages: list[str]
    internal: dict | None = None  # optional ephemeral densities/nodes; never serialized

    def summary(self):
        r=dict(shape=self.problem.profile.name,
               period_um=self.problem.profile.period_um,
               wavelength_um=self.problem.wavelength_um,
               theta_deg=self.problem.incidence_deg,
               target_order=self.problem.target_order,
               n_vertices=len(self.problem.profile.vertices),
               n_nodes=self.diagnostics['n_nodes'],
               time_total_s=self.timings['total_s'],
               time_assembly_s=self.timings['assembly_s'],
               time_solve_s=self.timings['solve_s'],
               warnings='; '.join(self.warning_messages))
        for pol in self.efficiencies:
            x=self.diagnostics[pol]
            r[pol+'_target_pct']=None if x['target_efficiency'] is None else 100*x['target_efficiency']
            r[pol+'_total_reflect_pct']=100*x['reflection_sum']
            r[pol+'_matrix_residual']=x['matrix_residual']
        return r

    def to_dict(self):
        data=dict(problem=self.problem.as_dict(),numerics=self.numerics.as_dict(),
                  diagnostic=self.diagnostics,timings=self.timings,warnings=self.warning_messages,
                  orders=[])
        for k,m in enumerate(self.modes):
            row=dict(order=int(m))
            for pol in self.efficiencies:
                a=self.amplitudes[pol][k]
                row[pol+'_amplitude_re']=float(a.real)
                row[pol+'_amplitude_im']=float(a.imag)
                row[pol+'_efficiency']=float(self.efficiencies[pol][k])
            data['orders'].append(row)
        return data

    def save(self,folder):
        dst=Path(folder);dst.mkdir(parents=True,exist_ok=True)
        data=self.to_dict()
        (dst/'result.json').write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')
        with (dst/'orders.csv').open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(data['orders'][0]));writer.writeheader();writer.writerows(data['orders'])
        self.problem.profile.save_svg(dst/'profile.svg')
        (dst/'profile_segments.json').write_text(json.dumps([s.as_dict() for s in
                self.problem.profile.geometric_segments],indent=2),encoding='utf8')
        with (dst/'profile_vertices.csv').open('w',newline='',encoding='utf8') as f:
            writer=csv.writer(f);writer.writerow(('x_um','z_um'))
            writer.writerows(self.problem.profile.vertices)
        return dst


def calculate(problem:Problem,numerics:Numerics|None=None,polarizations=('TE','TM'),
              *,_capture_internal=False,_assembly_function=None)->Calculation:
    """Compute all propagating orders of one single-interface periodic grating.

    Returns numeric results AND diagnostics. Passivity check is only R<=1;
    metal absorption is *not independently* integrated in this KM1 version.
    """
    num=numerics or Numerics()
    pols=tuple(str(p).upper() for p in polarizations)
    if len(set(pols))!=len(pols) or any(p not in ('TE','TM') for p in pols) or not pols:
        raise ValueError('polarizations must be a nonempty subset of TE,TM')
    t0=time.perf_counter()
    nodes=make_profile_nodes(problem,num)
    t_mesh=time.perf_counter()
    eps=problem.epsilon_value
    assembly_function=(_assembly_function if _assembly_function is not None else
                       (build_matrices_curved if problem.profile.has_arcs else build_matrices))
    ops=assembly_function(nodes,eps,mode_air=num.modes_air,mode_metal=num.modes_metal,
                          oversample=num.oversample,eta=num.eta)
    t_assembly=time.perf_counter()
    k1=nodes.spec.k0;k2=k1*np.sqrt(eps+0j);n=len(nodes.coords)
    ui=np.exp(1j*(nodes.spec.alpha0*nodes.coords[:,0]-nodes.spec.beta0*nodes.coords[:,1]))
    du=1j*(nodes.spec.alpha0*nodes.normals[:,0]-nodes.spec.beta0*nodes.normals[:,1])*ui
    amplitudes={};efficiencies={};diagnostics={}
    stored_densities={} if _capture_internal else None
    for pol in pols:
        chi=1+0j if pol=='TE' else eps
        z=chi*k2;c=z/abs(z)
        a1=1/(1+chi);a2=chi/(1+chi);a3=1/(c+chi);a4=chi/(c+chi)
        I=np.eye(n)
        A=np.block([[I-a2*ops['K2']+a1*ops['K1'], -a1*(c*ops['S2']-ops['S1'])],
                    [a4*ops['Td'], I+c*a3*ops['Ka2']-a4*ops['Ka1']]])
        rhs=np.r_[-2*a1*ui,2*a4*du]
        scaling=np.r_[np.ones(n),np.ones(n)/k1]
        A*=scaling[:,None];rhs*=scaling
        den=solve(A,rhs,check_finite=False)
        if _capture_internal:
            stored_densities[pol]=(den[:n].copy(),den[n:].copy())
        residual=float(np.linalg.norm(A@den-rhs)/np.linalg.norm(rhs))
        orders,amp,eff=(rayleigh_curved(nodes,den[:n],den[n:]) if problem.profile.has_arcs
                        else rayleigh_mu_rho(nodes,den[:n],den[n:]))
        amplitudes[pol]=amp;efficiencies[pol]=eff
        target=eff[orders==problem.target_order]
        diagnostics[pol]=dict(target_efficiency=float(target[0]) if len(target) else None,
                              reflection_sum=float(eff.sum()),
                              matrix_residual=residual,
                              passes_passivity=bool(eff.sum()<=1+5e-4))
    t_solve=time.perf_counter()
    diagnostics.update(n_nodes=int(n),n_panels=int(len(nodes.panels['ds'])),
                       panels_per_edge=list(map(int,nodes.panels['counts_by_edge'])),
                       n_propagating_orders=len(orders))
    warn=[]
    if not np.any(orders==problem.target_order):warn.append('Target order is evanescent; efficiency=None')
    for pol in pols:
        if not diagnostics[pol]['passes_passivity']:
            warn.append(f'{pol}: R_total>1 violates passive-medium check; refine/discard result')
    if problem.epsilon is None:
        warn.append('Al Rakić LD bulk optical constants; no Al2O3/coating/roughness')
    warn.append('Research KM1 solver without general-profile RCIP or rigorous error bounds')
    if problem.profile.has_arcs:
        warn.append('Analytic circular arc geometry; curved-panel quadrature, not polygon approximation')
    return Calculation(problem,num,orders,amplitudes,efficiencies,diagnostics,
                       dict(mesh_s=t_mesh-t0,assembly_s=t_assembly-t_mesh,
                            solve_s=t_solve-t_assembly,total_s=t_solve-t0),warn,
                       (dict(nodes=nodes,densities=stored_densities) if _capture_internal else None))


@dataclass
class ScanResult:
    parameter: str
    values: list
    results: list[Calculation|None]
    errors: list[str|None]
    timings: dict

    def rows(self):
        output=[]
        for i,(val,result,error) in enumerate(zip(self.values,self.results,self.errors)):
            row=dict(index=i,scan_parameter=self.parameter,scan_value=str(val),error=error or '')
            if isinstance(val,dict):row.update({'scan_'+k:str(v) for k,v in val.items()})
            if result is not None:row.update(result.summary())
            output.append(row)
        return output

    def save(self,folder):
        dst=Path(folder);dst.mkdir(parents=True,exist_ok=True)
        rows=self.rows()
        names=list(dict.fromkeys(k for row in rows for k in row))
        with (dst/'scan.csv').open('w',newline='',encoding='utf8') as f:
            wr=csv.DictWriter(f,fieldnames=names);wr.writeheader();wr.writerows(rows)
        for i,result in enumerate(self.results):
            if result is not None:result.save(dst/f'case_{i:03d}')
        (dst/'scan_metadata.json').write_text(json.dumps(dict(parameter=self.parameter,
            values=[str(v) for v in self.values],errors=self.errors,timings=self.timings),
            ensure_ascii=False,indent=2),encoding='utf8')
        return dst


def scan(problem:Problem,parameter:str,values:Sequence,
         numerics:Numerics|None=None,polarizations=('TE','TM'),
         update:Callable[[Problem,object],Problem]|None=None,
         on_error='raise',progress:Callable[[int,int,object,Calculation|None,str|None],None]|None=None)->ScanResult:
    """Parameter sweep; scan existing recipe params, materials, wavelengths, or profiles.

    For arbitrary custom geometry supply values=[Profile(...), ...], parameter='profile',
    or update=lambda problem,value: dataclasses.replace(problem,profile=...).
    """
    if on_error not in ('raise','record'):raise ValueError('on_error must be raise or record')
    t0=time.perf_counter();vals=list(values);runs=[];errs=[]
    for i,v in enumerate(vals):
        result=None;err=None
        try:
            if parameter.startswith('numerics.') and update is None:
                field_name=parameter.split('.',1)[1]
                if field_name not in Numerics.__dataclass_fields__:
                    raise ValueError(f'Unknown numerical option {field_name}')
                p=problem; ns=replace(numerics or Numerics(),**{field_name:v})
            else:
                p=update(problem,v) if update else problem.with_parameter(parameter,v)
                ns=numerics
            if not isinstance(p,Problem):raise TypeError('update() must return a Problem')
            result=calculate(p,ns,polarizations)
        except Exception as ex:
            if on_error=='raise':raise
            err=f'{type(ex).__name__}: {ex}'
        runs.append(result);errs.append(err)
        if progress:progress(i+1,len(vals),v,result,err)
    return ScanResult(parameter,vals,runs,errs,dict(total_s=time.perf_counter()-t0))


def scan_grid(problem:Problem,axes:dict[str,Sequence],numerics:Numerics|None=None,
              polarizations=('TE','TM'),on_error='raise',progress=None)->ScanResult:
    """Cartesian parameter grid, e.g. {'profile.height_um':[.1,.2], 'theta_deg':[0,10]}."""
    if not axes:raise ValueError('axes may not be empty')
    keys=list(axes)
    for k in keys:
        if k=='profile':raise ValueError('for arbitrary profiles use scan(parameter="profile",...)')
        if k.startswith('numerics.'):
            raise ValueError('scan_grid numerical axes are not supported; use scan()')
    values=[dict(zip(keys,vals)) for vals in itertools.product(*(list(axes[k]) for k in keys))]
    if not values:raise ValueError('empty scan grid')
    def upd(base,parameters):
        for k,value in parameters.items():base=base.with_parameter(k,value)
        return base
    return scan(problem,'grid:'+','.join(keys),values,numerics,polarizations,
                update=upd,on_error=on_error,progress=progress)


def convergence(problem:Problem,max_panel_wavelengths=(1.2,.9,.6),
                numerics:Numerics|None=None,polarizations=('TE','TM'),on_error='raise'):
    """Run a mesh-resolution study; differences are empirical, not rigorous bounds."""
    return scan(problem,'numerics.max_panel_wavelengths',max_panel_wavelengths,
                numerics,polarizations,on_error=on_error)
