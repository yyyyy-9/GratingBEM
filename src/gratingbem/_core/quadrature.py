"""Gauss–Nyström interpolation, analytic self-panel log product weights."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from math import comb
import numpy as np
from numpy.polynomial.legendre import leggauss


@lru_cache(maxsize=32)
def gauss_rule(order):
    """Shared, immutable Gauss rule; quadrature order is unchanged."""
    t,w=leggauss(order)
    t.setflags(write=False);w.setflags(write=False)
    return t,w

@lru_cache(None)
def _poly_basis_data(p):
    if not 2 <= p <= 12:
        raise ValueError('Gauss order p must be from 2 to 12')
    t,w=gauss_rule(p)
    coeff=np.linalg.inv(np.vander(t,N=p,increasing=True))
    # Coefficients of the Lagrange basis in monomials, shape (p,p).
    return t,w,coeff


def basis_at(t, coeff):
    t=np.asarray(t)
    V=np.vander(t,N=coeff.shape[0],increasing=True)
    return V@coeff


def _log_moments(x,p):
    """Exact moments int_-1^1 t^k log|t-x| dt, k=0,...,p-1."""
    x=np.longdouble(x)
    a=-np.longdouble(1)-x
    b=np.longdouble(1)-x
    moments=[]
    for k in range(p):
        acc=np.longdouble(0)
        for j in range(k+1):
            power=j+1
            f=lambda u: u**power*(np.log(np.abs(u))/power-1/(np.longdouble(power)**2))
            acc+=np.longdouble(comb(k,j))*x**(k-j)*(f(b)-f(a))
        moments.append(float(acc))
    return np.array(moments)


@lru_cache(None)
def log_product_weights(p):
    t,w,c=_poly_basis_data(p)
    return np.stack([_log_moments(x,p)@c for x in t])


@dataclass
class Nodes:
    spec:object
    panels:dict
    p_order:int
    coords:np.ndarray
    normals:np.ndarray
    weights:np.ndarray
    panel_idx:np.ndarray
    local_idx:np.ndarray


def _dist_periodic(x,a,b,d):
    diff=b-a
    ll=diff@diff
    smallest=np.inf
    for l in (-1,0,1):
        shift=np.array([l*d,0.])
        proj=np.clip((x-a-shift)@diff/ll,0,1)
        distance=np.linalg.norm(x-(a+shift+proj*diff))
        smallest=min(smallest,distance)
    return smallest


def _periodic_chord_geometry(x,starts,ends,sagitta,period,*,min_squared_length=1e-28):
    """Nearest chords over the same three periodic images, in one array pass."""
    vec=ends-starts
    den=np.maximum(np.einsum('ij,ij->i',vec,vec),min_squared_length)
    best_dist=np.full(len(starts),np.inf)
    best_t=np.zeros(len(starts))
    for shift in (-period,0.,period):
        origin=starts+np.array([shift,0.])
        pr=np.einsum('ij,ij->i',x[None,:]-origin,vec)/den
        pr=np.clip(pr,0.,1.)
        dd=np.linalg.norm(x[None,:]-origin-pr[:,None]*vec,axis=1)
        mask=dd<best_dist
        best_dist[mask]=dd[mask];best_t[mask]=2*pr[mask]-1
    return np.maximum(0.,best_dist-sagitta),best_t
