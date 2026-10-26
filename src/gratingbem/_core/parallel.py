"""Process-parallel exact-row Nyström operator for v13's analytic curves.

No approximation is changed: each row repeats the v13 layer-potential and
logarithmic product-integration formulas, including *all* near panels.
The only numerical change is evaluating independent collocation rows in
separate worker processes. Requires a Python main guard on spawn platforms.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import math
import os
import numpy as np

from gratingbem._core.quadrature import _poly_basis_data, basis_at, log_product_weights
from gratingbem._core.ewald import EwaldHessian
from gratingbem._core.periodic import kernel_components
from gratingbem._core.curved import curved_panel_quadrature

_KEYS=('S1','S2','K1','K2','Ka1','Ka2','Td')
_CTX=None


def _periodic_chord_geometry(x, starts, ends, sagitta, period):
    """Vectorized periodic nearest chord; reproduces v13's 3-image test.

    Distances used *only* for deciding which panels require product quadrature;
    physical kernels still use the full Ewald periodic Green function.
    """
    vec=ends-starts
    den=np.maximum(np.einsum('ij,ij->i',vec,vec),1e-28)
    best_dist=np.full(len(starts),np.inf)
    best_t=np.zeros(len(starts))
    for shift in (-period,0.,period):
        s=starts+np.array([shift,0.])
        pr=np.einsum('ij,ij->i',x[None,:]-s,vec)/den
        pr=np.clip(pr,0.,1.)
        dd=np.linalg.norm(x[None,:]-s-pr[:,None]*vec,axis=1)
        mask=dd<best_dist
        best_dist[mask]=dd[mask]
        best_t[mask]=2*pr[mask]-1
    return np.maximum(0.,best_dist-sagitta), best_t


def _init_worker(nodes, eps, mode_air, mode_metal, oversample, eta, near_cutoff):
    global _CTX
    sp=nodes.spec;k1=sp.k0;k2=k1*np.sqrt(eps+0j)
    curves=nodes.panels['curves']
    _CTX=dict(nodes=nodes, air=EwaldHessian(sp.period_um,k1,sp.alpha0,eta=eta,modes=mode_air),
              metal=EwaldHessian(sp.period_um,k2,sp.alpha0,eta=eta,modes=mode_metal),
              k1=k1,k2=k2,cT=-(k2*k2-k1*k1)/(2*np.pi),
              oversample=oversample,near_cutoff=near_cutoff,
              starts=np.asarray([c.start for c in curves]),
              ends=np.asarray([c.end for c in curves]),
              sagitta=np.asarray([c.sagitta for c in curves]),
              lengths=np.asarray([c.length for c in curves]),
              logw=log_product_weights(nodes.p_order),
              gl=_poly_basis_data(nodes.p_order)[0],
              wgl=_poly_basis_data(nodes.p_order)[1],
              coeff=_poly_basis_data(nodes.p_order)[2])


def _assemble_rows(bounds):
    ctx=_CTX
    nodes=ctx['nodes'];p=nodes.p_order
    curves=nodes.panels['curves'];N=len(nodes.coords)
    result=np.empty((len(_KEYS),bounds[1]-bounds[0],N),complex)
    for ilocal,i in enumerate(range(*bounds)):
        x=nodes.coords[i]; normal=nodes.normals[i]; ip=int(nodes.panel_idx[i])
        Y=nodes.coords.copy();Y[i]=x+np.array([.00053,.00081])
        raw=kernel_components(ctx['k1'],ctx['k2'],ctx['air'],ctx['metal'],x,normal,Y,nodes.normals)
        result[:,ilocal,:]=(raw*nodes.weights[:,None]).T
        distances,cuts=_periodic_chord_geometry(x,ctx['starts'],ctx['ends'],ctx['sagitta'],nodes.spec.period_um)
        nearby=np.flatnonzero((distances<=ctx['near_cutoff']*ctx['lengths'])|
                               (np.arange(len(curves))==ip))
        for j in nearby:
            j=int(j);curve=curves[j];same=(j==ip);dist=float(distances[j]);cut=float(cuts[j])
            if same:cut=ctx['gl'][nodes.local_idx[i]];sub=1
            else:sub=int(np.clip(math.ceil(math.log2(curve.length/max(dist,.0005*curve.length)))+1,1,12))
            t,wq,yq,nq=curved_panel_quadrature(curve,cut,ctx['oversample'],sub)
            vals=kernel_components(ctx['k1'],ctx['k2'],ctx['air'],ctx['metal'],x,normal,yq,nq)
            if same:
                r=np.linalg.norm(x-yq,axis=1)
                vals[:,0:2]+=(1/np.pi)*np.log(r)[:,None]
                vals[:,6]-=ctx['cT']*np.log(r)
            B=basis_at(t,ctx['coeff'])
            dense=(wq[:,None]*vals).T@B
            if same:
                lg=(curve.length/2)*(np.log(curve.length/2)*ctx['wgl']+ctx['logw'][nodes.local_idx[i]])
                dense[0,:]+=-lg/np.pi
                dense[1,:]+=-lg/np.pi
                dense[6,:]+=ctx['cT']*lg
            result[:,ilocal,j*p:(j+1)*p]=dense
    return bounds[0],result


def build_matrices_parallel(nodes,eps,mode_air=155,mode_metal=25,oversample=16,
                            eta=5.,near_cutoff=1.7,workers=4,chunk_rows=16):
    """Full v13 operator matrix, with independent row blocks in parallel.

    workers=1 uses the same optimized local worker without processes.
    For small matrices, process startup can outweigh parallel speedup.
    """
    if not 1<=int(workers)<=32 or int(workers)!=workers:
        raise ValueError('workers must be an integer between 1 and 32')
    if int(chunk_rows)!=chunk_rows or chunk_rows<1:
        raise ValueError('chunk_rows must be a positive integer')
    N=len(nodes.coords);bounds=[(i,min(N,i+int(chunk_rows))) for i in range(0,N,int(chunk_rows))]
    mats=np.empty((len(_KEYS),N,N),complex)
    params=(nodes,eps,mode_air,mode_metal,oversample,eta,near_cutoff)
    if workers==1:
        _init_worker(*params)
        for row,block in map(_assemble_rows,bounds):mats[:,row:row+block.shape[1],:]=block
    else:
        # On POSIX fork avoids repeated module import; spawn is used on Windows.
        ctx=get_context('fork' if os.name=='posix' else 'spawn')
        with ProcessPoolExecutor(max_workers=int(workers),mp_context=ctx,
                                 initializer=_init_worker,initargs=params) as executor:
            for row,block in executor.map(_assemble_rows,bounds):
                mats[:,row:row+block.shape[1],:]=block
    return dict(zip(_KEYS,mats))
