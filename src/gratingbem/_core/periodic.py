"""Periodic KM1 transmission layer operators and Rayleigh projections.

The physical Green function uses exp(-iωt); Helmholtz layer potentials
follow twice-normalized KM1 conventions. Numerical validation remains vital.
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from .ewald import EwaldHessian
from .quadrature import _poly_basis_data, basis_at, log_product_weights, _dist_periodic

def kernel_components(k1,k2,air,metal,x,n,Y,N):
    """Arrays of the six blocks S1,S2,K1,K2,K'1,K'2,Tdiff.

    Twice-normalized layer kernels.  Outward normal points from metal to air.
    """
    dx=x[0]-Y[:,0];dz=x[1]-Y[:,1]
    g1,g1x,g1z,h1xx,h1xz,h1zz=air.evaluate(dx,dz)
    g2,g2x,g2z,h2xx,h2xz,h2zz=metal.evaluate(dx,dz)
    K1=-2*(N[:,0]*g1x+N[:,1]*g1z)
    K2=-2*(N[:,0]*g2x+N[:,1]*g2z)
    Ka1=2*(n[0]*g1x+n[1]*g1z)
    Ka2=2*(n[0]*g2x+n[1]*g2z)
    Td=-2*(n[0]*N[:,0]*(h2xx-h1xx) +
           (n[0]*N[:,1]+n[1]*N[:,0])*(h2xz-h1xz) +
           n[1]*N[:,1]*(h2zz-h1zz))
    return np.column_stack((2*g1,2*g2,K1,K2,Ka1,Ka2,Td))


def panel_quadrature(x,panel_a,panel_b,t_,nq=20,near_div=1):
    """Gauss rules, split at target nearest parameter on straight panel."""
    vec=panel_b-panel_a;L=np.linalg.norm(vec)
    tc=2*np.dot(x-(panel_a+panel_b)/2,vec)/(L*L)
    cut=float(np.clip(tc,-1,1))
    bounds=[-1.,cut,1.] if abs(cut)<.999999 else [-1.,1.]
    t,w=leggauss(nq)
    u=[];ww=[]
    for v1,v2 in zip(bounds[:-1],bounds[1:]):
        if v2-v1<1e-15:continue
        steps=np.linspace(v1,v2,near_div+1)
        for a,b in zip(steps[:-1],steps[1:]):
            u.append(.5*((b-a)*t+a+b));ww.append(.5*(b-a)*w)
    tq=np.concatenate(u);wq=np.concatenate(ww)*(L/2)
    Y=(panel_a+panel_b)/2+tq[:,None]*vec[None,:]/2
    return tq,wq,Y


def build_matrices(nodes,eps, mode_air=155,mode_metal=25,oversample=16,eta=5.,max_rows=None,verbose=False):
    sp=nodes.spec;k1=sp.k0;k2=k1*np.sqrt(eps+0j)
    air=EwaldHessian(sp.period_um,k1,sp.alpha0,eta=eta,modes=mode_air)
    metal=EwaldHessian(sp.period_um,k2,sp.alpha0,eta=eta,modes=mode_metal)
    nd=len(nodes.coords);p=nodes.p_order
    tgl,wgl,coeff=_poly_basis_data(p)
    logw=log_product_weights(p)
    nb=len(nodes.panels['ds'])
    keys=['S1','S2','K1','K2','Ka1','Ka2','Td']
    mats={key:np.zeros((nd,nd),complex) for key in keys}
    # Tdiff=(T2-T1) has exactly only a logarithmic self singularity.
    cT=-(k2*k2-k1*k1)/(2*np.pi)
    for i in range(nd if max_rows is None else max_rows):
        x=nodes.coords[i];n=nodes.normals[i]
        Y=nodes.coords.copy()
        Y[i]=x-np.array([.0023,.0029])  # overwritten self panel
        val=kernel_components(k1,k2,air,metal,x,n,Y,nodes.normals)
        for c,key in enumerate(keys):
            mats[key][i,:]=val[:,c]*nodes.weights
        for j in range(nb):
            pa=nodes.panels['p'][j];pb=nodes.panels['q'][j]
            L=nodes.panels['ds'][j]
            same=(j==nodes.panel_idx[i])
            d=_dist_periodic(x,pa,pb,sp.period_um)
            if not (same or d <= 1.7*L):continue
            subdivisions=1 if same else int(np.clip(np.ceil(L/max(d,.003*L)/1.6),1,60))
            tq,wq,Yq=panel_quadrature(x,pa,pb,tgl,oversample,subdivisions)
            nq=np.tile(nodes.panels['n'][j],(len(tq),1))
            qval=kernel_components(k1,k2,air,metal,x,n,Yq,nq)
            if same:
                r=np.linalg.norm(x-Yq,axis=1)
                qval[:,0:2]+=(1/np.pi)*np.log(r)[:,None]
                qval[:,6]-=cT*np.log(r)
            B=basis_at(tq,coeff)
            dense=(wq[:,None]*qval).T@B
            if same:
                analytic=(L/2)*(np.log(L/2)*wgl+logw[nodes.local_idx[i]])
                dense[0,:]+=(-1/np.pi)*analytic
                dense[1,:]+=(-1/np.pi)*analytic
                dense[6,:]+=cT*analytic
            sl=slice(j*p,(j+1)*p)
            for c,key in enumerate(keys):mats[key][i,sl]=dense[c]
        if verbose and (i+1)%max(1,nd//10)==0: print('assembled',i+1,'/',nd,flush=True)
    return mats


def rayleigh_mu_rho(nodes,mu,rho,p_quad=24):
    sp=nodes.spec
    d=sp.period_um;k1=sp.k0
    mmin=int(np.ceil((-k1-sp.alpha0)*d/(2*np.pi)+1e-9))
    mmax=int(np.floor((k1-sp.alpha0)*d/(2*np.pi)-1e-9))
    modes=np.arange(mmin,mmax+1)
    alpha=sp.alpha0+2*np.pi*modes/d
    beta=np.sqrt(k1*k1-alpha*alpha+0j)
    t,w=leggauss(p_quad)
    _,_,poly=_poly_basis_data(nodes.p_order)
    B=basis_at(t,poly)
    summ=np.zeros(len(modes),complex)
    for j,(pa,pb,L,n) in enumerate(zip(nodes.panels['p'],nodes.panels['q'],nodes.panels['ds'],nodes.panels['n'])):
        yy=(pa+pb)/2+t[:,None]*(pb-pa)/2
        sl=slice(j*nodes.p_order,(j+1)*nodes.p_order)
        muj=B@mu[sl];rhoj=B@rho[sl]
        ph=np.exp(-1j*(alpha[:,None]*yy[None,:,0]+beta[:,None]*yy[None,:,1]))
        factor=(rhoj[None,:]-1j*(alpha[:,None]*n[0]+beta[:,None]*n[1])*muj[None,:])
        summ+=(L/2)*np.sum(ph*factor*w[None,:],axis=1)
    ampl=1j/(2*d*beta)*summ
    efficiencies=abs(ampl)**2*np.real(beta)/sp.beta0
    return modes,ampl,efficiencies
