"""Outgoing periodic Green value+first derivatives, without unused Hessians.

Matches :class:`gratingbem._core.ewald.EwaldHessian` conventions. This is a
performance implementation, not a different electromagnetic approximation.
"""
from __future__ import annotations
import numpy as np
from scipy.special import exp1, erfc, hankel1
from .._core.ewald import EwaldHessian


class EwaldValueGradient(EwaldHessian):
    def evaluate_vg(self, dx, dz):
        x=np.asarray(dx,float).reshape(-1)
        z=np.asarray(dz,float).reshape(-1)
        if x.shape!=z.shape:raise ValueError('dx,dz size mismatch')
        if np.any(x*x+z*z==0):raise ValueError('coincident; use product quadrature')
        n=len(x)
        g=np.zeros(n,complex);gx=g.copy();gz=g.copy()
        if self.method=='hankel':
            for l in range(-self.images,self.images+1):
                X=x-l*self.period;r=np.hypot(X,z)
                cutoff=45/self.k.imag if self.k.imag>15 else np.inf
                take=r<cutoff
                if not take.any():continue
                RR=r[take]
                h0=hankel1(0,self.k*RR)
                h1=hankel1(1,self.k*RR)
                first=-.25j*self.k*h1/RR
                ph=np.exp(1j*self.alpha0*l*self.period)
                g[take]+=ph*(.25j*h0)
                gx[take]+=ph*first*X[take]
                gz[take]+=ph*first*z[take]
            return g,gx,gz
        if self.method!='ewald':raise ValueError('bad method')
        absz=np.abs(z);sgn=np.sign(z);d=self.period;eta=self.eta
        for j in range(0,len(self.a),self.batchsize):
            a=self.a[j:j+self.batchsize,None]
            b=self.b[j:j+self.batchsize,None]
            q=b/(2*eta)
            A=np.exp(-b*absz)*erfc(q-eta*absz)
            B=np.exp(b*absz)*erfc(q+eta*absz)
            ph=np.exp(1j*a*x)
            p=ph*(A+B)/(4*d*b)
            dp=ph*(B-A)/(4*d)
            g+=p.sum(axis=0)
            gx+=(1j*a*p).sum(axis=0)
            gz+=sgn*dp.sum(axis=0)
        for l in range(-self.images,self.images+1):
            X=x-l*d
            tv=eta**2*(X*X+z*z)
            take=np.flatnonzero(tv<65)
            if not len(take):continue
            t=tv[take];et=np.exp(-t);e1=exp1(t)
            E=e1
            total=self.coeffs[0]*e1
            D1=self.coeffs[0]*et/t
            for order in range(1,self.nterms):
                en=(et-t*E)/order
                total+=self.coeffs[order]*en
                D1+=self.coeffs[order]*E
                E=en
            fac=np.exp(1j*self.alpha0*l*d)
            g[take]+=fac*total/(4*np.pi)
            first=fac*(-eta**2/(2*np.pi))*D1
            gx[take]+=first*X[take]
            gz[take]+=first*z[take]
        return g,gx,gz
