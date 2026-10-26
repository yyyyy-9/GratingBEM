"""Ewald quasi-periodic outgoing Green + first and second spatial derivatives.

Based on the verified v9 Ewald split, extended analytically to Gxx,Gxz,Gzz.
This kernel is the *physical* G=i/4 H0.  Multiply by two to obtain
Helsing--Karlsson KM1 layer operators.  It is not a singular Nyström panel
quadrature; zero separation is forbidden.

The diffraction Green function is singular at Wood anomalies. This method
rejects beta_m near zero; a proxy/radiation formulation would be required.
"""
from __future__ import annotations
import numpy as np
from math import factorial
from scipy.special import exp1,erfc,hankel1


class EwaldHessian:
    def __init__(self,period,k,alpha0,eta=5.,modes=155,images=2,nterms=42,batchsize=16,method='auto'):
        self.period=float(period);self.k=complex(k);self.alpha0=float(alpha0)
        self.eta=float(eta);self.images=int(images);self.nterms=int(nterms)
        self.batchsize=int(batchsize)
        if method=='auto':method='hankel' if self.k.imag>10 else 'ewald'
        self.method=method
        ms=np.arange(-modes,modes+1)
        self.a=alpha0+2*np.pi*ms/self.period
        self.b=np.sqrt(self.a**2-self.k**2+0j)
        self.b=np.where((self.b.real<0)|((abs(self.b.real)<1e-12)&(self.b.imag>0)),-self.b,self.b)
        if min(abs(self.b))<1e-8:raise ValueError('Wood anomaly; use another representation')
        self.coeffs=np.array([(self.k**2/(4*self.eta**2))**n/factorial(n) for n in range(nterms)],complex)

    def evaluate(self,dx,dz):
        x=np.asarray(dx,float).reshape(-1);z=np.asarray(dz,float).reshape(-1)
        if x.shape!=z.shape:raise ValueError('dx,dz size mismatch')
        if np.any((x*x+z*z)==0):raise ValueError('coincident, use product quadrature')
        n=len(x)
        g=np.zeros(n,complex);gx=g.copy();gz=g.copy();hxx=g.copy();hxz=g.copy();hzz=g.copy()
        if self.method=='hankel':
            for l in range(-self.images,self.images+1):
                X=x-l*self.period;r=np.hypot(X,z)
                cutoff=45/self.k.imag if self.k.imag>15 else np.inf
                take=r<cutoff
                if not take.any():continue
                RR=r[take];xx=X[take];zz=z[take]
                h0=hankel1(0,self.k*RR);h1=hankel1(1,self.k*RR)
                gr=.25j*h0
                first=-.25j*self.k*h1/RR
                second=-.25j*self.k**2*h0+.5j*self.k*h1/RR
                ph=np.exp(1j*self.alpha0*l*self.period)
                g[take]+=ph*gr
                gx[take]+=ph*first*xx;gz[take]+=ph*first*zz
                hxx[take]+=ph*(first+second*xx**2/RR**2)
                hzz[take]+=ph*(first+second*zz**2/RR**2)
                hxz[take]+=ph*second*xx*zz/RR**2
            return g,gx,gz,hxx,hxz,hzz
        if self.method!='ewald':raise ValueError('bad method')
        absz=np.abs(z);sgn=np.sign(z); d=self.period;eta=self.eta
        for j in range(0,len(self.a),self.batchsize):
            a=self.a[j:j+self.batchsize,None]
            b=self.b[j:j+self.batchsize,None]
            q=b/(2*eta)
            A=np.exp(-b*absz)*erfc(q-eta*absz)
            B=np.exp(b*absz)*erfc(q+eta*absz)
            ph=np.exp(1j*a*x)
            p=ph*(A+B)/(4*d*b)
            dp=ph*(B-A)/(4*d)
            e=np.exp(-q*q-eta*eta*absz*absz)
            g+=p.sum(axis=0)
            gx+=(1j*a*p).sum(axis=0)
            gz+=sgn*dp.sum(axis=0)
            hxx+=(-a*a*p).sum(axis=0)
            hxz+=(sgn*1j*a*dp).sum(axis=0)
            hzz+=(ph*(b*(A+B)-4*eta/np.sqrt(np.pi)*e)/(4*d)).sum(axis=0)
        for l in range(-self.images,self.images+1):
            X=x-l*d
            tv=eta**2*(X*X+z*z)
            take=np.flatnonzero(tv<65)
            if not len(take):continue
            t=tv[take];et=np.exp(-t);e1=exp1(t)
            E=[e1]
            total=self.coeffs[0]*e1
            D1=self.coeffs[0]*et/t
            for order in range(1,self.nterms):
                en=(et-t*E[-1])/order
                E.append(en)
                total+=self.coeffs[order]*en
                D1+=self.coeffs[order]*E[-2]
            D2=self.coeffs[0]*et*(1/t+1/(t*t))
            if self.nterms>1:D2+=self.coeffs[1]*et/t
            for order in range(2,self.nterms):
                D2+=self.coeffs[order]*E[order-2]
            fac=np.exp(1j*self.alpha0*l*d)
            g[take]+=fac*total/(4*np.pi)
            first=fac*(-eta**2/(2*np.pi))*D1
            second=fac*eta**4/np.pi*D2
            gx[take]+=first*X[take];gz[take]+=first*z[take]
            hxx[take]+=first+second*X[take]**2
            hzz[take]+=first+second*z[take]**2
            hxz[take]+=second*X[take]*z[take]
        return g,gx,gz,hxx,hxz,hzz
