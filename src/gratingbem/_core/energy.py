"""Independent total-field Poynting-flux quadrature above a smooth graph grating.

Unlike simply setting absorption = 1 - sum(Rm), this reconstructs E/H field
and normal derivative from the KM1 layer *densities*, then integrates flux on a
vertically translated copy of the periodic interface lying wholly in air.

This is a numerical cross-check (not a mathematical a posteriori bound). The
physical flux is constant through any such air contour, but near singularity
quadrature and finite Nyström resolution may spoil that invariance. For exact
corners with vertical walls there is no single-valued z(x) graph: reject.
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from .ewald import EwaldHessian


def air_contour_absorption(nodes, mu, rho, *, modes_air=155, eta=5.0,
                           lift_wavelengths=.6, max_target_nodes=None,
                           target_order=None):
    """Return normalized absorbed fraction from total-field flux (air contour).

    Normal always points from metal into air; inward flux is positive absorbed
    power. All reflected orders are accounted for implicitly. TE/TM scalar
    fields both have the same *air-side* flux normalization.
    """
    if lift_wavelengths <= 0:
        raise ValueError('lift_wavelengths must be positive')
    panels = nodes.panels
    edges = np.asarray(panels['q'])-np.asarray(panels['p'])
    # Vertical sides are not graphs, so translating those alone is not a valid
    # non-self-intersecting air contour; a general contour constructor is needed.
    if np.any(edges[:,0] <= 1e-12):
        raise NotImplementedError('Poynting air-contour check currently requires a strictly x-monotone profile')
    spec = nodes.spec
    qorder = nodes.p_order if target_order is None else int(target_order)
    if qorder < 2 or qorder > 32:
        raise ValueError('target_order must be in 2..32')
    if max_target_nodes is not None and len(panels['ds'])*qorder > max_target_nodes:
        raise ValueError('too many target nodes for requested Poynting diagnostic')
    delta = lift_wavelengths * spec.wavelength_um
    # The original implementation reused source Nyström nodes and weights on
    # the lifted contour. That underintegrates the rapidly varying total field
    # near the interface. Build an independent target quadrature while keeping
    # the original source density/weights unchanged.
    t, wt = leggauss(qorder)
    target_parts=[]; weight_parts=[]; normal_parts=[]
    curves=panels.get('curves')
    if curves is None:
        for p,q,ds,norm in zip(panels['p'],panels['q'],panels['ds'],panels['n']):
            d=q-p
            target_parts.append(.5*(p+q)+.5*t[:,None]*d)
            weight_parts.append(.5*ds*wt)
            normal_parts.append(np.repeat(norm[None,:],qorder,axis=0))
    else:
        for curve in curves:
            target_parts.append(curve.eval(t))
            weight_parts.append(.5*curve.length*wt)
            normal_parts.append(curve.normal(t))
    shifted=np.concatenate(target_parts,axis=0)
    target_weights=np.concatenate(weight_parts)
    target_normals=np.concatenate(normal_parts,axis=0)
    shifted[:,1] += delta
    air = EwaldHessian(spec.period_um,spec.k0,spec.alpha0,
                       eta=eta,modes=modes_air)
    normflux = 0.
    alpha0=spec.alpha0; beta0=spec.beta0
    # Source densities retain the original Nyström quadrature; target contour
    # integration uses its own independently selected Gauss rule.
    src = nodes.coords
    Nx = nodes.normals[:,0];Nz = nodes.normals[:,1]
    w_mu = nodes.weights*mu
    w_rho = nodes.weights*rho
    for i,x in enumerate(shifted):
        g,gx,gz,hxx,hxz,hzz=air.evaluate(x[0]-src[:,0],x[1]-src[:,1])
        ph=np.exp(1j*(alpha0*x[0]-beta0*x[1]))
        u=ph+np.sum(w_mu*(-Nx*gx-Nz*gz)+w_rho*g)
        ux=1j*alpha0*ph+np.sum(w_mu*(-Nx*hxx-Nz*hxz)+w_rho*gx)
        uz=-1j*beta0*ph+np.sum(w_mu*(-Nx*hxz-Nz*hzz)+w_rho*gz)
        du=target_normals[i,0]*ux+target_normals[i,1]*uz
        normflux += -target_weights[i]*np.imag(np.conj(u)*du)
    return float(normflux/(beta0*spec.period_um))
