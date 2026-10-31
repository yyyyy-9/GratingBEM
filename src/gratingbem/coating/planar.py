"""Flat film on a metal substrate: analytical Fresnel and 4-density BIE.

Coordinates z increasing towards air. The film lies 0<z<t and a wave
arrives from z>t. Fields use exp(-i*omega*t), TE=E_y, TM=H_y.

The BIE is solved in Fourier space without inserting the analytical
Fresnel reflection as a boundary condition; used to independently check
transmission jumps, complex-k branches and Fourier phase conventions.
"""
from __future__ import annotations
import numpy as np


def outgoing_beta(k:complex, alpha:float):
    v=np.sqrt(complex(k)**2-alpha**2+0j)
    if v.imag < 0 or (abs(v.imag)<1e-13 and v.real<0):v=-v
    return v


def fresnel_r(eps_i:complex,eps_j:complex,beta_i:complex,beta_j:complex,pol:str):
    if pol=='TE':
        return (beta_i-beta_j)/(beta_i+beta_j)
    if pol=='TM':
        return (eps_j*beta_i-eps_i*beta_j)/(eps_j*beta_i+eps_i*beta_j)
    raise ValueError('pol must be TE or TM')


def thin_film_r(wavelength_um,theta_deg,epsilon_film,epsilon_substrate, thickness_um,pol):
    """Complex reflection at z=t, referred to the top of the film."""
    if thickness_um<0:raise ValueError('thickness must be nonnegative')
    k0=2*np.pi/wavelength_um
    alpha=k0*np.sin(np.deg2rad(theta_deg))
    b0=outgoing_beta(k0,alpha)
    b1=outgoing_beta(k0*np.sqrt(complex(epsilon_film)),alpha)
    b2=outgoing_beta(k0*np.sqrt(complex(epsilon_substrate)),alpha)
    r01=fresnel_r(1.,epsilon_film,b0,b1,pol)
    r12=fresnel_r(epsilon_film,epsilon_substrate,b1,b2,pol)
    x=np.exp(2j*b1*thickness_um)
    return (r01+r12*x)/(1+r01*r12*x)


def solve_planar_bie(wavelength_um,theta_deg,epsilon_film,epsilon_substrate,thickness_um,pol):
    """4 independent single-layer densities from the two-interface BIE.

    Unknown order: (air top, film top, film bottom, metal bottom).
    This independently implements layer-potential boundary conditions.
    """
    if thickness_um<=0:raise ValueError('BIE needs distinct interfaces; use thin_film_r for zero thickness')
    k0=2*np.pi/wavelength_um; alpha=k0*np.sin(np.deg2rad(theta_deg))
    b0=outgoing_beta(k0,alpha)
    b1=outgoing_beta(k0*np.sqrt(complex(epsilon_film)),alpha)
    b2=outgoing_beta(k0*np.sqrt(complex(epsilon_substrate)),alpha)
    s0=1j/(2*b0);s1=1j/(2*b1);s2=1j/(2*b2)
    e=np.exp(1j*b1*thickness_um)
    f0=np.exp(-1j*b0*thickness_um)
    ga=1 if pol=='TE' else 1
    gl=1 if pol=='TE' else 1/epsilon_film
    gm=1 if pol=='TE' else 1/epsilon_substrate
    A=np.zeros((4,4),complex)
    # u_top_air = u_top_film
    A[0,:]=[s0,-s1,-s1*e,0]
    # u_bottom_film = u_bottom_metal
    A[1,:]=[0,s1*e,s1,-s2]
    # gamma * normal derivative at top; n points UP
    A[2,:]=[ga*(-.5),-gl*(+.5),+gl*(.5*e),0]
    # gamma * normal derivative at bottom
    A[3,:]=[0,gl*(+.5*e),gl*(-.5),-gm*(+.5)]
    rhs=np.asarray([-f0,0,-ga*(-1j*b0*f0),0],complex)
    dens=np.linalg.solve(A,rhs)
    r_origin=s0*dens[0]*np.exp(-1j*b0*thickness_um)
    r_top=r_origin*np.exp(2j*b0*thickness_um)
    residual=np.linalg.norm(A@dens-rhs)/np.linalg.norm(rhs)
    return {'r_top':r_top,'r_origin':r_origin,'densities':dens,'residual':float(residual),
            'reflectance':float(abs(r_top)**2)}


def planar_stack_r(wavelength_um,theta_deg,film_epsilons,thicknesses_um,substrate_epsilon,pol):
    """Stable recursive reflection of planar arbitrary absorbing dielectric films.

    epsilons/thicknesses ordered from AIR to substrate. Avoids exponential
    amplification from transfer matrix products when evanescent modes occur.
    """
    if len(film_epsilons)!=len(thicknesses_um):raise ValueError('material/thickness length mismatch')
    if any(float(t)<0 for t in thicknesses_um):raise ValueError('negative film thickness')
    e=[1+0j]+list(map(complex,film_epsilons))+[complex(substrate_epsilon)]
    k0=2*np.pi/wavelength_um
    a=k0*np.sin(np.deg2rad(theta_deg))
    beta=[outgoing_beta(k0*np.sqrt(q),a) for q in e]
    r=fresnel_r(e[-2],e[-1],beta[-2],beta[-1],pol)
    for j in range(len(e)-3,-1,-1):
        rij=fresnel_r(e[j],e[j+1],beta[j],beta[j+1],pol)
        ph=np.exp(2j*beta[j+1]*thicknesses_um[j])
        r=(rij+r*ph)/(1+rij*r*ph)
    return r
