"""Opt-in Bloch-equivalent multi-RHS KM1, with finite-Ewald validation.

Only identical physical profile/wavelength/epsilon and incident alpha differing
by integer reciprocal lattice vectors are eligible. Reuse is NOT valid for
arbitrary incident angles, wavelengths or blaze angles. Checks *both* Ewald
kernels at representative off-boundary separations and rejects weak convergence.
This utility never changes default calculate() behavior.
"""
from __future__ import annotations
import json
import math
import time
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from ._core.solver import (Calculation,make_profile_nodes,rayleigh_mu_rho,
                           build_matrices,rayleigh_cutoff_diagnostic)
from ._core.mesh_cache import make_nodes_cached
from ._core.curved import build_matrices_curved,rayleigh_curved
from ._core.ewald import EwaldHessian

KERNEL_REL_TOL = 2e-10
MAX_RHS_COLUMNS = 32
BLOCH_PHASE_TOL = 3e-12


def _sim_key(sim):
    """Safe grouping needs the entire geometry (not only period), material and k0."""
    return (json.dumps(sim.profile.as_dict(),sort_keys=True,default=str),
            float(sim.wavelength_nm),
            json.dumps(sim.material.as_dict(),sort_keys=True,default=str))


def _equiv(p1,p2):
    if p1.profile.period_um != p2.profile.period_um or p1.wavelength_um != p2.wavelength_um:
        return False
    g=2*math.pi/p1.profile.period_um
    q=(p2.spec().alpha0-p1.spec().alpha0)/g
    return abs(q-round(q)) < BLOCH_PHASE_TOL


def groups(simulations):
    """Stable, input-ordered groups, using a near-linear Bloch-phase index.

    Exact physical keys prevent accidental wavelength/geometry/material reuse.
    A coarse phase bin proposes candidates, then `_equiv` verifies the exact
    reciprocal-lattice difference; adjoining bins and periodic wrapping handle
    near-integer floating-point rounding without brittle hash equality.
    """
    bucket_width=1e-9  # deliberately > BLOCH_PHASE_TOL
    nbuckets=int(round(1./bucket_width))
    grouped=[]
    lookup={}
    representatives=[]
    for i,sim in enumerate(simulations):
        physical_key=_sim_key(sim)
        p=sim.to_problem()
        bloch_phase=(p.spec().alpha0*p.profile.period_um/(2*math.pi))%1.
        bucket=int(bloch_phase/bucket_width)%nbuckets
        match=None
        for offset in (-1,0,1):
            for j in lookup.get((physical_key,(bucket+offset)%nbuckets),()):
                if _equiv(representatives[j],p):
                    match=j
                    break
            if match is not None:break
        if match is None:
            j=len(grouped)
            grouped.append([i]);representatives.append(p)
            lookup.setdefault((physical_key,bucket),[]).append(j)
        else:
            grouped[match].append(i)
    return grouped


def kernel_discrepancy(sims,num):
    """Check finite-truncation Ewald equivalence for six components and both media.

    This is a cheap *diagnostic sampling*, not a mathematical proof that the
    entire dense operator is converged. Low `modes_air` typically fails.
    """
    if len(sims)<2:return 0.0
    problems=[s.to_problem() for s in sims]
    p0=problems[0]
    d=p0.profile.period_um; wavelength=p0.wavelength_um;k0=2*np.pi/wavelength
    eps=p0.epsilon_value
    # Include z~0, subwavelength and far-field, with noncoincident periodic x.
    dx=d*np.array([.001,.01,.07,.2,.38,.7])
    # Bound test heights by a fraction of the period: very high reciprocal
    # modes at large dz can overflow exp(b*dz) before erfc cancels it.
    # Avoid introducing artificial NaNs in this preflight while sampling z=0.
    dz=min(wavelength,0.25*d)*np.array([0.,.002,.02,.1,.3,.75])
    maximum=0.
    for wave_number,nmodes in [(k0,num.modes_air), (k0*np.sqrt(eps+0j),num.modes_metal)]:
        kernels=[EwaldHessian(d,wave_number,p.spec().alpha0,
                               eta=num.eta,modes=nmodes) for p in problems]
        ref=kernels[0].evaluate(dx,dz)
        for kernel in kernels[1:]:
            chk=kernel.evaluate(dx,dz)
            for a,b in zip(ref,chk):
                # A truncated Green kernel that overflows/underflows to NaN
                # can NEVER certify equivalence. Fail closed.
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    return float('inf')
                err=np.max(np.abs(a-b))/max(np.max(np.abs(a)),np.max(np.abs(b)),1.)
                if not np.isfinite(err):return float('inf')
                maximum=max(maximum,float(err))
    return maximum


def shareable(simulations,num,*,kernel_tol=KERNEL_REL_TOL):
    if len(simulations)<2:return False,0.
    key=_sim_key(simulations[0]);p0=simulations[0].to_problem()
    for s in simulations:
        p=s.to_problem()
        if _sim_key(s)!=key or not _equiv(p0,p):return False,float('inf')
        wood=rayleigh_cutoff_diagnostic(p)
        if wood['status']!='outside_warning_band':return False,float('inf')
    discrepancy=kernel_discrepancy(simulations,num)
    return (discrepancy <= kernel_tol),discrepancy


def calculate_group(simulations,num,polarizations=('TE','TM'),*,kernel_discrepancy_value=None):
    """One assembly & one LU per polarization; all RHS solved independently.

    The accepted group has passed `shareable`; callers must enforce this gate.
    Reports amortized timings plus explicit shared timing fields. For each
    incidence direction Rayleigh modes are recomputed using its own alpha0.
    """
    if len(simulations)<2:raise ValueError('requires at least two equivalent incidents')
    problems=[s.to_problem() for s in simulations]
    t0=time.perf_counter()
    nodes=[make_nodes_cached(p,num,make_profile_nodes) for p in problems]
    n=len(nodes[0].coords)
    eps=problems[0].epsilon_value
    assembly = build_matrices_curved if problems[0].profile.has_arcs else build_matrices
    t_mesh=time.perf_counter()
    ops=assembly(nodes[0],eps,mode_air=num.modes_air,mode_metal=num.modes_metal,
                 oversample=num.oversample,eta=num.eta)
    t_asm=time.perf_counter()
    k1=nodes[0].spec.k0;k2=k1*np.sqrt(eps+0j)
    scaling=np.r_[np.ones(n),np.ones(n)/k1]
    I=np.eye(n)
    all_modes={p:{} for p in polarizations}
    all_amps={p:{} for p in polarizations}
    all_effs={p:{} for p in polarizations}
    all_diags={p:{} for p in polarizations}
    lu_time=0.;rhs_time=0.;rhs_batches=0
    # Reuse one LU factorization per polarization; solve all independent
    # incidence columns in short batches to bound temporary memory.
    max_rhs_columns=MAX_RHS_COLUMNS
    for pol in polarizations:
        chi=1+0j if pol=='TE' else eps
        z=chi*k2;c=z/abs(z)
        a1=1/(1+chi);a2=chi/(1+chi);a3=1/(c+chi);a4=chi/(c+chi)
        A=np.block([[I-a2*ops['K2']+a1*ops['K1'], -a1*(c*ops['S2']-ops['S1'])],
                    [a4*ops['Td'],I+c*a3*ops['Ka2']-a4*ops['Ka1']]])
        A*=scaling[:,None]
        tl=time.perf_counter();lu,piv=lu_factor(A,check_finite=False)
        lu_time+=time.perf_counter()-tl
        ts=time.perf_counter()
        for start in range(0,len(nodes),max_rhs_columns):
            subset=nodes[start:start+max_rhs_columns]
            rhs_columns=[]
            for node in subset:
                sp=node.spec
                ui=np.exp(1j*(sp.alpha0*node.coords[:,0]-sp.beta0*node.coords[:,1]))
                du=1j*(sp.alpha0*node.normals[:,0]-sp.beta0*node.normals[:,1])*ui
                rhs_columns.append(np.r_[-2*a1*ui,2*a4*du]*scaling)
            rhs=np.column_stack(rhs_columns)
            dens=lu_solve((lu,piv),rhs,check_finite=False)
            rhs_batches+=1
            residuals=np.linalg.norm(A@dens-rhs,axis=0)/np.linalg.norm(rhs,axis=0)
            for offset,node in enumerate(subset):
                j=start+offset
                den=dens[:,offset]
                m,a,e=(rayleigh_curved(node,den[:n],den[n:]) if problems[j].profile.has_arcs
                        else rayleigh_mu_rho(node,den[:n],den[n:]))
                target=e[m==problems[j].target_order]
                all_modes[pol][j]=m;all_amps[pol][j]=a;all_effs[pol][j]=e
                all_diags[pol][j]=dict(target_efficiency=float(target[0]) if len(target) else None,
                                        reflection_sum=float(e.sum()),matrix_residual=float(residuals[offset]),
                                        passes_passivity=bool(e.sum()<=1+5e-4))
        rhs_time+=time.perf_counter()-ts
    t_end=time.perf_counter()
    ncases=len(nodes);results=[]
    for j,p in enumerate(problems):
        modes=all_modes[polarizations[0]][j]
        diags={pol:all_diags[pol][j] for pol in polarizations}
        wood=rayleigh_cutoff_diagnostic(p)
        diags.update(n_nodes=int(n),n_panels=int(len(nodes[j].panels['ds'])),
                     panels_per_edge=list(map(int,nodes[j].panels['counts_by_edge'])),
                     n_propagating_orders=int(len(modes)),wood_cutoff=wood,
                     bloch_group_reuse=dict(size=ncases,kernel_sample_rel_error=kernel_discrepancy_value,
                                            exact_finite_matrix_equivalence_not_guaranteed=True,
                                            rhs_batches=rhs_batches, multi_rhs=True))
        warns=[]
        if not np.any(modes==p.target_order):
            warns.append('Target order is evanescent; efficiency=None')
        for pol in polarizations:
            if not diags[pol]['passes_passivity']:
                warns.append(f'{pol}: R_total>1 violates passive-medium check; refine/discard result')
        if p.epsilon is None:warns.append('Al Rakić LD bulk optical constants; no Al2O3/coating/roughness')
        warns.append('Research KM1 solver without general-profile RCIP or rigorous error bounds')
        if p.profile.has_arcs:
            warns.append('Analytic circular arc geometry; curved-panel quadrature, not polygon approximation')
        warns.append('Opt-in Bloch-equivalent matrix/LU reuse; sampled finite-Ewald kernel equivalence only')
        timing=dict(mesh_s=(t_mesh-t0)/ncases, assembly_s=(t_asm-t_mesh)/ncases,
                    solve_s=(t_end-t_asm)/ncases, energy_diagnostic_s=0.,
                    total_s=(t_end-t0)/ncases,
                    shared_mesh_s=t_mesh-t0,shared_assembly_s=t_asm-t_mesh,
                    shared_lu_factor_s=lu_time,shared_rhs_postprocess_s=rhs_time,
                    shared_total_s=t_end-t0)
        results.append(Calculation(p,num,modes,
                                  {pol:all_amps[pol][j] for pol in polarizations},
                                  {pol:all_effs[pol][j] for pol in polarizations},
                                  diags,timing,warns))
    # An accepted *sampled* kernel preflight is not a proof. Fail closed on
    # nonfinite outputs or very large linear residuals; scan dispatcher then
    # independently re-solves every channel of this group.
    for result in results:
        for pol in polarizations:
            if (not np.isfinite(result.amplitudes[pol]).all() or
                not np.isfinite(result.efficiencies[pol]).all() or
                result.diagnostics[pol]['matrix_residual']>1e-9):
                raise ArithmeticError('nonfinite or inaccurate grouped Bloch solution')
    return results
