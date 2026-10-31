"""Integration tests for interrupt/resume and bounded worker execution."""
import numpy as np
import pytest
from gratingbem import Profile, Simulation, Material, Incidence, Numerics
from gratingbem.coating import CoatedSimulation,Film,calculate,AssemblyOptions
from gratingbem.coating import periodic
from gratingbem.coating.assembly import _matrix_rows,_fingerprint
from gratingbem.coating.value_gradient import EwaldValueGradient
from gratingbem._core.ewald import EwaldHessian
from gratingbem._core.curved import make_curve_nodes


def config():
    sim=Simulation(Profile.flat(1.2),800,Incidence.angle(17),0,Material.constant(-8+.95j))
    coated=CoatedSimulation(sim,Film(70,2.25))
    num=Numerics(p_order=4,panels_per_edge=(2,),modes_air=25,modes_metal=8,oversample=8)
    return coated,num


def test_row_block_matches_original_quadrature():
    coated,num=config()
    spec=coated.to_problem().substrate.spec()
    nodes=make_curve_nodes(coated.to_problem().substrate,num)
    kernel=EwaldHessian(spec.period_um,spec.k0,spec.alpha0,modes=25)
    ref_s,ref_d=periodic._SD(nodes,nodes,kernel,8,same=True)
    s1,d1=_matrix_rows(nodes,nodes,EwaldValueGradient(spec.period_um,spec.k0,spec.alpha0,modes=25),8,same=True,start=0,stop=4)
    s2,d2=_matrix_rows(nodes,nodes,EwaldValueGradient(spec.period_um,spec.k0,spec.alpha0,modes=25),8,same=True,start=4,stop=len(nodes.coords))
    np.testing.assert_allclose(np.vstack((s1,s2)),ref_s,rtol=0,atol=0)
    np.testing.assert_allclose(np.vstack((d1,d2)),ref_d,rtol=0,atol=0)


def test_resumable_matrix_assembly(tmp_path):
    coated,num=config()
    expected=calculate(coated,num,polarizations=('TE',))
    events=[]
    class Stop(Exception):pass
    def abort(event):
        events.append(event)
        if event.get('event')=='block_complete' and not event.get('restored'):
            raise Stop('interrupted after durable block')
    opts=AssemblyOptions(checkpoint_dir=tmp_path/'checkpoint',block_rows=3,
                         workers=1,block_timeout_s=20,progress=abort)
    with pytest.raises(Stop):
        calculate(coated,num,polarizations=('TE',),assembly_options=opts)
    checkpoint_files=list((tmp_path/'checkpoint').rglob('rows_*.npz'))
    assert len(checkpoint_files)>=1
    events=[]
    opts2=AssemblyOptions(checkpoint_dir=tmp_path/'checkpoint',block_rows=3,
                          workers=1,block_timeout_s=20,progress=events.append)
    got=calculate(coated,num,polarizations=('TE',),assembly_options=opts2)
    assert any(e.get('restored') for e in events)
    a=got.raw['polarizations']['TE'];b=expected.raw['polarizations']['TE']
    np.testing.assert_allclose(a['amplitude_real'],b['amplitude_real'],rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(a['amplitude_imag'],b['amplitude_imag'],rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(a['efficiencies'],b['efficiencies'],rtol=1e-12,atol=1e-14)
    assert got.raw['assembly_checkpoint_stats'][0]['restored_blocks']>=1
    # Completely resumed run restores all blocks.
    got2=calculate(coated,num,polarizations=('TE',),assembly_options=opts2)
    assert all(s['computed_blocks']==0 for s in got2.raw['assembly_checkpoint_stats'])
    # Inputs cannot reuse stale matrix checkpoint when wavelength changes.
    num3=Numerics(p_order=4,panels_per_edge=(3,),modes_air=25,modes_metal=8,oversample=8)
    got3=calculate(coated,num3,polarizations=('TE',),assembly_options=opts2)
    assert got3.raw['assembly_checkpoint_stats'][0]['computed_blocks']>0


def test_rejects_invalid_assembly_options():
    with pytest.raises(ValueError):AssemblyOptions(workers=0)
    with pytest.raises(ValueError):AssemblyOptions(block_rows=0)
    with pytest.raises(ValueError):AssemblyOptions(block_timeout_s=-1)


def test_value_gradient_matches_hessian_kernel():
    x=np.array([.0002,.015,.16,.63,1.11])
    z=np.array([.001,0.04,.008,.48,-.006])
    for k in (2*np.pi/.24833, (2*np.pi/.24833)*np.sqrt(-8+.95j)):
        base=EwaldHessian(1000/79,k,10.,modes=55)
        fast=EwaldValueGradient(1000/79,k,10.,modes=55)
        values=base.evaluate(x,z)
        faster=fast.evaluate_vg(x,z)
        for a,b in zip(values[:3],faster):
            np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)


def test_assembly_hard_timeout_kills_workers(tmp_path):
    from gratingbem.coating import AssemblyTimeout
    import multiprocessing
    coated,num=config()
    before={p.pid for p in multiprocessing.active_children()}
    with pytest.raises(AssemblyTimeout):
        calculate(coated,num,polarizations=('TE',),assembly_options=AssemblyOptions(
            checkpoint_dir=tmp_path/'timeout',block_rows=4,workers=2,
            block_timeout_s=0.001))
    after={p.pid for p in multiprocessing.active_children()}
    assert not (after-before), 'a timed-out assembly worker was left running'


def test_parallel_worker_equivalence(tmp_path):
    coated,num=config()
    expected=calculate(coated,num,polarizations=('TE','TM'))
    par=calculate(coated,num,polarizations=('TE','TM'),assembly_options=AssemblyOptions(
        checkpoint_dir=tmp_path/'parallel',block_rows=4,workers=2,block_timeout_s=30))
    for pol in ('TE','TM'):
        np.testing.assert_allclose(expected.raw['polarizations'][pol]['amplitude_real'],
                                         par.raw['polarizations'][pol]['amplitude_real'],rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(expected.raw['polarizations'][pol]['amplitude_imag'],
                                         par.raw['polarizations'][pol]['amplitude_imag'],rtol=1e-12,atol=1e-14)


@pytest.mark.slow
def test_curved_density_matrix_matches_baseline():
    from gratingbem.coating.periodic import assemble
    from gratingbem.coating.assembly import assemble_coated
    from gratingbem.coating import CoatedSimulation, Film
    profile=Profile.echelle(1.3,60,90,tip_radius_um=.04,valley_radius_um=.08)
    case=CoatedSimulation(Simulation(profile,600,Incidence.angle(20),-1,
                                     Material.constant(-8+.95j)),Film(15,2.25))
    num=Numerics(p_order=3,max_panel_wavelengths=.6,
                 min_panels_per_segment=1,grading=1.0,modes_air=20,modes_metal=8,
                 oversample=8,max_nodes=200)
    problem=case.to_problem()
    top0,bot0,mats0,_=assemble(problem,num)
    top1,bot1,mats1,_,_=assemble_coated(problem,num,AssemblyOptions(
        block_rows=12,workers=1,block_timeout_s=60))
    assert len(top0.coords)==len(top1.coords)
    assert len(bot0.coords)==len(bot1.coords)
    for key in mats0:
        for got,exp in zip(mats1[key],mats0[key]):
            np.testing.assert_allclose(got,exp,rtol=1e-12,atol=1e-12)
