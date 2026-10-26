"""Custom mixed line/arc grating demo; sketch first and inspect geometry."""
from gratingbem import Profile,Material,Incidence,Simulation,Numerics,calculate
p=Profile.custom(period_um=1.2,vertices=[(0.,0.),(.2,0.),(.2,.08),(.5,.08),(.8,.04),(1.2,0.)])
p=p.rounded(.004)
p.save_svg('custom_shape.svg')
s=Simulation(p,248.33,Incidence.angle(15),0,Material.constant(-8+1j))
r=calculate(s,Numerics(p_order=4,max_panel_wavelengths=.8,max_nodes=800))
print(r.summary())
