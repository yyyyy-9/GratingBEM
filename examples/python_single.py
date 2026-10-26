"""Bare Al echelle grating: one run and outputs. Execute after pip install ."""
from gratingbem import Profile,Material,Incidence,Simulation,Numerics,calculate

profile=Profile.echelle(period_um=1000/79,blaze_deg=74,apex_deg=90,
                        tip_radius_um=.02,valley_radius_um=.05)
simulation=Simulation(profile,248.33,Incidence.littrow(-98),-98,Material.aluminum())
numerics=Numerics(p_order=4,max_panel_wavelengths=1.0,max_nodes=1800)
result=calculate(simulation,numerics,workers=1)
print(result.summary())
result.save('results/python_single')
