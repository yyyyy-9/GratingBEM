"""Grid of corner angles and rounding; failure recorded per case."""
from gratingbem import Profile,Material,Incidence,Simulation,Numerics,scan_grid
profile=Profile.echelle(1000/79,blaze_deg=74,apex_deg=90,
                        tip_radius_um=.02,valley_radius_um=.05)
sim=Simulation(profile,248.33,Incidence.littrow(-98),-98,Material.aluminum())
settings=Numerics(p_order=4,max_panel_wavelengths=1.0,max_nodes=2000)
results=scan_grid(sim,{'profile.apex_deg':[85,90],
                       'profile.tip_radius_um':[.01,.02]},settings,workers=1,on_error='record')
results.save('results/python_scan')
print(results.rows())
