"""Aluminum bulk Lorentz-Drude permittivity (Rakić 1998)."""
WAVELENGTH = 0.24833

def eps_al_rakic_ld(wavelength=WAVELENGTH):
    # A. D. Rakic et al. Appl Opt 37, 5271 (1998), Al Lorentz-Drude.
    # Source params: doi:10.1364/AO.37.005271, eV.
    e=1.2398419843320026/wavelength
    wp=14.98; f0=.523;G0=.047
    f=[.227,.050,.166,.030]; gamma=[.333,.312,1.351,3.382];osc=[.162,1.544,1.808,3.473]
    return 1-f0*wp**2/(e*(e+1j*G0))+sum(fj*wp**2/(oj**2-e**2-1j*e*gj) for fj,oj,gj in zip(f,osc,gamma))
