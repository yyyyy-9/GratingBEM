"""GratingBEM: 2-D periodic TE/TM boundary integral solver for bare interfaces.

Units: wavelength_nm in nm; period_um/height/radius in micrometres.
The library is research software, not a certified design or manufacturing tool.
"""
__version__='0.2.0.dev9'
from .model import Profile,PathBuilder,Line,Arc,Material,Incidence,Simulation
from .api import (Numerics,Execution,AdaptiveOptions,Calculation,BalancedRun,
                  calculate,calculate_adaptive,calculate_many,BatchResult,scan,scan_grid,convergence,efficiency)
__all__=['Profile','PathBuilder','Line','Arc','Material','Incidence','Simulation','Numerics',
         'Execution','AdaptiveOptions','Calculation','BalancedRun','calculate',
         'calculate_adaptive','calculate_many','BatchResult','scan','scan_grid','convergence','efficiency']
