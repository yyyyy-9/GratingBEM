"""EXPERIMENTAL, opt-in coating tools for GratingBEM.

The stable ``gratingbem.calculate`` is still the bare-metal single-interface
KM1 solver; coated calculations currently use a two-interface first-kind BIE
with independent physical validation. No multilayer patterned solver yet.
"""
from .model import Film, CoatedSimulation, CoatedResult
from .api import calculate, planar_reflection, scan_thickness
from .geometry import offset_profile

__all__ = ['Film','CoatedSimulation','CoatedResult','calculate',
           'planar_reflection','scan_thickness','offset_profile',
           ]

from .assembly import AssemblyOptions, AssemblyTimeout, AssemblyWorkerError
__all__ += ["AssemblyOptions","AssemblyTimeout","AssemblyWorkerError"]
