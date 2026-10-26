"""Public, unit-explicit simulation model for bare interfaces.

Geometric dimensions use micrometres (µm); optical wavelengths use nanometres.
The solver uses exp(-i*omega*t) and accepts passive complex epsilon.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
from typing import Any
from ._core.profiles import Profile, PathBuilder, from_dict as profile_from_dict
from ._core.segments import Arc, Line
from ._core.solver import Problem as _Problem
from ._core.validation import integer_order


@dataclass(frozen=True)
class Material:
    """Bare half-space material: Rakić-Lorentz–Drude Al or fixed complex epsilon."""
    model: str = 'aluminum_rakic_ld'
    epsilon: complex | None = None

    def __post_init__(self):
        if self.model not in ('aluminum_rakic_ld', 'constant_epsilon'):
            raise ValueError('material model must be aluminum_rakic_ld or constant_epsilon')
        if self.model == 'constant_epsilon' and self.epsilon is None:
            raise ValueError('constant_epsilon requires complex epsilon')
        if self.model == 'aluminum_rakic_ld' and self.epsilon is not None:
            raise ValueError('aluminum_rakic_ld does not accept a constant epsilon')
        if self.epsilon is not None:
            v = complex(self.epsilon)
            if not (math.isfinite(v.real) and math.isfinite(v.imag)) or v.imag < 0 or v == 0:
                raise ValueError('epsilon must be finite, nonzero and passive (Im >= 0)')

    @classmethod
    def aluminum(cls):
        """Bulk Al, Rakić 1998 Lorentz–Drude optical model; no oxide/coating."""
        return cls()

    @classmethod
    def constant(cls, epsilon: complex):
        return cls('constant_epsilon', complex(epsilon))

    def as_dict(self) -> dict:
        if self.model == 'aluminum_rakic_ld':
            return {'model':'aluminum_rakic_ld'}
        return {'model':'constant_epsilon','epsilon_real':float(self.epsilon.real),
                'epsilon_imag':float(self.epsilon.imag)}

    @classmethod
    def from_dict(cls, data: str | dict | None):
        if data is None:
            return cls.aluminum()
        if isinstance(data, str):
            if data.lower() in ('al', 'aluminum', 'aluminum_rakic_ld', 'al_rakic_ld'):
                return cls.aluminum()
            raise ValueError('unknown material name')
        if 'epsilon_real' in data:
            if data.get('model','constant_epsilon')!='constant_epsilon':
                raise ValueError('constant epsilon conflicts with the named material model')
            return cls.constant(complex(float(data['epsilon_real']),float(data.get('epsilon_imag',0))))
        if 'epsilon_imag' in data:
            raise ValueError('epsilon_imag requires epsilon_real')
        return cls.from_dict(data.get('model', 'aluminum_rakic_ld'))


@dataclass(frozen=True)
class Incidence:
    """Incident angle in degrees, or Littrow order (signed diffraction index)."""
    angle_deg: float | None = None
    littrow_order: int | None = None

    def __post_init__(self):
        if (self.angle_deg is None) == (self.littrow_order is None):
            raise ValueError('exactly one incidence mode is required')
        if self.angle_deg is not None and not (-89.99 < float(self.angle_deg) < 89.99):
            raise ValueError('angle must be in (-89.99, 89.99) degrees')
        if self.littrow_order is not None:
            object.__setattr__(self,'littrow_order',integer_order(self.littrow_order,name='Littrow order',nonzero=True))

    @classmethod
    def angle(cls, deg: float):
        return cls(angle_deg=float(deg))

    @classmethod
    def littrow(cls, order: int):
        return cls(littrow_order=order)

    def as_dict(self):
        if self.littrow_order is not None:
            return {'littrow_order':self.littrow_order}
        return {'angle_deg':self.angle_deg}

    @classmethod
    def from_dict(cls, obj):
        if isinstance(obj, (float,int)):
            return cls.angle(float(obj))
        angles=[obj[key] for key in ('angle_deg','theta_deg') if obj.get(key) is not None]
        if len(angles)>1:
            raise ValueError('specify only one angle field: angle_deg or theta_deg')
        return cls(angle_deg=float(angles[0]) if angles else None,
                   littrow_order=obj.get('littrow_order'))


@dataclass(frozen=True)
class Simulation:
    profile: Profile
    wavelength_nm: float
    incidence: Incidence
    target_order: int
    material: Material = Material()

    def __post_init__(self):
        if not isinstance(self.profile,Profile):
            raise TypeError('profile must be a gratingbem.Profile')
        if not isinstance(self.incidence, Incidence):
            raise TypeError('incidence must be an Incidence.angle(...) or Incidence.littrow(...)')
        if not isinstance(self.material, Material):
            raise TypeError('material must be Material.aluminum() or Material.constant(...)')
        if not (math.isfinite(self.wavelength_nm) and self.wavelength_nm>0):
            raise ValueError('wavelength_nm must be positive')
        object.__setattr__(self,'target_order',integer_order(self.target_order,name='target_order'))
        self.to_problem()  # checks Littrow propagation and material consistency

    def to_problem(self) -> _Problem:
        return _Problem(profile=self.profile, wavelength_um=self.wavelength_nm*1e-3,
                        target_order=self.target_order,
                        theta_deg=self.incidence.angle_deg,
                        littrow_order=self.incidence.littrow_order,
                        epsilon=self.material.epsilon)

    @property
    def incidence_deg(self) -> float:
        return self.to_problem().incidence_deg

    def with_parameter(self,parameter: str,value: Any) -> 'Simulation':
        if parameter=='wavelength_nm':return replace(self,wavelength_nm=float(value))
        if parameter=='material.epsilon':return replace(self,material=Material.constant(complex(value)))
        if parameter=='incidence.angle_deg':return replace(self,incidence=Incidence.angle(float(value)))
        if parameter=='incidence.littrow_order':return replace(self,incidence=Incidence.littrow(value))
        if parameter in ('profile',) or parameter.startswith('profile.'):
            p = self.to_problem().with_parameter(parameter,value)
            return replace(self,profile=p.profile)
        if parameter=='target_order':return replace(self,target_order=value)
        raise ValueError(f'unknown scan parameter: {parameter}')

    def as_dict(self) -> dict:
        return {'profile': self.profile.as_dict(), 'wavelength_nm':self.wavelength_nm,
                'incidence':self.incidence.as_dict(),'material':self.material.as_dict(),
                'target_order':self.target_order}

    @classmethod
    def from_dict(cls, config: dict):
        if 'coating' in config or 'layers' in config:
            raise NotImplementedError('Coatings are not supported in the bare-interface package')
        return cls(profile=profile_from_dict(config['profile']),
                   wavelength_nm=float(config['wavelength_nm']),
                   incidence=Incidence.from_dict(config['incidence']),
                   target_order=config.get('target_order',0),
                   material=Material.from_dict(config.get('material')))

    @classmethod
    def from_legacy_problem(cls, p: _Problem):
        return cls(p.profile,1000*p.wavelength_um,
                   Incidence.littrow(p.littrow_order) if p.littrow_order is not None else Incidence.angle(p.theta_deg),
                   p.target_order,Material.aluminum() if p.epsilon is None else Material.constant(p.epsilon))

__all__=['Profile','PathBuilder','Arc','Line','Material','Incidence','Simulation']
