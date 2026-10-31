"""Public, unit-explicit models for an *experimental* conformal coating.

The stable bare-metal ``Simulation`` and ``gratingbem.calculate`` are not
modified. Coatings are currently limited to ONE homogeneous dielectric film
on one semi-infinite substrate. Thickness is in nanometres, geometry in µm.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import csv
import json
import math
from typing import Any
import numpy as np
from ..model import Simulation
from .._core.profiles import Profile, from_dict as profile_from_dict
from .periodic import CoatedProblem


@dataclass(frozen=True)
class Film:
    """Single homogeneous passive film of thickness_nm and relative epsilon.

    ``epsilon`` is the *relative permittivity*, not refractive index. To use
    a complex refractive index n+ik, set epsilon=(n+1j*k)**2.
    """
    thickness_nm: float
    epsilon: complex
    label: str = 'dielectric film'

    def __post_init__(self):
        t = float(self.thickness_nm)
        eps = complex(self.epsilon)
        if not math.isfinite(t) or t <= 0:
            raise ValueError('film thickness_nm must be strictly positive and finite')
        if not (math.isfinite(eps.real) and math.isfinite(eps.imag)) or eps == 0 or eps.imag < 0:
            raise ValueError('film epsilon must be nonzero, finite and passive (imag>=0)')
        object.__setattr__(self, 'thickness_nm', t)
        object.__setattr__(self, 'epsilon', eps)

    @classmethod
    def from_index(cls, thickness_nm: float, n: complex, label: str='dielectric film'):
        return cls(thickness_nm=thickness_nm, epsilon=complex(n)**2, label=label)

    def as_dict(self):
        return dict(thickness_nm=self.thickness_nm,
                    epsilon_real=self.epsilon.real, epsilon_imag=self.epsilon.imag,
                    label=self.label)

    @classmethod
    def from_dict(cls, data):
        return cls(float(data['thickness_nm']),
                   complex(float(data['epsilon_real']), float(data.get('epsilon_imag',0))),
                   str(data.get('label','dielectric film')))


@dataclass(frozen=True)
class CoatedSimulation:
    """Base metal interface plus one conformal dielectric film.

    A smooth, tangent-continuous profile can be automatically normal-offset;
    sharp corners require prior rounding or an explicit outer profile.
    """
    substrate: Simulation
    film: Film
    outer_profile: Profile | None = None

    def __post_init__(self):
        if not isinstance(self.substrate, Simulation):
            raise TypeError('substrate must be a gratingbem.Simulation')
        if not isinstance(self.film, Film):
            raise TypeError('film must be a gratingbem.coating.Film')
        if self.outer_profile is not None:
            if not isinstance(self.outer_profile, Profile):
                raise TypeError('outer_profile must be Profile')
            if not math.isclose(self.outer_profile.period_um, self.substrate.profile.period_um,
                                rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError('outer_profile must have same period as substrate profile')

    def to_problem(self):
        return CoatedProblem(self.substrate.to_problem(), self.film.epsilon,
                             self.film.thickness_nm / 1000., self.outer_profile)

    def with_thickness(self, thickness_nm: float):
        return replace(self, film=replace(self.film, thickness_nm=float(thickness_nm)))

    def as_dict(self):
        data={'substrate':self.substrate.as_dict(),'film':self.film.as_dict()}
        if self.outer_profile is not None:
            data['outer_profile']=self.outer_profile.as_dict()
        return data

    @classmethod
    def from_dict(cls, config):
        outer=config.get('outer_profile')
        return cls(substrate=Simulation.from_dict(config['substrate']),
                   film=Film.from_dict(config['film']),
                   outer_profile=None if outer is None else profile_from_dict(outer))


@dataclass
class CoatedResult:
    """Dimensionless efficiencies and explicit passivity/energy diagnostics."""
    simulation: CoatedSimulation
    numerics: Any
    raw: dict

    @property
    def status(self):
        return self.raw['status']

    @property
    def modes(self):
        return np.asarray(next(iter(self.raw['polarizations'].values()))['modes'],int)

    def efficiency(self, order: int, polarization='TE', *, percent=True, allow_failed=False):
        data=self.raw['polarizations'][polarization.upper()]
        if data['status'].startswith('failed') and not allow_failed:
            raise RuntimeError('Coated solve failed passivity/energy checks; efficiency is NOT physical. '
                               'Use allow_failed=True only for numerical debugging.')
        inds=np.flatnonzero(np.asarray(data['modes'])==order)
        if len(inds)!=1:
            raise ValueError(f'order {order} is not propagating')
        e=float(data['efficiencies'][int(inds[0])]);return e*(100 if percent else 1)

    def summary(self):
        d=dict(model=self.raw['model'], status=self.status,
               validation_scope='energy/passivity only; no mesh convergence proof',
               ready_for_design=False,
               wavelength_nm=self.simulation.substrate.wavelength_nm,
               period_um=self.simulation.substrate.profile.period_um,
               incidence_deg=self.simulation.substrate.incidence_deg,
               target_order=self.simulation.substrate.target_order,
               film_thickness_nm=self.simulation.film.thickness_nm,
               film_epsilon=str(self.simulation.film.epsilon),
               n_top=self.raw['n_top'],n_bottom=self.raw['n_bottom'],
               assembly_seconds=self.raw['assembly_seconds'],
               total_seconds=self.raw['total_seconds'])
        if 'assembly_checkpoint_stats' in self.raw:
            d['assembly_checkpoint_stats']=self.raw['assembly_checkpoint_stats']
        for pol,p in self.raw['polarizations'].items():
            d[pol+'_target_pct']=(None if p['target_efficiency'] is None or p['status'].startswith('failed')
                                   else 100*p['target_efficiency'])
            d[pol+'_raw_target_pct_for_debug']=(None if p['target_efficiency'] is None
                                   else 100*p['target_efficiency'])
            d[pol+'_reflect_pct']=100*p['reflection_sum']
            d[pol+'_film_absorb_pct']=100*p['film_absorption_flux']
            d[pol+'_substrate_absorb_pct']=100*p['metal_absorption_flux']
            d[pol+'_energy_error_pp']=100*p['energy_balance_error']
            d[pol+'_status']=p['status']
        return d

    def save(self, directory):
        path=Path(directory);path.mkdir(parents=True,exist_ok=True)
        (path/'summary.json').write_text(json.dumps(self.summary(),indent=2,ensure_ascii=False),encoding='utf-8')
        (path/'simulation.json').write_text(json.dumps(self.simulation.as_dict(),indent=2,ensure_ascii=False),encoding='utf-8')
        (path/'diagnostics.json').write_text(json.dumps(self.raw,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
        self.simulation.substrate.profile.save_svg(path/'substrate_profile.svg')
        self.simulation.to_problem().top_profile.save_svg(path/'outer_profile.svg')
        with (path/'orders.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f)
            ps=tuple(self.raw['polarizations'])
            writer.writerow(['order']+[f'{pol}_efficiency' for pol in ps])
            for j,m in enumerate(self.modes):
                writer.writerow([int(m)]+[self.raw['polarizations'][pol]['efficiencies'][j] for pol in ps])
        return path
