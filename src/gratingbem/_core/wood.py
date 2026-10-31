"""Rayleigh/Wood cutoff proximity diagnostics for quasi-periodic diffraction.

The diagnostic does not stabilize the Green function or certify accuracy. It
provides a dimensionless preflight measure so angle/wavelength scans can flag
cases where the existing Ewald representation needs special care.
"""
from __future__ import annotations
import math

# Heuristic only. No universal accuracy guarantee follows from this threshold.
WOOD_WARNING_BETA_RATIO = 5.0e-2
WOOD_NUMERICAL_CUTOFF_BETA_RATIO = 1.0e-10


def rayleigh_cutoff_diagnostic(problem) -> dict:
    """Return nearest Rayleigh cutoff mode and angular distance.

    Uses alpha_m/k0 = sin(theta) + m*lambda/d. The candidate search is finite
    and covers both alpha_m/k0=+1 and -1 for every physical cutoff order.
    """
    period = float(problem.profile.period_um)
    wavelength = float(problem.wavelength_um)
    theta = float(problem.incidence_deg)
    ratio = wavelength / period
    sine = math.sin(math.radians(theta))
    bound = int(math.ceil(2.0 / ratio)) + 10
    nearest = None
    cutoff_angles = []
    n_propagating = 0
    for m in range(-bound, bound + 1):
        alpha_ratio = sine + m * ratio
        gap = abs(abs(alpha_ratio) - 1.0)
        beta_ratio = math.sqrt(abs(1.0 - alpha_ratio * alpha_ratio))
        if abs(alpha_ratio) < 1.0:
            n_propagating += 1
        candidate = dict(order=int(m), alpha_over_k0=float(alpha_ratio),
                         relative_alpha_gap=float(gap), beta_over_k0=float(beta_ratio))
        if nearest is None or gap < nearest['relative_alpha_gap']:
            nearest = candidate
        for sign in (-1, 1):
            s_cut = sign - m * ratio
            if -1.0 <= s_cut <= 1.0:
                angle_cut = math.degrees(math.asin(s_cut))
                cutoff_angles.append(dict(order=int(m), sign=int(sign),
                    angle_deg=float(angle_cut),
                    angular_distance_deg=float(abs(theta - angle_cut))))
    cutoff_angles.sort(key=lambda q: q['angular_distance_deg'])
    min_beta = nearest['beta_over_k0']
    if min_beta <= WOOD_NUMERICAL_CUTOFF_BETA_RATIO:
        status = 'at_or_numerically_on_cutoff'
    elif min_beta < WOOD_WARNING_BETA_RATIO:
        status = 'near_cutoff_heuristic'
    else:
        status = 'outside_warning_band'
    return dict(incidence_angle_deg=theta, wavelength_um=wavelength,
        period_um=period, period_over_wavelength=period/wavelength,
        n_propagating_orders=n_propagating, nearest_mode=nearest,
        nearest_cutoff_angles=cutoff_angles[:4], min_abs_beta_over_k0=min_beta,
        warning_threshold_beta_ratio=WOOD_WARNING_BETA_RATIO,
        status=status,
        disclaimer='Geometric cutoff proximity only; not an accuracy certificate or Wood-stable formulation.')
