# Changelog

## 0.1.0a1 — 2026-10-26

- Public unit-explicit `Simulation`, `Material`, `Incidence`, `calculate`, `scan`, `scan_grid`, `calculate_adaptive` and `convergence` API.
- Single `gratingbem` package with hidden numerical kernel, instead of versioned public imports.
- CLI `init`, `inspect`, `run`, `scan`, `converge`; machine-readable results and reproducibility manifest.
- Analytic lines/arcs, rounded rectangle/step/echelle profiles, basic quality guards.
- Public scope excludes coating and unvalidated corner-RCIP; initial GitHub release remains a research alpha.
- Publication review rejects fractional diffraction orders and conflicting incidence/material settings; signed Littrow orders use the matching incidence sign. Negative-order reference values are preserved.
- CLI scans report passivity-failed cases and return nonzero, normalize polarization labels, and report filesystem errors clearly; grid scans reject unknown error policies.
- Added analytic Fresnel cases, signed-Littrow and input counterexamples, actual parallel-assembly parity, conservative convergence status checks and a coarse-grid passivity rejection case.
- Numerical formulation inherited from the v16 bare-interface prototype; reference parity is not an independent physical-accuracy certificate.
