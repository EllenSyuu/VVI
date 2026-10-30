# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-10-30

### Added
- **Waning.** `vvi.Bands` cuts time since the last exposure into bands; `aggregate(...,
  bands=)` splits at-risk intervals at band edges so the ledger stays exact while people
  move between cells. `HybridImmunityFit.covariate_effects()` returns the waning curve.
- **Power.** `vvi.power` and `vvi.required_size` give Wald power and the cohort size a
  contrast needs, from a single fit; `scripts/power_study.py` checks them by Monte Carlo.
- **Dated histories.** `vvi.histories` back-fills doses and infections consistent with each
  synthetic profile; `vvi.encode_table` encodes a long event table per person.
- `fit(covariates=...)` accepts any categorical covariate and a `{name: reference}` mapping.
- Static, dependency-free interactive explorer (`docs/explorer`, built by
  `scripts/build_site.py`, deployed on demand to GitHub Pages) with 21 scenarios.
- `vvi.plots` rebuilt on a light/dark theme system: hero, recovery, calendar, waning,
  power, histories and benchmark figures, plus GIF animation; `scripts/render_assets.py`
  re-renders every README image from synthetic data.
- Tutorial notebook, Makefile, NOTICE, figure smoke tests, `py.typed`, mypy and coverage
  gates in CI.

### Changed
- The simulator records each person's last exposure (`anchor`); optional waning effect.
- CLI prints p-values as `<0.001` instead of rounding to zero.

## [0.1.0] — 2026-10-28

### Added
- Profile encoding, calendar-time risk sets, sufficient-statistic aggregation, cloglog GLM
  with a B-spline calendar baseline, contrasts, synthetic simulator, CLI demo, benchmark,
  data guard.

[0.2.0]: https://github.com/EllenSyuu/VVI/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/EllenSyuu/VVI/releases/tag/v0.1.0
