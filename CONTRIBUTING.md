# Contributing

Thank you for your interest in VVI. Issues and pull requests are welcome.

## Setup

```bash
git clone https://github.com/EllenSyuu/VVI && cd VVI
pip install -e ".[dev]"
pytest --cov            # 38 tests, ~3 minutes
ruff check src tests scripts && ruff format --check src tests scripts && mypy
```

## Ground rules

- **No data, ever.** Do not commit records, extracts, schemas or results derived from real
  data — not even aggregates. Tests, examples and figures use `vvi.simulate` only. The CI data
  guard (`python scripts/check_no_data.py`) enforces the common cases; the rest is on us.
- **Exactness is the contract.** Any change to `riskset` must keep the aggregated and
  person-day fits identical; extend `tests/test_riskset_model.py` or
  `tests/test_waning_power.py` when you add a cell type.
- **Figures** are drawn with `vvi.plots` so they stay in one visual language; regenerate them
  with the scripts in `scripts/` rather than editing images.

## Releasing

Update `CHANGELOG.md`, bump the version in `pyproject.toml`, `src/vvi/__init__.py` and
`CITATION.cff`, then tag `vX.Y.Z`.
