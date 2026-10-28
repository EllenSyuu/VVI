# Data governance

The research this repository grew out of used individual-level records
(vaccinations, confirmed cases, hospital episodes) provided under licence by
Hong Kong data custodians for that study only. Those records may not be
transmitted or released in any form.

This repository is therefore **code-only by design**:

| in the repository | never in the repository |
|---|---|
| a clean-room implementation of the method | any record, extract, sample or schema of the licensed data |
| a simulator with invented parameters | estimates, tables or figures computed from licensed data |
| tests and figures driven by synthetic cohorts | counts or summary statistics of the study population |
| documentation of the method | file paths, server names or credentials of the analysis environment |
| | copies of published papers |

Every number and figure in this repository comes from `vvi.simulate` and says so.

## Guard rails

* `.gitignore` blocks common data and result formats (`*.csv`, `*.feather`,
  `*.parquet`, `*.sas7bdat`, `*.xlsx`, `*.rds`, …) and `data/`, `results/`, `private/`.
* `scripts/check_no_data.py` runs in CI and fails the build if a data-like file
  or a field name of the licensed source tables is committed (project-specific
  patterns are kept in an untracked `.data-guard-patterns` file).

## Using the code on restricted data

Run it inside the data custodian's approved environment. Keep any adapter that
reads the real schema in a private location outside this repository, and export
only what the licence allows.
