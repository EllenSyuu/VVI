"""Build and execute examples/quickstart.ipynb (synthetic data only).

python scripts/make_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    (
        "md",
        """# VVI quickstart

From dated exposure histories to hazard ratios, waning and power — on a **synthetic**
cohort. Nothing in this notebook touches real data.""",
    ),
    (
        "code",
        """import io
import numpy as np
import vvi
from vvi import plots
from IPython.display import Image

print("vvi", vvi.__version__)""",
    ),
    (
        "md",
        """## 1 · A cohort and its histories

`simulate` draws people, their profile, the day of their last exposure (`anchor`) and
recurrent infections in calendar time. `histories` back-fills dated doses and infections
consistent with each profile; `encode_table` turns them back into profiles.""",
    ),
    (
        "code",
        """cohort = vvi.simulate(200_000, seed=1)
history = vvi.histories(cohort, seed=1)
history.head(7)""",
    ),
    (
        "code",
        """profiles = vvi.encode_table(history)
print("round-trip exact:", bool((profiles == cohort.people.set_index("id")["profile"]).all()))
profiles.value_counts()""",
    ),
    (
        "md",
        """## 2 · A ledger instead of a panel

The fit only needs people at risk and events per covariate cell and calendar day.""",
    ),
    (
        "code",
        """ledger = vvi.aggregate(cohort.people, cohort.events, horizon=365, blank_days=28)
print(f"{ledger['at_risk'].sum():,} person-days -> {len(ledger):,} rows")
ledger.head()""",
    ),
    ("md", """## 3 · Fit, hazard ratios, the imprinting contrast"""),
    (
        "code",
        """fit = vvi.fit(ledger, reference="BB", df=7)
fit.table().round(3)""",
    ),
    (
        "code",
        """c = fit.contrast("BBBI", "BBI")
lo, hi = c.ci
truth = cohort.truth.profile_effects
print(f"BBBI vs BBI: HR {c.hr:.2f} (95% CI {lo:.2f}-{hi:.2f}), "
      f"planted {np.exp(truth['BBBI'] - truth['BBI']):.2f}")""",
    ),
    (
        "code",
        """buf = io.BytesIO()
plots.recovery(fit, truth, buf)
Image(buf.getvalue(), width=760)""",
    ),
    ("md", """## 4 · Waning: a covariate that changes during follow-up"""),
    (
        "code",
        """wcohort = vvi.simulate(300_000, seed=2, truth=vvi.Truth(waning=vvi.WANING_EXAMPLE))
bands = vvi.Bands()
wledger = vvi.aggregate(wcohort.people, wcohort.events, horizon=365, bands=bands)
wfit = vvi.fit(wledger, reference="BB", covariates=("sex", "age_band", "since"))
[(e.term, round(e.hr, 3), round(np.exp(vvi.WANING_EXAMPLE[e.term]), 3))
 for e in wfit.covariate_effects("since")]""",
    ),
    ("md", """## 5 · How many people would a smaller effect need?"""),
    (
        "code",
        """sizes = [100_000, 300_000, 1_000_000, 1_500_000]
for hr in (1.1, 1.2, 1.3):
    pw = vvi.power(c, n_fit=200_000, sizes=sizes, log_hr=np.log(hr))
    need = vvi.required_size(c, n_fit=200_000, log_hr=np.log(hr))
    print(f"HR {hr}: power " + ", ".join(f"{n/1e3:.0f}k {p:.0%}" for n, p in zip(sizes, pw))
          + f"  |  80% needs {need:,}")""",
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    for kind, src in CELLS:
        nb.cells.append(
            nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        )
    NotebookClient(
        nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    ).execute()
    out = ROOT / "examples" / "quickstart.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    print(out, out.stat().st_size // 1024, "KiB")


if __name__ == "__main__":
    main()
