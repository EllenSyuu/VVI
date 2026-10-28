# Method

## 1. Question

Hong Kong's 2022 Omicron waves hit a population vaccinated with two very different
platforms — Comirnaty (mRNA, **B**) and CoronaVac (inactivated, **S**) — often
with two, three or four doses before a first infection (**I**). Two questions follow:

1. **Hybrid immunity** — how much protection against reinfection does each
   vaccination–infection *profile* (``BBI``, ``SSSI``, …) carry, relative to a
   reference profile?
2. **Immune imprinting** — among people who were infected, is an *extra
   pre-infection dose* associated with **more** reinfection
   (``HR(VVVI vs VVI) > 1``), as reported for Qatar by Chemaitelly et al. (2023)?

## 2. Profiles

A profile is the ordered exposure history up to and including the first
infection (`vvi.sequence`). Rules applied by `encode()`:

| rule | rationale |
|---|---|
| positives < 28 days apart are one episode | avoids counting a single infection twice |
| only the first infection is an exposure; later ones are outcomes | `BBIII → BBI` |
| doses after the first infection are dropped from the profile | `BIB → BI` (baseline history) |
| same-day dose and positive: dose first | a same-day dose cannot have protected |
| optional variant coding `P` (BA.1/2) / `A` (BA.4/5) by date | variant-specific profiles |

Study filters (`select()`): homologous schedules only, at most four doses.

## 3. Risk set in calendar time

Time scale = calendar day `t` (day 0 = 1 January). A person contributes to day
`t` if they have entered the cohort, have not been censored, and are not inside
the 28-day post-infection window. Events can recur (Andersen–Gill style): after
the blanking window the person is at risk again.

Using **calendar** rather than time-on-study as the time scale means every
profile is compared on the same days, under the same force of infection. The
epidemic curve is absorbed by `η(t)` instead of confounding the profile effects,
and there is no immortal time.

## 4. Model

For person *i* at risk on day *t*,

$$\log\{-\log(1-p_{it})\} = \eta(t) + \alpha^\top x_i + \beta_{\text{profile}(i)}.$$

* The complementary log–log link is the grouped-time proportional-hazards model:
  `exp(β)` is a hazard ratio, directly comparable with a Cox / Andersen–Gill fit.
* `η(t)` is a cubic B-spline with interior knots **equally spaced in calendar
  time** (`df = 7` by default). Fixing knots by the calendar window (not by data
  quantiles) keeps the design identical whatever the row layout — see §5.
* Protection effectiveness relative to the reference: `PE = 1 − HR`.
* Any two profiles are compared with a linear contrast
  `L'β`, `Var = L' Σ L` — `HybridImmunityFit.contrast("BBBI", "BBI")` is the
  imprinting test.

## 5. Sufficient statistics: why sampling is unnecessary

The person-day log-likelihood is

$$\ell = \sum_{i,t} y_{it}\log p_{it} + (1-y_{it})\log(1-p_{it}).$$

Because `p_it` depends on *i* only through its covariate cell *c* (profile × sex
× age band), it collapses to

$$\ell = \sum_{c,t} d_{ct}\log p_{ct} + (n_{ct}-d_{ct})\log(1-p_{ct}),$$

a binomial GLM on `(n_ct, d_ct)` — people at risk and events per cell and day.
Same MLE, same Fisher information, same standard errors. The table has at most
`cells × days` rows (here 48 × 365 = 17,520) regardless of cohort size.

`vvi.riskset.aggregate` builds it without ever creating a person-day row:

1. split each person's follow-up into at-risk intervals (`at_risk_intervals`);
2. scatter `+1` at each interval start and `−1` at its end into a
   `cells × days` difference array, then take a cumulative sum → `n_ct`;
3. scatter events into `d_ct`.

Cost is `O(people + events + cells × days)`. The test-suite checks that the
aggregated and person-day fits agree to `1e-7`, and the benchmark shows the
person-day route exhausting memory long before the aggregated one notices.

**Caveat.** Exactness requires covariates that are constant within a cell-day.
Continuous age or time-since-dose can be handled by discretising them into the
cell key (the number of rows grows with the number of levels, not with people).

## 6. Validation

`vvi.simulate` generates a cohort from the model above with known `β`, `α`,
`η(t)`, staggered entry, censoring and recurrent events, and plants an
imprinting-like pattern (`β_BBBI > β_BBI`). Tests assert:

* every simulated event lands on an at-risk day;
* aggregated counts equal person-day counts cell-by-cell;
* aggregated and person-day estimates coincide;
* every profile effect is recovered within 4 standard errors;
* the planted imprinting contrast is covered by its 95% CI and detected.

## 7. Limitations and extensions

* **Constant effects.** `β` does not wane. A natural extension adds
  time-since-last-exposure bands to the cell key, giving piecewise-constant
  waning curves at the same cost.
* **Selection.** Who gets boosted, and who gets *tested*, is not random. Monge
  et al. (2023) show that apparent imprinting can arise from such selection;
  negative-control outcomes or test-negative designs are the usual checks.
* **Unmeasured confounding.** Comorbidity and socioeconomic status were not
  available in the original analysis.

## References

* Chemaitelly H. et al. History of primary-series and booster vaccination and
  protection against Omicron reinfection. *Science Advances* 9 (2023) eadh0761.
* Monge S. et al. The imprinting effect of covid-19 vaccines: an expected
  selection bias in observational studies. *BMJ* 381 (2023) e074404.
* Andersen P. K., Gill R. D. Cox's regression model for counting processes:
  a large sample study. *Annals of Statistics* 10 (1982) 1100–1120.
* Prentice R. L., Gloeckler L. A. Regression analysis of grouped survival data
  with application to breast cancer data. *Biometrics* 34 (1978) 57–67.
* McMenamin M. E. et al. Vaccine effectiveness of one, two, and three doses of
  BNT162b2 and CoronaVac against COVID-19 in Hong Kong. *Lancet Infectious
  Diseases* 22 (2022) 1435–1443.
