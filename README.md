# VN1 Forecasting — "5 Ways to Improve Forecasting Accuracy Fast"

Do five well-known data-side forecasting techniques actually work? This
measures all five against one model trained on the **full** training set, on
real retail data, scored against a published competition leaderboard.

Every result below is reproducible with the scripts here.

## Why this dataset

VN1 is real retail data: weekly units for 15,053 product-warehouse series from
46 e-vendors, 328 warehouses and 11,171 products, July 2020 to April 2024,
provided by Flieber for the 2024 competition.

It is also *ragged*, in all the ways a live catalogue is and most public
forecasting benchmarks are not:

| | |
|---|---|
| Series launching after week 0 | **85.2%** (median: week 72 of 170) |
| Series dead before the final week | **55.9%** |
| Median active span | **69 of 170 weeks** |
| Zeros among observed weeks | **71.8%** |
| Share of units in the top 5% of series | **72.8%** |
| Strata (warehouse x velocity decile) | **1,486, median 18 rows** |
| Strata a random 5,000-row draw leaves empty | **684 of 1,486** |

That last row is why Case 1 is testable here at all: the uneven grouping that
stratified sampling exists to fix is genuinely present. On a balanced panel —
where every item, store and period exists and strata come out evenly sized —
the technique has nothing to rescue, and no amount of tuning changes that.

VN1 also ships the official scoring code, two benchmarks and all five winning
forecasts, so results sit on a real leaderboard of 978 competitors.

## Layout

```
build_vn1_features.py    raw weekly panel -> modelling frame + FEATURE_GROUPS
vn1_lgbm.py              shared LightGBM model, official scoring, benchmarks
vn1_sampling2.py         Case 1: sampling vs full data, with fit timings
vn1_ablation2.py         Cases 2-5 against the full-data baseline
vn1_caveats.py           allocation exponent (Case 1) + encoding source (Case 3)
vn1_combo.py             combining only the significant techniques

make_timeline.py         plot: train/test origins
make_coverage.py         plot: stratum coverage by budget
make_sampling_plots.py   plots: Case 1 accuracy, speedup, long tail
make_waterfall.py        plot: both metrics, all four techniques

data/                    experiment results (CSV)
```

Three things are deliberately not tracked: the upstream `VN1/` data, which
belongs to its authors; the built `.parquet` frames, which rebuild in about a
minute; and the plots, which the `make_*.py` scripts write to a local
`figures/` folder. Everything in this repo is source.

## Get the data

```bash
git clone https://github.com/SupChains/VN1.git
```

That is all the setup there is. `VN1/` must sit beside the scripts; every
script resolves it relative to its own location. The data is CC-BY-4.0 and is
deliberately not vendored here, so you always get the authors' current copy.

## Run

```bash
pip install -r requirements.txt
python build_vn1_features.py    # ~1 min   -> data/vn1_{train,test}.parquet
python vn1_lgbm.py              # ~15 s    benchmarks + full-data baseline
python vn1_ablation2.py         # ~3 min   Cases 2-5, 5 seeds each
python vn1_sampling2.py         # ~15 min  Case 1, 5 budgets x 8 seeds
python vn1_caveats.py           # ~12 min  the two caveat experiments
```

`vn1_lgbm.py` reproduces the two published benchmarks exactly — naive 50.70%,
12-week moving average 80.46%. If those two numbers do not come out, the data
wiring is wrong and nothing downstream is worth reading.

## The task

Mirrors the competition. Phase 0 gives 170 weeks of history, Phase 1 the next
13, Phase 2 the 13 weeks of the final ranking. Competitors forecast Phase 2
from Phase 0 + Phase 1.

One row per (series, forecast origin), as the timeline above shows. The target
is total units over the 13 weeks from the origin; every feature comes from
weeks strictly before it. The test origin is the first Phase 2 week, with all
183 prior weeks available — exactly what competitors had.

Score is the official metric, `MAE% + |Bias%|`. The 13-week total is spread
flat across the 13 weeks, as both official benchmarks do.

**The benchmarks reproduce the published figures exactly — naive 50.70%,
12-week moving average 80.46%.** That is the check that the data wiring is
right, and it runs at the top of `vn1_lgbm.py`.

## Baseline

LightGBM on all 135,477 rows with volume-and-trend features only (16
features): **51.22%** (sd 0.13 across 5 seeds), fitting in ~4.1 s.

Two choices that carry real weight:

1. **Target is the log ratio of the next 13 weeks to the last 13**, not the
   level. Forecasting the level scores 64.76% — worse than naive — because the
   model burns capacity re-deriving a baseline it was handed.
2. **MAE objective** (`regression_l1`). The metric is MAE-based and the target
   reaches ~90,000 units; squared error lets a few giant series own every
   split. Poisson loss diverges outright on this tail (699%).

## Results

### Cases 2-5, against the full-data baseline

Five seeds each, paired against the baseline seed by seed.

```
                              n_feats   score   delta       p     mae  d_mae    bias     slow
0 Baseline (volume + trend)        16   51.22    0.00     ---   46.30   0.00   -4.92   176.87
2 Velocity classification          17   51.24   +0.02  0.8997   46.24  -0.06   -5.00   176.76
3 Target encoding (Warehouse)      17   50.21   -1.01  0.0070   46.81  +0.51   -3.40   174.06
4 Value proposition                26   50.76   -0.46  0.0562   46.26  -0.04   -4.49   176.64
5 RFMT                             27   51.14   -0.08  0.7509   46.02  -0.28   -5.12   175.91
ALL feature techniques             39   49.85   -1.37  0.0010   46.35  +0.05   -3.50   173.97
```

**All techniques combined: 49.85%, past the naive benchmark at 50.70%.**

#### The rankings invert

```
by competition metric (MAE% + |Bias%|)     by MAE% alone
  -1.01  p=0.0070  Target encoding           -0.28  RFMT
  -0.46  p=0.0562  Value proposition         -0.06  Velocity classification
  -0.08  p=0.7509  RFMT                      -0.04  Value proposition
  +0.02  p=0.8997  Velocity classification   +0.51  Target encoding
```

Target encoding is first on the metric and last on accuracy. RFMT is the
reverse. The inversion is significant in **both** directions: target encoding
improves the composite at p=0.0070 and degrades MAE% at p=0.0126, while RFMT
improves MAE% at p=0.0269 and does nothing to the composite (p=0.7509). The baseline under-forecasts by 4.92%, so `|Bias%|` is a large share
of the composite score and anything that lifts forecasts scores well whether
or not it predicts better. Report both columns or the comparison is not honest.

### Case 1: sampling

With the full training set as the reference, sampling is not trying to beat it.
The question is how close you get and how much fit time you save.

First: does this dataset even have the imbalance the technique targets?
Strata are warehouse x velocity tier — 1,486 of them, median 18 rows.

```
strata a random draw reaches, mean of 8 draws (of 1,486)
  n= 2,000    926 never drawn   1,371 under 5 rows
  n= 5,000    684 never drawn   1,214 under 5 rows
  n=10,000    509 never drawn   1,070 under 5 rows
  n=20,000    334 never drawn     894 under 5 rows
  n=50,000    137 never drawn     610 under 5 rows
```

A random draw of 5,000 rows never touches 684 of the 1,486 strata, and leaves
another 530 with fewer than five. Even at 50,000 rows, 137 are still missing
entirely — the uneven grouping the technique targets is genuinely present
here.

```
 budget    strategy    rows   score  vs full     slow   fit_s  speedup
  2,000      random    2000   53.33    +2.27   174.53    0.29    14.2x
  2,000  stratified    2362   54.78    +3.72   170.24    0.42     9.8x
  5,000      random    5000   53.83    +2.78   174.19    0.75     5.6x
  5,000  stratified    5097   52.57    +1.52   172.69    0.87     4.7x
 10,000      random   10000   54.22    +3.16   175.67    1.04     4.0x
 10,000  stratified    9949   52.43    +1.38   173.03    0.92     4.5x
 20,000      random   20000   53.21    +2.16   176.48    1.26     3.3x
 20,000  stratified   19812   52.95    +1.90   173.82    1.28     3.2x
 50,000      random   50000   52.39    +1.34   176.57    2.15     1.9x
 50,000  stratified   45817   51.73    +0.67   175.60    1.92     2.2x
                                       full:   176.69    4.14
```

```
stratified vs random at the same budget, paired by seed
  n= 2,000   score: +1.45 sd=4.42 p=0.3854    slow: -4.29 sd=4.60 p=0.0337 *
  n= 5,000   score: -1.26 sd=4.51 p=0.4553    slow: -1.50 sd=2.58 p=0.1447
  n=10,000   score: -1.79 sd=3.63 p=0.2067    slow: -2.64 sd=1.63 p=0.0025 *
  n=20,000   score: -0.26 sd=2.58 p=0.7854    slow: -2.66 sd=1.74 p=0.0035 *
  n=50,000   score: -0.66 sd=1.62 p=0.2856    slow: -0.97 sd=1.29 p=0.0717
```

Three findings:

1. **Speed is a straight win.** A stratified 10,000-row sample — 7% of the data
   — lands about 1.4 points off the full-data score at roughly 4x the fit
   speed. Fit times vary with machine load; the ratio is the stable part.
2. **Stratified beats random on the long tail** at four of five budgets, and
   **the aggregate score is never significant** at any budget (p = 0.21 to
   0.79). Watch only the headline number and this lever looks inert.
3. **Stratified sampling beats training on all the data for slow movers, at
   every budget.** Full data scores 176.69 on the bottom two velocity tiers;
   stratified at 2,000 rows scores 170.24, on 1.7% of the data. Convergence is
   monotone as the budget grows — 170.24, 172.69, 173.03, 173.82, 175.60 —
   approaching the full-data value from below. The volume-weighted loss drowns
   out the tail; up-weighting it by construction beats having more data about
   items the loss ignores.

#### Caveat: the allocation exponent

`quota ~ count ** power`, swept at a 5,000-row budget against the random draw:

```
 power    rows   score   delta        p      slow   delta        p      bias
   0.0    4172   54.02   +0.19  0.9084   168.17   -6.02  0.0004 *   -2.75
   0.3    4951   54.23   +0.39  0.7805   171.45   -2.74  0.0149 *   -4.80
   0.5    5097   52.57   -1.26  0.4553   172.69   -1.50  0.1447     -3.01
   0.7    5253   52.81   -1.02  0.5783   173.25   -0.94  0.4734     -3.95
   1.0    5570   52.75   -1.08  0.5617   173.08   -1.11  0.3903     -3.81
```

**Uniform allocation was the best setting for the tail, not the trap** the
usual advice describes. The gain rises monotonically as allocation flattens. It
does push bias upward exactly as theory predicts (-3.81 proportional ->
-2.75 uniform), but this baseline under-forecasts, so over-forecasting the tail
happens to help. On a calibrated baseline it would hurt — treat this as a
reason to sweep the exponent, not a recommendation to set it to zero.

**The floor matters more than the exponent.** Proportional allocation
(power=1.0) should equal random sampling and does not: it draws 5,570 rows, not
5,000, and scores -1.11 on the tail. The cause is `quota.clip(lower=1)`, which
lifts 653 of the 1,486 strata from a rounded quota of zero up to one — 14.6% of
the budget departing from proportional. Guarantee every stratum a row before
tuning anything else.

Note also that the realised sample size is a function of the exponent: uniform
allocation draws only 4,172 of a 5,000 budget, because small strata run out of
rows.

#### Caveat: on a balanced panel this does nothing

Before VN1 I ran the same experiment on the M5 Walmart dataset, aggregated to
one row per item-store-year. It produced nothing — including one result that
was significant in the *wrong* direction. M5 is a balanced panel: every item x
store x period exists, which gives 69 evenly-sized strata (median 325 rows), so
a random 6,000-row draw already covered 67 of them. Those runs are not in this
repo; the figures are quoted for contrast.

```
M5, aggregate wSMAPE       +0.42   p=0.008   (significant, wrong direction)
M5, macro wSMAPE           +0.10   p=0.635
M5, 8 budget x regime cells: 1 of 8 significant
    cold-start n=300       -1.15   p=0.004
    the other seven        p = 0.23 to 0.80
Exponents 0.0/0.3/0.5/0.7 across budgets 1,000-8,000: random equal or better
```

Stratified sampling pays when groups within your categories are unevenly
represented. When categories are evenly represented it does nothing, and no
exponent tuning or change of grain fixes that.

### Combining only the significant techniques

Target encoding is significant on both metrics; RFMT on accuracy. Stacking
just those two should give the bias fix and the accuracy gain. It does not.

```
                  score    MAE%    Bias%    slow   d_score   d_MAE
baseline          51.22   46.30   -4.92   176.87    0.000   0.000
encoding only     50.21   46.81   -3.40   174.06   -1.008  +0.512
RFMT only         51.14   46.02   -5.12   175.91   -0.082  -0.279
encoding + RFMT   50.35   46.33   -4.01   173.78   -0.875  +0.034
all four          49.85   46.35   -3.50   173.97   -1.373  +0.049

vs baseline:   encoding + RFMT   score p=0.0231 *   MAE% p=0.8248
vs all four:   encoding + RFMT   score +0.498 p=0.1461   MAE% -0.016 p=0.9055
```

RFMT's accuracy gain (-0.279 alone, p=0.0269) vanishes in combination (+0.034,
p=0.82): target encoding's accuracy penalty cancels it. The interaction is
mildly favourable — addition predicts +0.233, measured is +0.034 — but nowhere
near enough to reach the baseline. Dropping the two insignificant levers also
costs composite performance (-0.875 vs -1.373), though not significantly.

Selecting levers by single-lever significance does not work here, because the
levers interact. Reproduce with `vn1_combo.py`.

### Case 3 caveat: the encoding source, and mean reversion

The article's Part 1 claims the asymmetry is the trick — encode from the full
population, apply to the sample. Tested on the same 5,000 rows with only the
source changed, 8 seeds:

```
                 score    mae    bias    slow
full population  52.21  48.76   -2.82  171.94
sample only      52.94  48.65   -3.81  173.21

difference: -0.73  sd=2.22  p=0.3801   NOT significant
```

**Directionally right, not significant.** An earlier single-seed run made this
look like a 3-point effect; across 8 seeds it is noise. One seed is not a
result.

The caveat that *is* significant came from the M5 runs mentioned above, where
the model forecast year-on-year *growth* rather than a level:

```
product-level growth, year over year:   corr = -0.236   (mean-reverts)
product-level volume, year over year:   corr = +0.882   (persists)
```

Growth reverses at item level, while the level itself persists. A
full-population encoding is sharper, so the model trusts it more and is hurt more when the
pattern inverts: encoding from the full population there was **8.2 points worse
at p < 0.0001**. The sample-only version was accidentally protected by its own
noise.

> Encode from the full population when the quantity persists. Don't when it
> reverts. Group by the categorical, compute the quantity in two consecutive
> periods, correlate them. Positive means encode.

### Case 4 caveat: it needs data volume

On a 5,000-row sample the same ten relative-price features made things **1.20
points worse**; on the full 135,477 rows they help by 0.46. Weak individual
features need volume to separate from noise — which cuts against Case 1, and
is worth stating as a trade.

Note also what this dataset can express: VN1 is anonymised and single-sector,
with no brand, no rating and no review count, so price position was the only
value signal available. This tests relative price, not value positioning.

## Model standing

```
46.37%  1st: Figura & Stubbs
48.08%  5th: An Hoang
49.85%  all feature techniques, full data
50.70%  Naive forecast
51.22%  baseline, volume + trend only
80.46%  12-week moving average
```

Caveat: forecasting a 13-week total spread flat is a
*simplified* target. Competitors shaped weekly profiles. The scoring is
verified correct against both published benchmarks, but this is not a
like-for-like leaderboard entry and should not be read as one.

## Licence

Code: MIT, see `LICENSE`.

Data: not redistributed here. VN1 Forecasting – Accuracy Challenge (Vandeput / SupChains, 2024), data by
Flieber, CC-BY-4.0, DOI 10.5281/zenodo.23265103 —
https://github.com/SupChains/VN1 · https://supchains.com/vn1-forecasting-competition/

If you use the data, cite the competition — the VN1 repo ships a `CITATION.cff`
with the reference.
