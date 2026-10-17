"""Case 1: sampling, measured against training on ALL the data.

With the full training set as the reference, sampling is not trying to beat
it. The question is how close you get, and how much training time you save --
and whether stratifying beats a random draw at the same budget.

Eight paired seeds per cell. Timing is model fit only; feature building is
identical in every arm.
"""

import numpy as np
import pandas as pd
from scipy import stats

from build_vn1_features import FEATURE_GROUPS
from vn1_lgbm import (DATA, benchmarks, fit_predict, load_frames,
                      official_score, segment_score)

BUDGETS = [2_000, 5_000, 10_000, 20_000, 50_000]
SEEDS = range(8)
STRATUM_KEYS = ["Warehouse", "velocity"]
FEATS = FEATURE_GROUPS["base"]


def random_sample(df, n, seed):
    return df.sample(n=min(n, len(df)), random_state=seed)


def stratified_sample(df, n, seed, power=0.5):
    """Allocate quota across Warehouse x velocity strata by count ** power.

    The clip(lower=1) floor is not a detail. At power=1.0 -- nominally
    proportional allocation -- it lifts 653 of the 1,486 strata from a rounded
    quota of zero up to one, so 14.6% of the budget departs from proportional.
    That floor, not the exponent, does most of the work.

    Uniform allocation (power=0.0) cannot always spend the budget: small
    strata run out of rows, so the realised sample comes in under target.
    """
    df = df.copy()
    df["stratum"] = df[STRATUM_KEYS].astype(str).agg("|".join, axis=1)
    w = df.groupby("stratum").size().astype(float) ** power
    quota = (w / w.sum() * n).round().astype(int).clip(lower=1)
    rng = np.random.default_rng(seed)
    parts = [g.sample(n=min(len(g), quota[k]),
                      random_state=int(rng.integers(1 << 32)))
             for k, g in df.groupby("stratum")]
    return pd.concat(parts).drop(columns="stratum")


def main():
    train, test, phase2 = load_frames()
    tier = test["velocity"].to_numpy()
    naive, _ = benchmarks(test, phase2)

    # The reference: train on everything.
    full, full_s = fit_predict(train, test, FEATS, seed=0, timed=True)
    f_score, f_mae, f_bias = official_score(full, test, phase2)
    f_slow = segment_score(full, test, phase2, tier <= 1)

    strata = train[STRATUM_KEYS].astype(str).agg("|".join, axis=1)
    sizes = strata.value_counts()
    print(f"full training set: {len(train):,} rows | {len(test):,} test series")
    print(f"  score {f_score:.2f}%  MAE% {f_mae:.2f}  Bias% {f_bias:+.2f}  "
          f"slow {f_slow:.2f}  fit {full_s:.2f}s")
    print(f"  naive benchmark {naive:.2f}%")
    print(f"\n{len(sizes):,} strata (Warehouse x velocity), "
          f"median {int(sizes.median())} rows, smallest {sizes.min()}")
    for b in BUDGETS:
        d = strata[random_sample(train, b, 0).index].value_counts() \
            .reindex(sizes.index).fillna(0)
        print(f"  a random draw of {b:6,} leaves {int((d == 0).sum()):4d} "
              f"strata empty, {int((d < 5).sum()):4d} under 5 rows")

    rows = []
    for budget in BUDGETS:
        for seed in SEEDS:
            for name, fn in [("random", random_sample),
                             ("stratified", stratified_sample)]:
                s = fn(train, budget, seed)
                pred, secs = fit_predict(s, test, FEATS, seed=0, timed=True)
                score, mae, bias = official_score(pred, test, phase2)
                rows.append(dict(budget=budget, seed=seed, strategy=name,
                                 n_rows=len(s), score=score, mae=mae,
                                 bias=bias, fit_s=secs,
                                 slow=segment_score(pred, test, phase2, tier <= 1)))

    res = pd.DataFrame(rows)
    res.to_csv(DATA / "vn1_sampling2_raw.csv", index=False)

    print(f"\n\nmean of {len(SEEDS)} seeds. 'vs full' is the gap to training "
          f"on all {len(train):,} rows.\n")
    hdr = (f"{'budget':>7} {'strategy':>11} {'rows':>7} {'score':>7} "
           f"{'vs full':>8} {'slow':>8} {'fit_s':>7} {'speedup':>8}")
    print(hdr)
    print("-" * len(hdr))
    for budget in BUDGETS:
        for name in ("random", "stratified"):
            g = res[(res.budget == budget) & (res.strategy == name)]
            print(f"{budget:7,} {name:>11} {g.n_rows.mean():7.0f} "
                  f"{g.score.mean():7.2f} {g.score.mean() - f_score:+8.2f} "
                  f"{g.slow.mean():8.2f} {g.fit_s.mean():7.2f} "
                  f"{full_s / g.fit_s.mean():7.1f}x")
        print("-" * len(hdr))

    print("\nstratified vs random at the same budget, paired by seed")
    for budget in BUDGETS:
        g = res[res.budget == budget]
        out = f"  n={budget:6,}"
        for metric in ("score", "slow"):
            p_ = g.pivot(index="seed", columns="strategy", values=metric)
            d = p_["stratified"] - p_["random"]
            _, pv = stats.ttest_rel(p_["stratified"], p_["random"])
            out += (f"   {metric}: {d.mean():+6.2f} sd={d.std():4.2f} "
                    f"p={pv:.4f}{'*' if pv < 0.05 else ' '}")
        print(out)
    print("\n* p < 0.05")


if __name__ == "__main__":
    main()
