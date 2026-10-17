"""The two dataset-specific caveats, both measured with LightGBM.

A. Case 1 -- the allocation exponent. quota ~ count ** power.
B. Case 3 -- where the target encoding is computed FROM, which only matters
   once you are training on a sample rather than the whole population.
"""

import numpy as np
import pandas as pd
from scipy import stats

from build_vn1_features import FEATURE_GROUPS
from vn1_ablation2 import add_warehouse_encoding
from vn1_lgbm import (DATA, fit_predict, load_frames, official_score,
                      segment_score)
from vn1_sampling2 import random_sample, stratified_sample

SEEDS = range(8)
BUDGET = 5_000
BASE = FEATURE_GROUPS["base"]
ENC = BASE + FEATURE_GROUPS["encoding"]


def part_a(train, test, phase2, tier):
    print("A. Case 1 caveat: the allocation exponent\n")
    ref = {}
    for seed in SEEDS:
        p = fit_predict(random_sample(train, BUDGET, seed), test, BASE)
        ref[seed] = (official_score(p, test, phase2)[0],
                     segment_score(p, test, phase2, tier <= 1))
    r = pd.DataFrame(ref, index=["score", "slow"]).T

    print(f"random reference at n={BUDGET:,}: score {r.score.mean():.2f}, "
          f"slow {r.slow.mean():.2f}\n")
    print(f"{'power':>6} {'rows':>7} {'score':>7} {'delta':>7} {'p':>8}   "
          f"{'slow':>7} {'delta':>7} {'p':>8}   {'bias':>7}")
    rows = []
    for power in [0.0, 0.3, 0.5, 0.7, 1.0]:
        vals = {}
        nrow = []
        for seed in SEEDS:
            s = stratified_sample(train, BUDGET, seed, power=power)
            nrow.append(len(s))
            p = fit_predict(s, test, BASE)
            sc, _, bias = official_score(p, test, phase2)
            vals[seed] = (sc, segment_score(p, test, phase2, tier <= 1), bias)
        g = pd.DataFrame(vals, index=["score", "slow", "bias"]).T
        line = f"{power:6.1f} {np.mean(nrow):7.0f}"
        for m in ("score", "slow"):
            d = g[m] - r[m]
            _, pv = stats.ttest_rel(g[m], r[m])
            line += (f" {g[m].mean():7.2f} {d.mean():+7.2f} {pv:7.4f}"
                     f"{'*' if pv < 0.05 else ' '}")
        line += f"   {g.bias.mean():+7.2f}"
        print(line)
        rows.append(dict(power=power, rows=np.mean(nrow), **g.mean().to_dict()))
    pd.DataFrame(rows).to_csv(DATA / "vn1_caveat_power.csv", index=False)
    print("\n* p < 0.05 vs the random draw, paired by seed.")


def part_b(train, test, phase2, tier):
    print("\n\nB. Case 3 caveat: where the encoding is computed from\n")
    print(f"both arms train on the SAME random {BUDGET:,} rows; only the "
          f"encoding source differs\n")
    rows = []
    for seed in SEEDS:
        s = random_sample(train, BUDGET, seed)
        for label, source in [("sample only", s), ("full population", train)]:
            smp, tst = s.copy(), test.copy()
            a, b = add_warehouse_encoding(smp, tst, source, seed=seed)
            smp["wh_enc"], tst["wh_enc"] = a, b
            p = fit_predict(smp, tst, ENC, seed=seed)
            sc, mae, bias = official_score(p, tst, phase2)
            rows.append(dict(seed=seed, source=label, score=sc, mae=mae,
                             bias=bias,
                             slow=segment_score(p, tst, phase2, tier <= 1)))
    res = pd.DataFrame(rows)
    res.to_csv(DATA / "vn1_caveat_encoding.csv", index=False)
    print(res.groupby("source")[["score", "mae", "bias", "slow"]]
             .agg(["mean", "std"]).round(2).to_string())

    piv = res.pivot(index="seed", columns="source", values="score")
    d = piv["full population"] - piv["sample only"]
    _, pv = stats.ttest_rel(piv["full population"], piv["sample only"])
    print(f"\nfull population minus sample only: {d.mean():+.2f} "
          f"sd={d.std():.2f} p={pv:.4f} "
          f"{'SIGNIFICANT' if pv < 0.05 else 'not significant'}")


def main():
    train, test, phase2 = load_frames()
    tier = test["velocity"].to_numpy()
    part_a(train, test, phase2, tier)
    part_b(train, test, phase2, tier)


if __name__ == "__main__":
    main()
