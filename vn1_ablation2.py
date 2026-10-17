"""Ablation: each technique added to a model trained on ALL the data.

The baseline is the full 135,477-row training set with volume-and-trend
features only -- what a first-pass forecasting model usually has. Each
technique is then switched on alone, on the same rows, so every delta is
attributable to one thing and nothing is confounded by which rows were drawn.

Sampling is measured separately, in vn1_sampling2.py, because it is the one
technique that changes the rows rather than the features.

Five seeds per configuration, paired, because LightGBM subsamples.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

from build_vn1_features import FEATURE_GROUPS
from vn1_lgbm import (DATA, LEADERBOARD, benchmarks, fit_predict, load_frames,
                      official_score, ratio_target, segment_score)

SEEDS = range(5)
SMOOTH_K = 50.0


def add_warehouse_encoding(frame, test, source, seed=0):
    """Smoothed, out-of-fold Warehouse encoding of the model's target.

    Guardrails that are not optional:
      - out-of-fold on the training rows, or the model reads its own answers
      - smoothed toward the global mean, so a warehouse with four rows does
        not earn a confident encoding
      - built only from rows at or before the forecast origin
    """
    y_src, _ = ratio_target(source)
    src = source[["Warehouse"]].assign(_y=y_src)
    prior = float(y_src.mean())

    def smoothed(f):
        g = f.groupby("Warehouse")["_y"].agg(["mean", "count"])
        return (g["mean"] * g["count"] + prior * SMOOTH_K) / (g["count"] + SMOOTH_K)

    enc = np.full(len(frame), np.nan)
    for _, hold in KFold(5, shuffle=True, random_state=seed).split(frame):
        keep = src.drop(index=frame.index[hold], errors="ignore")
        enc[hold] = (frame.iloc[hold]["Warehouse"]
                     .map(smoothed(keep)).fillna(prior).to_numpy())
    return enc, test["Warehouse"].map(smoothed(src)).fillna(prior).to_numpy()


def run(train, test, phase2, tier, groups, seed):
    tr, te = train, test
    if "encoding" in groups:
        tr, te = train.copy(), test.copy()
        a, b = add_warehouse_encoding(tr, te, train, seed=seed)
        tr["wh_enc"], te["wh_enc"] = a, b

    feats = [c for g in groups for c in FEATURE_GROUPS[g]]
    pred, secs = fit_predict(tr, te, feats, seed=seed, timed=True)
    score, mae, bias = official_score(pred, te, phase2)
    return dict(score=score, mae=mae, bias=bias, fit_s=secs,
                slow=segment_score(pred, te, phase2, tier <= 1),
                fast=segment_score(pred, te, phase2, tier >= 5),
                n_feats=len(feats))


def main():
    train, test, phase2 = load_frames()
    tier = test["velocity"].to_numpy()
    naive, ma12 = benchmarks(test, phase2)

    print(f"baseline rows {len(train):,} | test {len(test):,} series")
    print(f"wiring check: naive {naive:.2f}% (published 50.70), "
          f"MA12 {ma12:.2f}% (published 80.46)\n")

    configs = {
        "0 Baseline (volume + trend only)": ["base"],
        "2 Velocity classification": ["base", "velocity"],
        "3 Target encoding (Warehouse)": ["base", "encoding"],
        "4 Value proposition": ["base", "value"],
        "5 RFMT": ["base", "rfmt"],
        "ALL feature techniques": ["base", "velocity", "encoding", "value", "rfmt"],
    }

    rows = []
    for name, groups in configs.items():
        for seed in SEEDS:
            rows.append(dict(technique=name, seed=seed,
                             **run(train, test, phase2, tier, groups, seed)))
    res = pd.DataFrame(rows)
    res.to_csv(DATA / "vn1_ablation2_raw.csv", index=False)

    base = res[res.technique == "0 Baseline (volume + trend only)"].set_index("seed")
    agg = res.groupby("technique").agg(
        n_feats=("n_feats", "first"), score=("score", "mean"),
        score_sd=("score", "std"), mae=("mae", "mean"),
        bias=("bias", "mean"), slow=("slow", "mean"), fit_s=("fit_s", "mean"))
    agg["delta"] = agg["score"] - base["score"].mean()
    agg["d_mae"] = agg["mae"] - base["mae"].mean()

    # Paired against the baseline, seed by seed.
    pvals = {}
    for name in configs:
        g = res[res.technique == name].set_index("seed")
        if name.startswith("0"):
            pvals[name] = np.nan
            continue
        _, pv = stats.ttest_rel(g["score"], base["score"])
        pvals[name] = pv
    agg["p"] = pd.Series(pvals)

    print(f"mean of {len(SEEDS)} seeds, paired against the baseline\n")
    print(agg[["n_feats", "score", "score_sd", "delta", "p",
               "mae", "d_mae", "bias", "slow", "fit_s"]].round(3).to_string())

    print(f"\nnote: baseline bias is {base['bias'].mean():+.2f}%, so |Bias%| is a "
          "large share of\nthe composite score. Read d_mae for accuracy and "
          "delta for the competition metric.")

    single = agg.drop(index=["0 Baseline (volume + trend only)",
                             "ALL feature techniques"])
    print(f"\n{'=' * 62}")
    print("by competition metric:")
    for n, r in single.sort_values("delta").iterrows():
        print(f"  {r['delta']:+6.2f}  p={r['p']:.4f}  {n}")
    print("\nby MAE% alone:")
    for n, r in single.sort_values("d_mae").iterrows():
        print(f"  {r['d_mae']:+6.2f}  {n}")

    best = agg.loc["ALL feature techniques"]
    print(f"\nbaseline {base['score'].mean():.2f}%  ->  "
          f"all techniques {best['score']:.2f}%  ({best['delta']:+.2f})")
    print(f"vs naive {naive:.2f}%, best competitor 46.37%")
    print("=" * 62)


if __name__ == "__main__":
    main()
