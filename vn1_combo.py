"""Does combining only the SIGNIFICANT techniques beat combining all four?

Target encoding is significant on both metrics (composite p=0.0070,
MAE% p=0.0126). RFMT is significant on accuracy (p=0.0269). Velocity and value
proposition clear neither bar.

All four together improved the composite but left MAE% unchanged — the whole
gain was bias correction. This asks whether dropping the two insignificant
levers recovers the accuracy that RFMT contributes on its own.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from build_vn1_features import FEATURE_GROUPS
from vn1_ablation2 import add_warehouse_encoding
from vn1_lgbm import (DATA, benchmarks, fit_predict, load_frames,
                      official_score, segment_score)

warnings.filterwarnings("ignore")

SEEDS = range(5)
CONFIGS = {
    "baseline": ["base"],
    "encoding only": ["base", "encoding"],
    "RFMT only": ["base", "rfmt"],
    "encoding + RFMT": ["base", "encoding", "rfmt"],
    "all four": ["base", "velocity", "encoding", "value", "rfmt"],
}


def run(train, test, phase2, tier, groups, seed):
    tr, te = train, test
    if "encoding" in groups:
        tr, te = train.copy(), test.copy()
        a, b = add_warehouse_encoding(tr, te, train, seed=seed)
        tr["wh_enc"], te["wh_enc"] = a, b
    feats = [c for g in groups for c in FEATURE_GROUPS[g]]
    pred = fit_predict(tr, te, feats, seed=seed)
    score, mae, bias = official_score(pred, te, phase2)
    return dict(score=score, mae=mae, bias=bias,
                slow=segment_score(pred, te, phase2, tier <= 1),
                n_feats=len(feats))


def main():
    train, test, phase2 = load_frames()
    tier = test["velocity"].to_numpy()
    naive, _ = benchmarks(test, phase2)

    rows = []
    for name, groups in CONFIGS.items():
        for seed in SEEDS:
            rows.append(dict(config=name, seed=seed,
                             **run(train, test, phase2, tier, groups, seed)))
    res = pd.DataFrame(rows)
    res.to_csv(DATA / "vn1_combo_raw.csv", index=False)

    agg = (res.groupby("config")
              .agg(n_feats=("n_feats", "first"), score=("score", "mean"),
                   score_sd=("score", "std"), mae=("mae", "mean"),
                   mae_sd=("mae", "std"), bias=("bias", "mean"),
                   slow=("slow", "mean"))
              .reindex(CONFIGS))
    base = res[res.config == "baseline"].set_index("seed")
    agg["d_score"] = agg["score"] - agg.loc["baseline", "score"]
    agg["d_mae"] = agg["mae"] - agg.loc["baseline", "mae"]

    print(f"{len(SEEDS)} seeds | naive {naive:.2f}%\n")
    print(agg.round(3).to_string())

    print("\npaired vs baseline")
    for name in CONFIGS:
        if name == "baseline":
            continue
        g = res[res.config == name].set_index("seed")
        ps = stats.ttest_rel(g["score"], base["score"]).pvalue
        pm = stats.ttest_rel(g["mae"], base["mae"]).pvalue
        print(f"  {name:18s} score p={ps:.4f}{'*' if ps < .05 else ' '}"
              f"   mae p={pm:.4f}{'*' if pm < .05 else ' '}")

    print("\npaired: encoding + RFMT  vs  all four")
    a = res[res.config == "encoding + RFMT"].set_index("seed")
    b = res[res.config == "all four"].set_index("seed")
    for m in ("score", "mae"):
        d = a[m] - b[m]
        pv = stats.ttest_rel(a[m], b[m]).pvalue
        print(f"  {m:5s} delta={d.mean():+.3f} sd={d.std():.3f} p={pv:.4f}"
              f"{'  SIGNIFICANT' if pv < .05 else ''}")


if __name__ == "__main__":
    main()
