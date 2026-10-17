"""Shared model and scoring for the VN1 experiments. LightGBM.

The official metric needs 13 weekly cells per series. We forecast the 13-week
total and spread it flat, exactly as the two competition benchmarks do, then
score with the competition's own formula.
"""

import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO, DATA = HERE / "VN1", HERE / "data"
INDEX = ["Client", "Warehouse", "Product"]
HORIZON = 13

NON_FEATURES = INDEX + ["split", "origin", "units_13w", "velocity",
                        "stratum", "wh_enc"]

LEADERBOARD = [
    ("1st: Figura & Stubbs", 46.37), ("2nd: Furlotte", 46.57),
    ("3rd: Nikzad", 47.58), ("4th: Schwartz", 47.74),
    ("5th: An Hoang", 48.08), ("Naive forecast", 50.70),
    ("12-week moving average", 80.46),
]

LGB_PARAMS = dict(
    objective="regression_l1",   # MAE: the metric is MAE-based, and squared
                                 # error would let a handful of giant series
                                 # own every split
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=40,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    verbosity=-1,
    n_jobs=-1,
)


# ----------------------------------------------------------------- scoring

def load_phase2():
    t = pd.read_csv(REPO / "Phase 2 - Sales.csv").set_index(INDEX)
    t.columns = pd.to_datetime(t.columns)
    return t


def official_score(total_forecast, test, phase2):
    """Score a per-series 13-week total with the competition metric.

    Score = MAE% + |Bias%| over the 13 weekly cells. A perfect forecast is 0.
    """
    weekly = np.repeat(np.asarray(total_forecast, float)[:, None] / HORIZON,
                       HORIZON, axis=1)
    fc = pd.DataFrame(weekly, index=pd.MultiIndex.from_frame(test[INDEX]),
                      columns=phase2.columns).loc[phase2.index]
    err = fc.to_numpy() - phase2.to_numpy()
    total = np.nansum(phase2.to_numpy())
    mae = np.nansum(np.abs(err)) / total
    bias = np.nansum(err) / total
    return 100 * (mae + abs(bias)), 100 * mae, 100 * bias


def segment_score(total_forecast, test, phase2, mask):
    """The same metric restricted to a subset of series."""
    return official_score(np.asarray(total_forecast, float)[mask],
                          test[mask].reset_index(drop=True),
                          phase2[mask])[0]


# -------------------------------------------------------------- velocity

def add_velocity(df, edges=None, q=10):
    """Velocity tier from trailing 52-week units -- known at forecast time.

    Binning on the target would leak. Test rows use the train edges so a tier
    means the same thing in both.
    """
    if edges is None:
        edges = np.unique(np.quantile(df["sum_last_52w"], np.linspace(0, 1, q + 1)))
    out = df.copy()
    v = pd.cut(out["sum_last_52w"], bins=edges, labels=False, include_lowest=True)
    out["velocity"] = v.fillna(len(edges) - 2).astype(int)
    return out, edges


# ------------------------------------------------------------------- model

def ratio_target(df):
    """Log ratio of the next 13 weeks to the last 13, plus the baseline.

    Forecasting the level directly loses to a naive carry-forward: the model
    burns capacity re-deriving a baseline it was handed.
    """
    base = df["sum_last_13w"].to_numpy()
    return np.log1p(df["units_13w"]) - np.log1p(base), base


def fit_predict(sample, test, feats, seed=0, timed=False):
    """Fit on `sample` with exactly `feats`; predict the 13-week total."""
    y, _ = ratio_target(sample)

    t0 = time.perf_counter()
    model = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
    model.fit(sample[feats], y)
    fit_s = time.perf_counter() - t0

    base_te = test["sum_last_13w"].to_numpy()
    pred = np.clip(np.expm1(np.log1p(base_te) + model.predict(test[feats])), 0, None)
    return (pred, fit_s) if timed else pred


def load_frames():
    train, edges = add_velocity(pd.read_parquet(DATA / "vn1_train.parquet"))
    test, _ = add_velocity(pd.read_parquet(DATA / "vn1_test.parquet"), edges)
    return train, test, load_phase2()


def benchmarks(test, phase2):
    """Reproduce the two published benchmarks -- the data-wiring check."""
    s0 = pd.read_csv(REPO / "Phase 0 - Sales.csv").set_index(INDEX)
    s1 = pd.read_csv(REPO / "Phase 1 - Sales.csv").set_index(INDEX)
    hist = pd.concat([s0, s1], axis=1).loc[phase2.index]
    order = pd.MultiIndex.from_frame(test[INDEX])
    naive = (hist.iloc[:, -1] * HORIZON).reindex(order).to_numpy()
    ma12 = (hist.iloc[:, -12:].mean(axis=1) * HORIZON).reindex(order).to_numpy()
    return (official_score(naive, test, phase2)[0],
            official_score(ma12, test, phase2)[0])


if __name__ == "__main__":
    from build_vn1_features import FEATURE_GROUPS

    train, test, phase2 = load_frames()
    n, m = benchmarks(test, phase2)
    print(f"naive {n:.2f}% (published 50.70) | MA12 {m:.2f}% (published 80.46)\n")

    pred, secs = fit_predict(train, test, FEATURE_GROUPS["base"], timed=True)
    s, mae, bias = official_score(pred, test, phase2)
    print(f"LightGBM, full {len(train):,} rows, base features only")
    print(f"  score {s:.2f}%  MAE% {mae:.2f}  Bias% {bias:+.2f}  fit {secs:.1f}s")

    allf = [c for g in FEATURE_GROUPS for c in FEATURE_GROUPS[g]
            if g not in ("encoding",)]
    pred, secs = fit_predict(train, test, allf, timed=True)
    s, mae, bias = official_score(pred, test, phase2)
    print(f"\nall features except encoding ({len(allf)})")
    print(f"  score {s:.2f}%  MAE% {mae:.2f}  Bias% {bias:+.2f}  fit {secs:.1f}s")
