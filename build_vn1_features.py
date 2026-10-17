"""Build a 13-week-ahead forecasting frame from the VN1 competition data.

VN1 (Vandeput / SupChains, 2024, CC-BY-4.0): 15,053 product-warehouse series
from 46 e-vendors, weekly units. https://github.com/SupChains/VN1

Task, mirroring the competition
-------------------------------
Phase 0 gives 170 weeks of history, Phase 1 the next 13, Phase 2 the 13 weeks
of the final ranking. Competitors forecast Phase 2 from Phase 0 + Phase 1.

This builds one row per (series, forecast origin). The target is total units
over the 13 weeks starting at the origin. Every feature comes from weeks
strictly before the origin.

    test   origin = first Phase 2 week   (features: all 183 prior weeks)
    train  origins inside Phase 0, spaced 13 weeks so targets never overlap

Grain: one row per (Client, Warehouse, Product, origin).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# A series with no sales, or no price, in a window yields an all-NaN slice.
# That is the correct answer -- the tree model reads NaN natively as "no
# history" -- so the warnings numpy raises about it are noise.
warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO = Path(__file__).resolve().parent / "VN1"
OUT = Path(__file__).resolve().parent / "data"

INDEX = ["Client", "Warehouse", "Product"]
HORIZON = 13          # weeks per forecast, as in the competition
MIN_HISTORY = 52      # weeks of history an origin needs before it is usable

# The ablation switches these groups on one at a time, so each article section
# maps to exactly one group. BASE is the stripped-down starting point: volume
# and trend only, which is what a first-pass forecasting model usually has.
FEATURE_GROUPS = {
    "base": [
        "sum_last_4w", "mean_last_4w", "sum_last_13w", "mean_last_13w",
        "sum_last_26w", "mean_last_26w", "sum_last_52w", "mean_last_52w",
        "last_week", "std_last_13w", "max_last_52w",
        "prior_13w", "trend_13w", "same_block_last_year", "yoy_ratio",
        "origin_week_of_year",
    ],
    # Case 2 -- turns a Pareto target into tiers the model can condition on.
    "velocity": ["velocity"],
    # Case 3 -- high-cardinality Warehouse, encoded from the full population.
    "encoding": ["wh_enc"],
    # Case 4 -- absolute price says little across anonymised vendors; where an
    # item sits inside its assortment says a lot.
    "value": ["price_13w", "price_prev_13w", "price_change",
              "price_min_52w", "price_max_52w", "discount_depth",
              "price_rank_in_wh", "price_vs_wh_median",
              "price_rank_in_cl", "price_vs_cl_median"],
    # Case 5 -- recency, frequency, monetary, tenure. The group that tells the
    # model why a zero is a zero.
    "rfmt": ["weeks_since_last_sale",                      # recency
             "sale_weeks_52w", "sale_week_frac_52w",        # frequency
             "monetary_52w", "revenue_per_week_52w",        # monetary
             "weeks_since_first_sale", "active_span",
             "maturity", "ever_sold",                       # tenure
             "avg_units_per_sale_week", "demand_cv2"],      # sale shape
}


# --------------------------------------------------------------------- load

def load_panel():
    """Sales and prices as (series x week) matrices, phases concatenated.

    Returns ids, sales, price, weeks, n_phase0 -- where sales covers
    Phase 0 + 1 + 2 and price covers Phase 0 + 1 only (Phase 2 prices were
    never released, which is also why price can never be a target-week
    feature here).
    """
    def read(name):
        t = pd.read_csv(REPO / name).set_index(INDEX)
        t.columns = pd.to_datetime(t.columns)
        return t

    s0, s1, s2 = (read(f"Phase {p} - Sales.csv") for p in (0, 1, 2))
    p0, p1 = (read(f"Phase {p} - Price.csv") for p in (0, 1))

    # Every file lists the same 15,053 rows in the same order; align anyway.
    s1, s2 = s1.loc[s0.index], s2.loc[s0.index]
    p1 = p1.loc[s0.index]

    sales = pd.concat([s0, s1, s2], axis=1)
    price = pd.concat([p0, p1], axis=1)

    ids = s0.index.to_frame(index=False)
    return (ids,
            sales.to_numpy(dtype=np.float64),
            price.to_numpy(dtype=np.float64),
            list(sales.columns),
            s0.shape[1])


# ----------------------------------------------------------------- features

def _window(mat, origin, length):
    """The `length` weeks immediately before `origin`."""
    return mat[:, max(0, origin - length):origin]


def sales_features(sales, origin):
    """Volume, trend, intermittency and lifecycle, all from before `origin`."""
    hist = sales[:, :origin]
    f = {}

    # --- volume at several scales ---
    for w in (4, 13, 26, 52):
        blk = _window(sales, origin, w)
        f[f"sum_last_{w}w"] = blk.sum(axis=1)
        f[f"mean_last_{w}w"] = blk.mean(axis=1)
    f["last_week"] = sales[:, origin - 1]
    f["std_last_13w"] = _window(sales, origin, 13).std(axis=1)
    f["max_last_52w"] = _window(sales, origin, 52).max(axis=1)

    # --- trend: this 13-week block against the one before it ---
    recent = _window(sales, origin, 13).sum(axis=1)
    prior = sales[:, max(0, origin - 26):max(0, origin - 13)].sum(axis=1)
    f["prior_13w"] = prior
    f["trend_13w"] = (recent + 1.0) / (prior + 1.0)
    yr_ago = sales[:, max(0, origin - 52):max(0, origin - 39)].sum(axis=1)
    f["same_block_last_year"] = yr_ago
    f["yoy_ratio"] = (recent + 1.0) / (yr_ago + 1.0)

    # --- intermittency: how sparse is demand when it happens ---
    blk = _window(sales, origin, 52)
    nz = blk > 0
    n_sale = nz.sum(axis=1)
    f["sale_weeks_52w"] = n_sale
    f["sale_week_frac_52w"] = n_sale / blk.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        size = np.where(nz, blk, np.nan)
        mean_size = np.nanmean(size, axis=1)
        std_size = np.nanstd(size, axis=1)
        f["avg_units_per_sale_week"] = mean_size
        # Squared coefficient of variation -- the standard Syntetos-Boylan
        # axis for separating smooth, erratic, intermittent and lumpy demand.
        f["demand_cv2"] = (std_size / mean_size) ** 2

    # --- lifecycle: the reason a zero is a zero ---
    ever = hist > 0
    has = ever.any(axis=1)
    T = hist.shape[1]
    first = np.where(has, np.argmax(ever, axis=1), T)
    last = np.where(has, T - 1 - np.argmax(ever[:, ::-1], axis=1), -1)
    f["weeks_since_first_sale"] = np.where(has, T - first, 0)
    f["weeks_since_last_sale"] = np.where(has, T - 1 - last, T)
    f["active_span"] = np.where(has, last - first + 1, 0)
    f["ever_sold"] = has.astype(int)
    # A long-established series and a three-week-old one both sell 40 units a
    # week; only this ratio tells the model which is which.
    f["maturity"] = f["weeks_since_first_sale"] / T

    return pd.DataFrame(f)


def monetary_features(price, sales, origin):
    """Revenue contribution -- the M in RFMT.

    Units alone cannot separate a cheap fast mover from an expensive slow one,
    and the assortment decisions a forecast feeds care about the second.
    """
    o = min(origin, price.shape[1])
    pw = price[:, max(0, o - 52):o]
    sw = sales[:, max(0, origin - 52):origin][:, -pw.shape[1]:] if pw.size else pw

    with np.errstate(invalid="ignore"):
        rev = np.nansum(np.where(np.isnan(pw), 0.0, pw) * sw, axis=1)
    weeks = max(pw.shape[1], 1)
    return pd.DataFrame({
        "monetary_52w": rev,
        "revenue_per_week_52w": rev / weeks,
    })


def price_features(price, sales, origin):
    """Price level, movement and position. Empty price = nothing sold."""
    f = {}
    n_price_weeks = price.shape[1]
    o = min(origin, n_price_weeks)

    blk = price[:, max(0, o - 13):o]
    prev = price[:, max(0, o - 26):max(0, o - 13)]
    with np.errstate(invalid="ignore"):
        cur = np.nanmean(blk, axis=1) if blk.size else np.full(len(price), np.nan)
        old = np.nanmean(prev, axis=1) if prev.size else np.full(len(price), np.nan)
    f["price_13w"] = cur
    f["price_prev_13w"] = old
    f["price_change"] = cur / old - 1

    yr = price[:, max(0, o - 52):o]
    with np.errstate(invalid="ignore"):
        f["price_min_52w"] = np.nanmin(yr, axis=1) if yr.size else np.nan
        f["price_max_52w"] = np.nanmax(yr, axis=1) if yr.size else np.nan
    f["discount_depth"] = 1 - f["price_min_52w"] / f["price_max_52w"]
    return pd.DataFrame(f)


def relative_price(df):
    """Price rank within warehouse and within client.

    Absolute price is meaningless across anonymized vendors; position inside
    the assortment is the part a model can use. This is the Case 4
    "value proposition" signal in its cheapest form.
    """
    df = df.copy()
    for key, name in [("Warehouse", "wh"), ("Client", "cl")]:
        g = df.groupby(key)["price_13w"]
        df[f"price_rank_in_{name}"] = g.rank(pct=True)
        df[f"price_vs_{name}_median"] = df["price_13w"] / g.transform("median")
    return df


# ------------------------------------------------------------------- frame

def build_origin(ids, sales, price, weeks, origin, split):
    """One block of rows: every series, forecast from a single origin."""
    target = sales[:, origin:origin + HORIZON].sum(axis=1)

    df = pd.concat(
        [ids.reset_index(drop=True),
         sales_features(sales, origin),
         price_features(price, sales, origin),
         monetary_features(price, sales, origin)],
        axis=1,
    )
    df = relative_price(df)

    df.insert(0, "split", split)
    df.insert(1, "origin", weeks[origin])
    # Calendar position of the window being forecast -- the data is seasonal.
    df.insert(2, "origin_week_of_year", pd.Timestamp(weeks[origin]).isocalendar()[1])
    df["units_13w"] = target
    return df


def main():
    ids, sales, price, weeks, n_phase0 = load_panel()
    n_weeks = sales.shape[1]
    test_origin = n_phase0 + HORIZON          # first Phase 2 week

    print(f"{len(ids):,} series | {n_weeks} weeks "
          f"({weeks[0]:%Y-%m-%d} to {weeks[-1]:%Y-%m-%d})")
    print(f"phase 0 ends at week {n_phase0}, test origin = week {test_origin} "
          f"({weeks[test_origin]:%Y-%m-%d})\n")

    # Train origins stay inside Phase 0 and are spaced by HORIZON so that no
    # two training targets share a week.
    last_train_origin = n_phase0 - HORIZON
    train_origins = list(range(last_train_origin, MIN_HISTORY - 1, -HORIZON))[::-1]

    frames = [build_origin(ids, sales, price, weeks, o, "train")
              for o in train_origins]
    train = pd.concat(frames, ignore_index=True)
    test = build_origin(ids, sales, price, weeks, test_origin, "test")

    OUT.mkdir(parents=True, exist_ok=True)
    train.to_parquet(OUT / "vn1_train.parquet", index=False)
    test.to_parquet(OUT / "vn1_test.parquet", index=False)

    print(f"train origins ({len(train_origins)}): "
          f"{', '.join(f'{weeks[o]:%Y-%m-%d}' for o in train_origins)}")
    print(f"train {len(train):,} rows x {train.shape[1]} cols")
    print(f"test  {len(test):,} rows")
    print(f"\ntarget (13w units): train median {train.units_13w.median():.0f} "
          f"max {train.units_13w.max():.0f} | "
          f"zero share {100 * (train.units_13w == 0).mean():.1f}%")


if __name__ == "__main__":
    main()
