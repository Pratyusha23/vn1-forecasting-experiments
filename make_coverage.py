"""How much of the stratum space a random draw actually reaches.

The question Case 1 turns on: does this dataset have the uneven grouping that
stratified sampling exists to fix? Counts how many of the 1,486
warehouse x velocity strata a random draw leaves empty, thinly covered, or
properly covered, at each sampling budget.

Mean of 8 draws per budget, so no single lucky seed sets the picture.
"""

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vn1_lgbm import DATA, add_velocity

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent / "figures"
BUDGETS = [2_000, 5_000, 10_000, 20_000, 50_000]
SEEDS = range(8)
THIN = 5                      # fewer than this many rows = too thin to learn from

# Ordinal ramp, one hue. Severity is carried by salience in both modes: on the
# light surface that means darker, on the dark surface lighter.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
                  ink3="#86857d", grid="#d8d5cd",
                  ok="#86b6ef", thin="#2a78d6", empty="#104281",
                  empty_ink="#fcfcfb"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7",
                 ink3="#8a8982", grid="#3e3e3a",
                 ok="#1c5cab", thin="#3987e5", empty="#cde2fb",
                 empty_ink="#10203a"),
}


def coverage(train, strata, n_strata):
    rows = []
    for budget in BUDGETS:
        counts = []
        for seed in SEEDS:
            drawn = strata[train.sample(budget, random_state=seed).index]
            hit = drawn.value_counts()
            empty = n_strata - len(hit)
            thin = int((hit < THIN).sum())
            counts.append((empty, thin, n_strata - empty - thin))
        m = np.mean(counts, axis=0)
        rows.append(dict(budget=budget, empty=m[0], thin=m[1], ok=m[2]))
    return pd.DataFrame(rows)


def draw(df, n_strata, theme_name):
    c = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(10.0, 4.9), dpi=220)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])

    y = np.arange(len(df))[::-1]
    h = 0.52
    gap = n_strata * 0.0022          # ~2px of surface between segments

    for i, r in df.iterrows():
        yy = y[i]
        x = 0.0
        for key, col in (("empty", c["empty"]), ("thin", c["thin"]),
                         ("ok", c["ok"])):
            w = r[key]
            ax.barh(yy, w - gap, left=x, height=h, color=col,
                    edgecolor="none", zorder=3)
            x += w
        # Direct label on the segment that carries the point.
        ax.text(r["empty"] / 2, yy, f"{r['empty']:.0f}", ha="center",
                va="center", fontsize=9.0, color=c["empty_ink"],
                fontweight="bold", zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{b:,}" for b in df["budget"]], fontsize=9,
                       color=c["ink2"])
    ax.set_ylabel("rows drawn", fontsize=8.8, color=c["ink3"], labelpad=10)

    ax.set_xlim(0, n_strata)
    ax.set_xticks([0, 400, 800, 1200, n_strata])
    ax.set_xticklabels(["0", "400", "800", "1,200", f"{n_strata:,}"],
                       fontsize=7.8, color=c["ink3"])
    ax.set_xlabel(f"warehouse × velocity strata  (of {n_strata:,})",
                  fontsize=8.8, color=c["ink3"], labelpad=8)

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(c["grid"])
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", length=3, color=c["grid"], pad=4)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=8, mec="none", mfc=c["empty"],
                   label="never drawn (empty)"),
        plt.Line2D([], [], marker="s", ls="", ms=8, mec="none", mfc=c["thin"],
                   label=f"under {THIN} rows"),
        plt.Line2D([], [], marker="s", ls="", ms=8, mec="none", mfc=c["ok"],
                   label=f"{THIN}+ rows"),
    ]
    leg = ax.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(0.0, -0.20), ncol=3, frameon=False,
                    fontsize=8.4, handletextpad=0.5, columnspacing=2.2)
    for t in leg.get_texts():
        t.set_color(c["ink2"])

    fig.text(0.012, 0.972,
             "A random draw never even sees most of the stratum space",
             fontsize=11.0, color=c["ink"], fontweight="bold", va="top")
    fig.text(0.012, 0.918,
             f"{n_strata:,} warehouse × velocity strata, median 18 rows. "
             f"Mean of {len(SEEDS)} random draws per budget.",
             fontsize=8.4, color=c["ink2"], va="top")

    fig.subplots_adjust(top=0.80, bottom=0.26, left=0.105, right=0.985)

    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"stratum-coverage-{theme_name}.{ext}",
                    facecolor=c["surface"])
    plt.close(fig)
    print(f"wrote {OUT / f'stratum-coverage-{theme_name}.png'}")


def main():
    mpl.rcParams["font.family"] = ["DejaVu Sans"]
    train, _ = add_velocity(pd.read_parquet(DATA / "vn1_train.parquet"))
    strata = train[["Warehouse", "velocity"]].astype(str).agg("|".join, axis=1)
    n_strata = strata.nunique()

    df = coverage(train, strata, n_strata)
    print(f"{n_strata:,} strata | median {int(strata.value_counts().median())} rows\n")
    print(df.round(1).to_string(index=False))
    df.to_csv(DATA / "vn1_stratum_coverage.csv", index=False)

    for name in THEMES:
        draw(df, n_strata, name)


if __name__ == "__main__":
    main()
