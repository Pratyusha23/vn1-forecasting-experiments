"""The three Case 1 plots: accuracy cost, long-tail gain, and training speedup.

Reads data/vn1_sampling2_raw.csv (8 seeds per budget per strategy) and refits
the full-data model once for the reference lines. Bands are +/-1 sd across
seeds — the aggregate score is noisy, and hiding that would misrepresent it.
"""

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_vn1_features import FEATURE_GROUPS
from vn1_lgbm import (DATA, fit_predict, load_frames, official_score,
                      segment_score)

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "figures"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
                  ink3="#86857d", grid="#e4e1d9", rule="#9a998f",
                  rnd="#2a78d6", strat="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7",
                 ink3="#8a8982", grid="#333330", rule="#7a7970",
                 rnd="#3987e5", strat="#d95926"),
}
LABEL = {"random": "random draw", "stratified": "stratified draw"}


def frame(ax, c, title, sub, ylab):
    ax.set_facecolor(c["surface"])
    ax.set_xscale("log")
    ax.set_xticks([2_000, 5_000, 10_000, 20_000, 50_000])
    ax.set_xticklabels(["2k", "5k", "10k", "20k", "50k"], fontsize=8.6,
                       color=c["ink3"])
    ax.minorticks_off()
    ax.set_xlabel("rows drawn for training  (of 135,477)", fontsize=8.8,
                  color=c["ink3"], labelpad=8)
    ax.set_ylabel(ylab, fontsize=8.8, color=c["ink3"], labelpad=9)
    ax.tick_params(axis="both", length=3, color=c["grid"], pad=4,
                   labelcolor=c["ink3"], labelsize=8.4)
    ax.grid(axis="y", color=c["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(c["grid"])
    ax.spines["bottom"].set_linewidth(0.9)
    return title, sub


def series(ax, g, c, col, key, band=True):
    m = g.groupby("budget")[key].mean()
    s = g.groupby("budget")[key].std()
    if band:
        ax.fill_between(m.index, m - s, m + s, color=col, alpha=0.13,
                        lw=0, zorder=2)
    ax.plot(m.index, m.values, color=col, lw=2.0, zorder=4,
            solid_capstyle="round")
    ax.plot(m.index, m.values, ls="", marker="o", ms=7, mfc=col,
            mec=c["surface"], mew=1.6, zorder=5)
    return m


def finish(fig, ax, c, title, sub, handles):
    leg = ax.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(0.0, -0.165), ncol=len(handles),
                    frameon=False, fontsize=8.4, handletextpad=0.6,
                    columnspacing=2.0)
    for t in leg.get_texts():
        t.set_color(c["ink2"])
    fig.text(0.012, 0.972, title, fontsize=11.0, color=c["ink"],
             fontweight="bold", va="top")
    fig.text(0.012, 0.917, sub, fontsize=8.4, color=c["ink2"], va="top")
    fig.subplots_adjust(top=0.80, bottom=0.235, left=0.105, right=0.975)


def marker(c, col, label):
    return plt.Line2D([], [], color=col, lw=2.0, marker="o", ms=7,
                      mfc=col, mec=c["surface"], mew=1.4, label=label)


def rule(c, label):
    return plt.Line2D([], [], color=c["rule"], lw=1.3, ls=(0, (5, 3)),
                      label=label)


# ------------------------------------------------------------------ plots

def plot_gap(res, ref, theme):
    c = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=220)
    fig.patch.set_facecolor(c["surface"])
    t, s = frame(ax, c, "Sampling costs accuracy — but not much, and not evenly",
                 "Competition score above the full-data model. Lower is better; "
                 "band is ±1 sd over 8 seeds.",
                 "points worse than training on everything")

    ax.axhline(0, color=c["rule"], lw=1.3, ls=(0, (5, 3)), zorder=3)
    ax.text(1_850, 0, "full data", va="bottom", ha="left", fontsize=8.0,
            color=c["ink2"])

    for strat, col in (("random", c["rnd"]), ("stratified", c["strat"])):
        g = res[res.strategy == strat].copy()
        g["gap"] = g["score"] - ref["score"]
        series(ax, g, c, col, "gap")

    ax.set_xlim(1_700, 76_000)
    finish(fig, ax, c, t, s,
           [marker(c, c["rnd"], LABEL["random"]),
            marker(c, c["strat"], LABEL["stratified"]),
            rule(c, "full 135,477 rows")])
    save(fig, f"sampling-gap-{theme}", c)


def plot_slow(res, ref, theme):
    c = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=220)
    fig.patch.set_facecolor(c["surface"])
    t, s = frame(ax, c, "On slow movers, a stratified sample beats all the data",
                 "Competition score on the two slowest velocity tiers. Lower is "
                 "better; band is ±1 sd over 8 seeds.",
                 "score, slowest two velocity tiers")

    ax.axhline(ref["slow"], color=c["rule"], lw=1.3, ls=(0, (5, 3)), zorder=3)
    ax.text(1_850, ref["slow"], "full data", va="bottom", ha="left",
            fontsize=8.0, color=c["ink2"])

    for strat, col in (("random", c["rnd"]), ("stratified", c["strat"])):
        series(ax, res[res.strategy == strat], c, col, "slow")

    g = res[res.strategy == "stratified"].groupby("budget")["slow"].mean()
    ax.annotate("every stratified point sits below the full-data line",
                xy=(g.index[2], g.iloc[2]), xytext=(3_400, g.iloc[2] - 2.6),
                fontsize=8.2, color=c["ink"], ha="left", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.9, color=c["ink3"],
                                shrinkA=2, shrinkB=4))

    ax.set_xlim(1_700, 76_000)
    finish(fig, ax, c, t, s,
           [marker(c, c["rnd"], LABEL["random"]),
            marker(c, c["strat"], LABEL["stratified"]),
            rule(c, "full 135,477 rows")])
    save(fig, f"sampling-slow-{theme}", c)


def plot_speed(res, ref, theme):
    c = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=220)
    fig.patch.set_facecolor(c["surface"])
    t, s = frame(ax, c, "What the sample buys you: training time",
                 f"Model fit speed against the full-data fit "
                 f"({ref['fit_s']:.1f} s). Higher is faster.",
                 "times faster than the full-data fit")

    ax.axhline(1, color=c["rule"], lw=1.3, ls=(0, (5, 3)), zorder=3)
    ax.text(1_850, 1, "full data", va="bottom", ha="left", fontsize=8.0,
            color=c["ink2"])

    for strat, col in (("random", c["rnd"]), ("stratified", c["strat"])):
        g = res[res.strategy == strat].copy()
        g["speedup"] = ref["fit_s"] / g["fit_s"]
        m = series(ax, g, c, col, "speedup")
        if strat == "random":
            ax.annotate(f"{m.iloc[0]:.0f}× at 2k rows",
                        xy=(m.index[0], m.iloc[0]),
                        xytext=(2_600, m.iloc[0] + 0.4), fontsize=8.2,
                        color=c["ink"], ha="left",
                        arrowprops=dict(arrowstyle="-", lw=0.9,
                                        color=c["ink3"]))

    ax.set_xlim(1_700, 76_000)
    ax.set_ylim(0, None)
    finish(fig, ax, c, t, s,
           [marker(c, c["rnd"], LABEL["random"]),
            marker(c, c["strat"], LABEL["stratified"]),
            rule(c, "full 135,477 rows")])
    save(fig, f"sampling-speed-{theme}", c)


def save(fig, stem, c):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", facecolor=c["surface"])
    plt.close(fig)
    print(f"wrote {OUT / (stem + '.png')}")


def main():
    mpl.rcParams["font.family"] = ["DejaVu Sans"]
    res = pd.read_csv(DATA / "vn1_sampling2_raw.csv")

    train, test, phase2 = load_frames()
    tier = test["velocity"].to_numpy()
    pred, secs = fit_predict(train, test, FEATURE_GROUPS["base"], seed=0,
                             timed=True)
    ref = dict(score=official_score(pred, test, phase2)[0],
               slow=segment_score(pred, test, phase2, tier <= 1),
               fit_s=secs)
    print(f"full-data reference: score {ref['score']:.2f}, "
          f"slow {ref['slow']:.2f}, fit {ref['fit_s']:.2f}s\n")

    for theme in THEMES:
        plot_gap(res, ref, theme)
        plot_slow(res, ref, theme)
        plot_speed(res, ref, theme)


if __name__ == "__main__":
    main()
