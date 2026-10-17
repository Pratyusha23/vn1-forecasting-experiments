"""Waterfall of the four feature techniques, on both metrics.

Each technique is measured ALONE against the baseline, so the steps do not
truly accumulate. The chart shows that: the running total is drawn as a
waterfall, then "if additive" is set beside the measured all-four result, and
the two differ. Hiding that behind a tidy waterfall would overstate the case.

Two panels, same technique order, separate y-scales — never a dual axis.
"""

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from vn1_lgbm import DATA

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "figures"

BASE = "0 Baseline (volume + trend only)"
ALL = "ALL feature techniques"
ORDER = ["2 Velocity classification", "3 Target encoding (Warehouse)",
         "4 Value proposition", "5 RFMT"]
SHORT = {"2 Velocity classification": "Velocity\nclassification",
         "3 Target encoding (Warehouse)": "Target\nencoding",
         "4 Value proposition": "Value\nproposition",
         "5 RFMT": "RFMT"}

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
                  ink3="#86857d", grid="#e4e1d9", anchor="#9a998f",
                  better="#2a78d6", worse="#e34948", conn="#b5b3a9"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7",
                 ink3="#8a8982", grid="#333330", anchor="#7a7970",
                 better="#3987e5", worse="#e66767", conn="#4d4d48"),
}


def panel(ax, c, agg, base_row, metric, pvals, title, lower_note):
    deltas = [agg.loc[t, metric] - base_row[metric] for t in ORDER]
    measured = agg.loc[ALL, metric]
    levels = list(np.cumsum([base_row[metric]] + deltas))

    # Fix the scale BEFORE drawing, so every label uses the same offset.
    lo = min(levels + [measured])
    hi = max(levels + [measured])
    span = max(hi - lo, 1e-6)
    ax.set_ylim(lo - span * 0.50, hi + span * 0.46)
    pad = span * 0.035

    labels = ["Baseline"] + [SHORT[t] for t in ORDER] + ["measured,\nall four"]
    x = np.arange(len(labels))
    run = base_row[metric]

    ax.bar(0, run, width=0.62, color=c["anchor"], zorder=3)
    ax.text(0, run + pad, f"{run:.2f}", ha="center", va="bottom", fontsize=8.6,
            color=c["ink"], fontweight="bold")

    for i, (t, d) in enumerate(zip(ORDER, deltas), start=1):
        pv = pvals.get(t, 1.0)
        sig = pv < 0.05
        col = c["better"] if d < 0 else c["worse"]
        bottom, top = (run + d, run) if d < 0 else (run, run + d)
        # Significance is double-encoded: solid fill plus the p-value in text,
        # so it survives greyscale and colour-vision differences.
        ax.bar(i, max(top - bottom, span * 0.004), bottom=bottom, width=0.62,
               color=col if sig else "none", zorder=3,
               edgecolor=col, linewidth=0 if sig else 1.4,
               alpha=1.0 if sig else 0.95,
               hatch=None if sig else "///")
        ax.plot([i - 0.69, i - 0.31], [run, run], lw=1.0, ls=(0, (3, 2)),
                color=c["conn"], zorder=2)
        ax.text(i, max(top, run) + pad, f"{d:+.2f}", ha="center",
                va="bottom", fontsize=9.0, color=c["ink"], fontweight="bold")
        ax.text(i, max(top, run) + pad * 4.6,
                f"p={pv:.3f}" + ("  ✓" if sig else ""), ha="center",
                va="bottom", fontsize=7.6,
                color=c["ink"] if sig else c["ink3"],
                fontweight="bold" if sig else "normal")
        run = run + d

    j = len(ORDER) + 1
    ax.bar(j, measured, width=0.62, color=c["anchor"], zorder=3)
    ax.text(j, measured + pad, f"{measured:.2f}", ha="center", va="bottom",
            fontsize=8.6, color=c["ink"], fontweight="bold")
    ax.plot([j - 0.69, j - 0.31], [run, run], lw=1.0, ls=(0, (3, 2)),
            color=c["conn"], zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.0, color=c["ink2"], linespacing=1.4)
    ax.set_ylabel(title, fontsize=8.8, color=c["ink3"], labelpad=9)
    ax.tick_params(axis="both", length=3, color=c["grid"], pad=4,
                   labelcolor=c["ink3"], labelsize=8.0)
    ax.tick_params(axis="x", labelcolor=c["ink2"])
    ax.grid(axis="y", color=c["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(c["grid"])
    ax.spines["bottom"].set_linewidth(0.9)
    ax.text(0.995, 0.975, lower_note, transform=ax.transAxes, ha="right",
            va="top", fontsize=8.0, color=c["ink3"])


def draw(agg, pvals, pmae, theme):
    c = THEMES[theme]
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 8.4), dpi=220)
    fig.patch.set_facecolor(c["surface"])
    for ax in axes:
        ax.set_facecolor(c["surface"])

    base_row = agg.loc[BASE]
    panel(axes[0], c, agg, base_row, "score", pvals,
          "competition score  (MAE% + |Bias%|)", "lower is better  ·  ✓ = p < 0.05")
    panel(axes[1], c, agg, base_row, "mae", pmae,
          "MAE%  (accuracy only)", "lower is better")

    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=9, mec="none",
                   mfc=c["better"], label="improves the metric"),
        plt.Line2D([], [], marker="s", ls="", ms=9, mec="none",
                   mfc=c["worse"], label="worsens it"),
        plt.Line2D([], [], marker="s", ls="", ms=9, mec="none",
                   mfc=c["anchor"], label="absolute level"),
        plt.Line2D([], [], marker="s", ls="", ms=9, mfc="none",
                   mec=c["ink3"], mew=1.4, label="hollow + hatched = p ≥ 0.05"),
    ]
    leg = axes[1].legend(handles=handles, loc="upper left",
                         bbox_to_anchor=(0.0, -0.17), ncol=3, frameon=False,
                         fontsize=8.4, handletextpad=0.5, columnspacing=2.0)
    for t in leg.get_texts():
        t.set_color(c["ink2"])

    fig.text(0.012, 0.978,
             "The same four techniques, ranked almost backwards by two metrics",
             fontsize=11.4, color=c["ink"], fontweight="bold", va="top")
    fig.text(0.012, 0.950,
             "Each technique measured alone against the full-data baseline, "
             "mean of 5 seeds, paired t-test. Because the steps are independent "
             "measurements rather than a\nsequence, the running total is "
             "indicative only — the final bar is the measured result of running "
             "all four together.",
             fontsize=8.4, color=c["ink2"], va="top", linespacing=1.5)

    fig.subplots_adjust(top=0.875, bottom=0.115, left=0.085, right=0.985,
                        hspace=0.42)

    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"ablation-waterfall-{theme}.{ext}",
                    facecolor=c["surface"])
    plt.close(fig)
    print(f"wrote {OUT / f'ablation-waterfall-{theme}.png'}")


def main():
    mpl.rcParams["font.family"] = ["DejaVu Sans"]
    res = pd.read_csv(DATA / "vn1_ablation2_raw.csv")
    agg = res.groupby("technique")[["score", "mae"]].mean()

    base = res[res.technique == BASE].set_index("seed")
    pvals, pmae = {}, {}
    for t in ORDER:
        g = res[res.technique == t].set_index("seed")
        pvals[t] = stats.ttest_rel(g["score"], base["score"]).pvalue
        pmae[t] = stats.ttest_rel(g["mae"], base["mae"]).pvalue
    print("score p:", {k.split()[0]: round(v, 4) for k, v in pvals.items()})
    print("mae   p:", {k.split()[0]: round(v, 4) for k, v in pmae.items()})

    for theme in THEMES:
        draw(agg, pvals, pmae, theme)


if __name__ == "__main__":
    main()
