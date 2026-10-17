"""Draw the train/test timeline for the article.

A schematic, not a chart: the job is structure, so there is no quantitative
axis beyond the week number. Colours are the reference categorical slots.
Renders a light and a dark PNG, plus SVG sources.
"""

from datetime import date, timedelta
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"

WEEK0 = date(2020, 7, 6)
N_PHASE0, HORIZON = 170, 13
TEST_ORIGIN = N_PHASE0 + HORIZON          # 183
N_WEEKS = TEST_ORIGIN + HORIZON           # 196
MIN_HISTORY = 52

# Train origins: inside Phase 0, spaced by HORIZON so no two targets overlap.
# Same expression as build_vn1_features.py, so the figure cannot drift.
ORIGINS = list(range(N_PHASE0 - HORIZON, MIN_HISTORY - 1, -HORIZON))[::-1]

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
                  ink3="#86857d", hist="#2a78d6", targ="#eb6834",
                  test="#6250d6", band="#eceae4", grid="#d8d5cd"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7",
                 ink3="#8a8982", hist="#3987e5", targ="#d95926",
                 test="#9085e9", band="#2c2c2a", grid="#3e3e3a"),
}


def wk(n):
    return (WEEK0 + timedelta(weeks=n)).strftime("%b %Y")


def bar(ax, x0, x1, y, h, color):
    ax.add_patch(FancyBboxPatch(
        (x0, y - h / 2), x1 - x0, h,
        boxstyle="round,pad=0,rounding_size=1.5",
        linewidth=0, facecolor=color, zorder=3, mutation_aspect=0.33))


def draw(theme_name):
    c = THEMES[theme_name]
    n_train = len(ORIGINS)
    fig, ax = plt.subplots(figsize=(10.8, 6.6), dpi=220)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])

    pitch, h = 1.0, 0.54
    top = n_train - 1
    y_test = top - n_train * pitch

    band_y = top + 1.30
    phases = [
        (0, N_PHASE0, "PHASE 0", "170 weeks of history", False),
        (N_PHASE0, TEST_ORIGIN, "PHASE 1", "13 wks", True),
        (TEST_ORIGIN, N_WEEKS, "PHASE 2", "13 wks", True),
    ]
    for i, (x0, x1, label, sub, narrow) in enumerate(phases):
        ax.add_patch(Rectangle((x0 + 0.8, band_y - 0.27), x1 - x0 - 1.6, 0.54,
                               facecolor=c["band"], edgecolor="none", zorder=2))
        mid = (x0 + x1) / 2
        if narrow:
            # Too thin for inline text: label above, with a leader, and stagger
            # the two so they cannot collide.
            ly = band_y + (0.70 if i == 1 else 1.22)
            ax.plot([mid, mid], [band_y + 0.30, ly - 0.16], lw=0.8,
                    color=c["grid"], zorder=2, solid_capstyle="butt")
            ax.text(mid, ly, f"{label}  ·  {sub}", ha="center", va="bottom",
                    fontsize=8.0, color=c["ink2"], fontweight="bold", zorder=4)
        else:
            ax.text(mid, band_y, f"{label}  ·  {sub}", ha="center",
                    va="center", fontsize=9.2, color=c["ink2"],
                    fontweight="bold", zorder=4)

    for x in (N_PHASE0, TEST_ORIGIN):
        ax.plot([x, x], [y_test - 0.95, band_y + 0.30], lw=1.0,
                color=c["grid"], zorder=1, solid_capstyle="butt")

    # ---- training rows: history expands, target slides right ------------
    for i, origin in enumerate(ORIGINS):
        y = top - i * pitch
        bar(ax, 0, origin - 0.7, y, h, c["hist"])
        bar(ax, origin + 0.7, origin + HORIZON, y, h, c["targ"])
        if i == 0:
            lab = "origin 1"
        elif i == n_train - 1:
            lab = f"origin {n_train}"
        else:
            lab = "·"
        ax.text(-5, y, lab, ha="right", va="center",
                fontsize=8.0 if lab != "·" else 10,
                color=c["ink2"] if lab != "·" else c["ink3"])

    bar(ax, 0, TEST_ORIGIN - 0.7, y_test, h, c["hist"])
    bar(ax, TEST_ORIGIN + 0.7, N_WEEKS, y_test, h, c["test"])
    ax.text(-5, y_test, "TEST", ha="right", va="center", fontsize=8.2,
            color=c["ink"], fontweight="bold")

    # ---- direct labels, placed clear of the marks -----------------------
    ax.text(ORIGINS[0] / 2, top + 0.52, "features: every week before the origin",
            ha="center", va="bottom", fontsize=8.2, color=c["ink2"])
    ax.text(ORIGINS[0] + HORIZON / 2, top + 0.52, "target: the next 13 weeks",
            ha="left", va="bottom", fontsize=8.2, color=c["ink2"])
    # Sits directly under the only purple mark, so it needs no leader.
    ax.text(N_WEEKS, y_test - 0.60, "the 13 weeks competitors were scored on",
            ha="right", va="center", fontsize=8.2, color=c["ink"])

    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=8, mec="none", mfc=c["hist"],
                   label="history used to build features"),
        plt.Line2D([], [], marker="s", ls="", ms=8, mec="none", mfc=c["targ"],
                   label="training target (13 weeks)"),
        plt.Line2D([], [], marker="s", ls="", ms=8, mec="none", mfc=c["test"],
                   label="test target — Phase 2"),
    ]
    leg = ax.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(0.0, -0.105), ncol=3, frameon=False,
                    fontsize=8.4, handletextpad=0.5, columnspacing=2.2)
    for t in leg.get_texts():
        t.set_color(c["ink2"])

    ax.set_xlim(-34, N_WEEKS + 5)
    ax.set_ylim(y_test - 1.30, band_y + 1.85)
    ax.set_yticks([])
    ticks = [0, 52, 104, N_PHASE0, N_WEEKS]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"wk {t}\n{wk(t)}" for t in ticks],
                       fontsize=7.6, color=c["ink3"], linespacing=1.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(c["grid"])
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="x", length=3, color=c["grid"], pad=5)

    fig.text(0.012, 0.975,
             "One row per series per forecast origin — features only ever look backwards",
             fontsize=11.0, color=c["ink"], fontweight="bold", va="top")
    fig.text(0.012, 0.930,
             f"{n_train} training origins inside Phase 0, spaced 13 weeks so no "
             f"two targets overlap  ·  {n_train} × 15,053 series = 135,477 rows",
             fontsize=8.4, color=c["ink2"], va="top")

    fig.subplots_adjust(top=0.855, bottom=0.165, left=0.075, right=0.985)

    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"timeline-{theme_name}.{ext}", facecolor=c["surface"])
    plt.close(fig)
    print(f"wrote {OUT / f'timeline-{theme_name}.png'}")


if __name__ == "__main__":
    mpl.rcParams["font.family"] = ["DejaVu Sans"]
    print(f"origins (weeks): {ORIGINS}")
    print(f"test origin: week {TEST_ORIGIN} ({wk(TEST_ORIGIN)}), "
          f"target weeks {TEST_ORIGIN}-{N_WEEKS - 1}")
    for name in THEMES:
        draw(name)
