"""
CSCI 5612 - Project Part 3
Script 12: Explanatory diagrams for the Naive Bayes and Decision Tree overviews.

Writes ../docs/images/diag_*.png  (light and dark renderings)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

import theme

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "data", "clean")
IMG = os.path.join(HERE, "..", "docs", "images")


def save(fig, name):
    out = theme.out_name(name)
    fig.savefig(os.path.join(IMG, out)); plt.close(fig)
    print(f"    saved {out}")


def head(fig, text):
    fig.suptitle(text, x=0.012, ha="left", fontsize=14, fontweight="bold", y=0.975)


def box(ax, x, y, w, h, text, face, edge, fontsize=9, textcolor=None, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=face, edgecolor=edge, linewidth=1.6))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor or theme.INK, weight=weight, linespacing=1.45)


# ---------------------------------------------------------------------------

def diagram_train_test():
    """Why the two sets must not overlap."""
    C = theme.C
    fig, ax = plt.subplots(figsize=(10.2, 4.3))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.80, bottom=0.06)

    ax.add_patch(Rectangle((0.5, 6.4), 9.0, 1.15, facecolor=theme.GRID,
                           edgecolor=theme.INK_SOFT, linewidth=1.4))
    ax.text(5.0, 6.97, "25,817 labelled songs", ha="center", va="center",
            fontsize=10.5, color=theme.INK, weight="bold")

    for x0, wdt, col, lbl, sub, fs in [
            (0.5, 7.04, C[0], "Training set\n20,653 songs  (80%)",
             "the model is built from these", 9.6),
            (7.74, 1.76, C[1], "Test set\n5,164  (20%)",
             "never seen during training", 8.4)]:
        ax.add_patch(Rectangle((x0, 4.3), wdt, 1.15, facecolor=col,
                               edgecolor=theme.BG, linewidth=1.8))
        ax.text(x0 + wdt/2, 4.87, lbl, ha="center", va="center", fontsize=fs,
                color="#ffffff", weight="bold", linespacing=1.4)
        ax.text(x0 + wdt/2, 3.75, sub, ha="center", va="center", fontsize=8.8,
                color=theme.INK_SOFT)

    ax.add_patch(FancyArrowPatch((4.02, 6.3), (4.02, 5.55), color=theme.INK_SOFT,
                                 arrowstyle="-|>", mutation_scale=15, linewidth=1.4))
    ax.add_patch(FancyArrowPatch((8.62, 6.3), (8.62, 5.55), color=theme.INK_SOFT,
                                 arrowstyle="-|>", mutation_scale=15, linewidth=1.4))

    ax.text(5.0, 2.45, "No song appears in both boxes.",
            ha="center", fontsize=10.4, color=theme.ACCENT, weight="bold")
    ax.text(5.0, 1.5, "If a song were in both, the model would be marked on work it had already seen,\n"
                      "and the score would measure memory rather than understanding.",
            ha="center", fontsize=9.2, color=theme.INK_SOFT, linespacing=1.5)

    ax.set_xlim(0, 10); ax.set_ylim(0.6, 8.0)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    head(fig, "Splitting the data, and why the halves must stay separate")
    save(fig, "diag_train_test.png")


def diagram_nb_variants():
    """What each Naive Bayes variant assumes a feature is."""
    C = theme.C
    rng = np.random.default_rng(4)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.0))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.72, bottom=0.17, wspace=0.22)

    a1, a2, a3 = axes
    # Multinomial: non-negative weights, like counts
    vals = [0.72, 0.31, 0.58, 0.12, 0.44]
    a1.bar(range(5), vals, color=C[0], width=0.6, edgecolor=theme.BG, linewidth=1.5)
    a1.set_ylim(0, 1.0)
    a1.set_title("Multinomial", loc="left", fontsize=11.5, color=C[0])
    a1.set_xlabel("a feature is a non-negative\namount, like a count", fontsize=8.8)

    # Bernoulli: on/off
    on = [1, 0, 1, 0, 1]
    a2.bar(range(5), on, color=[C[2] if v else theme.GRID for v in on], width=0.6,
           edgecolor=theme.BG, linewidth=1.5)
    a2.set_ylim(0, 1.0); a2.set_yticks([0, 1], ["absent", "present"])
    a2.set_title("Bernoulli", loc="left", fontsize=11.5, color=C[2])
    a2.set_xlabel("a feature is simply\npresent or absent", fontsize=8.8)

    # Gaussian: a bell curve per class
    xs = np.linspace(-3.4, 3.4, 240)
    for mu, col, lab in [(-0.9, C[1], None), (0.9, C[6], None)]:
        a3.fill_between(xs, np.exp(-0.5*((xs-mu)/0.85)**2), color=col, alpha=0.45,
                        linewidth=0)
        a3.plot(xs, np.exp(-0.5*((xs-mu)/0.85)**2), color=col, linewidth=2)
    a3.set_ylim(0, 1.18); a3.set_yticks([])
    a3.set_title("Gaussian", loc="left", fontsize=11.5, color=C[1])
    a3.set_xlabel("a feature is a bell curve,\none per class", fontsize=8.8)

    for ax in axes:
        ax.set_xticks([]); ax.grid(False)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.spines["left"].set_edgecolor(theme.GRID)
        ax.spines["bottom"].set_edgecolor(theme.GRID)
    head(fig, "The three Naive Bayes variants differ in what a feature is assumed to be")
    save(fig, "diag_nb_variants.png")


def diagram_information_gain():
    """The real worked split, with the arithmetic shown."""
    ex = json.load(open(os.path.join(CLEAN, "dt_metrics.json")))["worked_example"]
    C = theme.C
    p, l, r = ex["parent"], ex["left"], ex["right"]

    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.02)

    box(ax, 5.0, 7.5, 4.5, 1.5,
        f"{p['n']:,} training songs\n{p['rap']:,} rap   ·   {p['not_rap']:,} not rap\n"
        f"entropy  {p['entropy']:.4f}      Gini  {p['gini']:.4f}",
        theme.GRID, theme.INK_SOFT, fontsize=9.4)

    ax.text(5.0, 6.52, f"split on  {ex['split']}", ha="center", fontsize=10,
            color=theme.ACCENT, weight="bold")

    for x, d, col, lab in [(2.3, l, C[0], "speechiness ≤ 0.15"),
                           (7.7, r, C[1], "speechiness > 0.15")]:
        ax.add_patch(FancyArrowPatch((5.0, 6.24), (x, 5.42), color=theme.INK_SOFT,
                                     arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.3, connectionstyle="arc3,rad=0.08"))
        box(ax, x, 4.55, 4.1, 1.6,
            f"{lab}\n{d['n']:,} songs  ({d['weight']*100:.1f}% of the parent)\n"
            f"{d['rap']:,} rap  ·  {d['not_rap']:,} not rap\nentropy  {d['entropy']:.4f}",
            col, theme.BG, fontsize=9.2, textcolor="#ffffff", weight="bold")

    ax.text(5.0, 2.72,
            f"weighted child entropy  =  {l['weight']:.3f} × {l['entropy']:.4f}"
            f"  +  {r['weight']:.3f} × {r['entropy']:.4f}  =  "
            f"{ex['weighted_child_entropy']:.4f}",
            ha="center", fontsize=9.8, color=theme.INK)
    ax.text(5.0, 1.75,
            f"information gain  =  {p['entropy']:.4f}  −  "
            f"{ex['weighted_child_entropy']:.4f}  =  {ex['information_gain']:.4f}",
            ha="center", fontsize=11.6, color=theme.ACCENT, weight="bold")
    ax.text(5.0, 0.85, "The tree tries every variable and every threshold, and keeps "
            "whichever split gains the most.",
            ha="center", fontsize=9.2, color=theme.INK_SOFT)

    ax.set_xlim(0, 10); ax.set_ylim(0.3, 8.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    head(fig, "Measuring how good a split is, on this project's own data")
    save(fig, "diag_information_gain.png")


def diagram_infinite_trees():
    """Why there is no such thing as 'the' tree for a dataset."""
    C = theme.C
    rng = np.random.default_rng(12)
    pts = rng.uniform(0.08, 0.92, size=(26, 2))
    lab = (pts[:, 0] + pts[:, 1] * 0.55 + rng.normal(0, 0.12, 26)) > 0.78

    cuts = [
        ([0.52], [], "split on loudness first"),
        ([], [0.46], "split on tempo first"),
        ([0.30, 0.68], [0.55], "split on both, repeatedly"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.0))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.74, bottom=0.14, wspace=0.14)

    for ax, (vx, hy, title) in zip(axes, cuts):
        ax.scatter(pts[~lab, 0], pts[~lab, 1], s=36, color=C[0], alpha=0.85,
                   linewidths=0)
        ax.scatter(pts[lab, 0], pts[lab, 1], s=36, color=C[1], alpha=0.85,
                   linewidths=0)
        for x in vx:
            ax.axvline(x, color=theme.ACCENT, linewidth=2.0)
        for y in hy:
            ax.axhline(y, color=theme.ACCENT, linewidth=2.0)
        ax.set_title(title, loc="left", fontsize=10, color=theme.INK_SOFT)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values(): sp.set_edgecolor(theme.GRID)
    axes[0].set_ylabel("Tempo", fontsize=9.4)
    for ax in axes: ax.set_xlabel("Loudness", fontsize=9.4)

    fig.text(0.5, 0.045, "Same songs, same labels, three different trees — and the "
             "thresholds can be nudged endlessly, so the supply never runs out.",
             ha="center", fontsize=9.3, color=theme.INK_SOFT)
    head(fig, "Why a dataset has infinitely many possible trees")
    save(fig, "diag_infinite_trees.png")


if __name__ == "__main__":
    print("Building Module 3 concept diagrams")
    for mode in theme.MODES:
        theme.set_theme(mode)
        print(f"  --- {mode} ---")
        diagram_train_test()
        diagram_nb_variants()
        diagram_information_gain()
        diagram_infinite_trees()
    print("Done.")
