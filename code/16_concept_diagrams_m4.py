"""
CSCI 5612 - Project Part 4
Script 16: Explanatory diagrams for the Regression and AdaBoost overviews.

Writes ../docs/images/diag_*.png  (light and dark renderings)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import theme

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "..", "docs", "images")


def save(fig, name):
    out = theme.out_name(name)
    fig.savefig(os.path.join(IMG, out)); plt.close(fig)
    print(f"    saved {out}")


def head(fig, text):
    fig.suptitle(text, x=0.012, ha="left", fontsize=14, fontweight="bold", y=0.975)


def panel(ax, title, color=None):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_edgecolor(theme.GRID)
    ax.spines["bottom"].set_edgecolor(theme.GRID)
    ax.set_title(title, loc="left", fontsize=11.5, pad=9, color=color or theme.INK)


# ---------------------------------------------------------------------------

def diagram_linear_vs_logistic():
    C = theme.C
    rng = np.random.default_rng(5)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.5))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.15, wspace=0.26)

    # Linear: a number in, a number out
    x = rng.uniform(0, 10, 70)
    y = 14 + 4.1 * x + rng.normal(0, 7, 70)
    a1.scatter(x, y, s=26, color=theme.GRID, linewidths=0)
    xs = np.linspace(0, 10, 50)
    a1.plot(xs, 14 + 4.1 * xs, color=C[0], linewidth=2.6)
    a1.annotate("fits a straight line\nthrough the points", xy=(6.6, 14 + 4.1 * 6.6),
                xytext=(-6, 26), textcoords="offset points", ha="right",
                fontsize=9, color=C[0], fontweight="bold")
    a1.set_xlabel("A measurement"); a1.set_ylabel("Predicted number")
    a1.set_yticks([]); a1.set_xticks([])
    panel(a1, "Linear  ·  predicts a number", C[0])

    # Logistic: a number in, a probability out, then a class
    xs2 = np.linspace(-6, 6, 300)
    sig = 1 / (1 + np.exp(-xs2))
    pos = rng.normal(2.1, 1.5, 34); neg = rng.normal(-2.1, 1.5, 34)
    a2.scatter(neg, np.zeros(34) + 0.02, s=26, color=C[0], linewidths=0, alpha=0.8)
    a2.scatter(pos, np.ones(34) - 0.02, s=26, color=C[1], linewidths=0, alpha=0.8)
    a2.plot(xs2, sig, color=C[2], linewidth=2.6)
    a2.axhline(0.5, color=theme.INK_SOFT, linestyle="--", linewidth=1.4)
    a2.annotate("0.5 cut-off", xy=(-5.8, 0.5), xytext=(0, 6),
                textcoords="offset points", fontsize=8.8, color=theme.INK_SOFT)
    a2.set_ylim(-0.12, 1.12)
    a2.set_xlabel("A measurement"); a2.set_ylabel("Probability of the class")
    a2.set_xticks([]); a2.set_yticks([0, 0.5, 1])
    panel(a2, "Logistic  ·  predicts a probability", C[2])

    head(fig, "Two regressions that answer different questions")
    save(fig, "diag_linear_vs_logistic.png")


def diagram_sigmoid():
    """Score -> probability -> class, as three stages."""
    C = theme.C
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.1))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.76, bottom=0.17, wspace=0.30)
    a1, a2, a3 = axes

    xs = np.linspace(-6, 6, 300)

    # 1. the linear score, unbounded
    a1.plot(xs, 0.9 * xs, color=C[0], linewidth=2.6)
    a1.axhline(0, color=theme.INK_SOFT, linewidth=1)
    a1.set_ylim(-6, 6)
    a1.set_xlabel("weighted sum of the measurements")
    a1.set_ylabel("score")
    a1.set_xticks([]); a1.set_yticks([0], ["0"])
    a1.annotate("runs from −∞ to +∞,\nso it is not a probability",
                xy=(0.5, 0.04), xycoords="axes fraction", ha="center",
                fontsize=8.8, color=theme.INK_SOFT)
    panel(a1, "1 · a linear score", C[0])

    # 2. the sigmoid squashes it
    a2.plot(xs, 1 / (1 + np.exp(-xs)), color=C[2], linewidth=2.8)
    a2.axhline(0.5, color=theme.INK_SOFT, linestyle=":", linewidth=1.3)
    a2.axvline(0, color=theme.INK_SOFT, linestyle=":", linewidth=1.3)
    a2.set_ylim(-0.06, 1.06)
    a2.set_xlabel("score"); a2.set_ylabel("probability")
    a2.set_xticks([0], ["0"]); a2.set_yticks([0, 0.5, 1])
    a2.annotate("a score of 0\nbecomes 0.5", xy=(0, 0.5), xytext=(10, -28),
                textcoords="offset points", fontsize=8.8, color=theme.INK_SOFT)
    panel(a2, "2 · the sigmoid squashes it", C[2])

    # 3. a threshold turns it into a class
    probs = np.array([0.04, 0.12, 0.26, 0.41, 0.58, 0.73, 0.88, 0.96])
    cols = [C[1] if p >= 0.5 else C[0] for p in probs]
    a3.bar(range(len(probs)), probs, color=cols, width=0.62,
           edgecolor=theme.BG, linewidth=1.5)
    a3.axhline(0.5, color=theme.INK_SOFT, linestyle="--", linewidth=1.6)
    a3.set_ylim(0, 1.08)
    a3.set_xticks([]); a3.set_yticks([0, 0.5, 1])
    a3.set_xlabel("eight example songs"); a3.set_ylabel("probability")
    a3.text(1.5, 1.015, "no", ha="center", fontsize=9.2, color=C[0],
            fontweight="bold")
    a3.text(5.5, 1.015, "yes", ha="center", fontsize=9.2, color=C[1],
            fontweight="bold")
    panel(a3, "3 · a cut-off makes it a class", C[1])

    head(fig, "How logistic regression gets from a score to a yes or no")
    save(fig, "diag_sigmoid.png")


def diagram_boosting_vs_bagging():
    C = theme.C
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.05, wspace=0.08)

    def tree(ax, x, y, col, w=1.15, h=0.78, label=""):
        ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                    boxstyle="round,pad=0.03,rounding_size=0.1",
                                    facecolor=col, edgecolor=theme.BG, linewidth=1.6))
        if label:
            ax.text(x, y, label, ha="center", va="center", fontsize=8.6,
                    color="#ffffff", weight="bold")

    # Bagging: independent, parallel, then averaged
    for i, yy in enumerate([3.4, 2.1, 0.8]):
        tree(a1, 2.0, yy, C[0], label=f"tree {i+1}")
        a1.add_patch(FancyArrowPatch((2.72, yy), (4.5, 2.1), color=theme.INK_SOFT,
                                     arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.2, connectionstyle="arc3,rad=0.12"))
    a1.text(2.0, 4.35, "built independently,\nall at the same time", ha="center",
            fontsize=9, color=theme.INK_SOFT)
    tree(a1, 5.3, 2.1, theme.INK_SOFT, w=1.5, label="average")
    a1.text(5.3, 0.9, "every tree gets\nan equal vote", ha="center", fontsize=9,
            color=theme.INK_SOFT)
    panel(a1, "Bagging  ·  in parallel", C[0])

    # Boosting: sequential, each fixing the last, weighted vote
    for i, xx in enumerate([1.3, 3.0, 4.7]):
        tree(a2, xx, 2.6, C[1], label=f"stump {i+1}")
        if i < 2:
            a2.add_patch(FancyArrowPatch((xx + 0.62, 2.6), (xx + 1.08, 2.6),
                                         color=theme.ACCENT, arrowstyle="-|>",
                                         mutation_scale=13, linewidth=1.6))
    a2.text(3.0, 4.35, "built one after another, each one\nfocused on the previous "
            "mistakes", ha="center", fontsize=9, color=theme.INK_SOFT)
    for xx, a in [(1.3, "α₁"), (3.0, "α₂"), (4.7, "α₃")]:
        a2.text(xx, 1.78, a, ha="center", fontsize=10.5, color=theme.ACCENT,
                weight="bold")
    tree(a2, 6.3, 2.6, theme.INK_SOFT, w=1.5, label="weighted\nvote")
    a2.add_patch(FancyArrowPatch((5.42, 2.6), (5.55, 2.6), color=theme.INK_SOFT,
                                 arrowstyle="-|>", mutation_scale=12, linewidth=1.2))
    a2.text(3.6, 0.9, "better stumps get a bigger say", ha="center", fontsize=9,
            color=theme.INK_SOFT)
    panel(a2, "Boosting  ·  in sequence", C[1])

    for ax in (a1, a2):
        ax.set_xlim(0, 7.3); ax.set_ylim(0.3, 4.9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(False)
    head(fig, "Boosting and bagging combine trees in opposite ways")
    save(fig, "diag_boosting_vs_bagging.png")


def diagram_adaboost_rounds():
    """Reweighting across three rounds, then the weighted vote."""
    C = theme.C
    rng = np.random.default_rng(21)
    n = 22
    pts = rng.uniform(0.1, 0.9, size=(n, 2))
    lab = (pts[:, 0] + pts[:, 1] > 1.0)

    cuts = [("loudness", 0.46, "v"), ("tempo", 0.52, "h"), ("loudness", 0.70, "v")]
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.9))
    fig.subplots_adjust(left=0.02, right=0.985, top=0.72, bottom=0.14, wspace=0.09)

    w = np.ones(n) / n
    for r, (ax, (var, thr, orient)) in enumerate(zip(axes[:3], cuts)):
        sizes = 26 + w * n * 150
        ax.scatter(pts[~lab, 0], pts[~lab, 1], s=sizes[~lab], color=C[0],
                   alpha=0.85, linewidths=0)
        ax.scatter(pts[lab, 0], pts[lab, 1], s=sizes[lab], color=C[1],
                   alpha=0.85, linewidths=0)
        if orient == "v":
            ax.axvline(thr, color=theme.ACCENT, linewidth=2.2)
            pred = pts[:, 0] > thr
        else:
            ax.axhline(thr, color=theme.ACCENT, linewidth=2.2)
            pred = pts[:, 1] > thr
        wrong = pred != lab
        ax.scatter(pts[wrong, 0], pts[wrong, 1], s=sizes[wrong] + 70,
                   facecolor="none", edgecolor=theme.INK_SOFT, linewidth=1.3)
        err = max(w[wrong].sum(), 1e-6)
        alpha = 0.5 * np.log((1 - err) / err)
        ax.set_title(f"Round {r+1}   α = {alpha:.2f}", loc="left", fontsize=10.2,
                     color=theme.INK)
        ax.set_xlabel(f"split on {var}", fontsize=8.6)
        # Rows the stump got wrong are weighted up for the next round.
        w = w * np.exp(alpha * np.where(wrong, 1, -1))
        w /= w.sum()

    a4 = axes[3]
    a4.scatter(pts[~lab, 0], pts[~lab, 1], s=30, color=C[0], alpha=0.9, linewidths=0)
    a4.scatter(pts[lab, 0], pts[lab, 1], s=30, color=C[1], alpha=0.9, linewidths=0)
    for thr, orient, al in [(0.46, "v", 0.9), (0.52, "h", 0.6), (0.70, "v", 0.35)]:
        if orient == "v":
            a4.axvline(thr, color=theme.ACCENT, linewidth=2.2, alpha=al)
        else:
            a4.axhline(thr, color=theme.ACCENT, linewidth=2.2, alpha=al)
    a4.set_title("Final  ·  weighted vote", loc="left", fontsize=10.2,
                 color=theme.ACCENT)
    a4.set_xlabel("all stumps, each weighted by its α", fontsize=8.6)

    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values(): sp.set_edgecolor(theme.GRID)

    fig.text(0.5, 0.035, "Circled points were misclassified. They grow for the next "
             "round, so the following stump is pulled toward them.",
             ha="center", fontsize=9, color=theme.INK_SOFT)
    head(fig, "How AdaBoost reweights the rows and combines its stumps")
    save(fig, "diag_adaboost_rounds.png")


if __name__ == "__main__":
    print("Building Module 4 concept diagrams")
    for mode in theme.MODES:
        theme.set_theme(mode)
        print(f"  --- {mode} ---")
        diagram_linear_vs_logistic()
        diagram_sigmoid()
        diagram_boosting_vs_bagging()
        diagram_adaboost_rounds()
    print("Done.")
