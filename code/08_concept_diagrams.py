"""
CSCI 5612 - Project Part 2
Script 08: Explanatory diagrams for the Clustering and PCA overviews.

These illustrate how the methods work using invented points, so the Overview
sections can explain the idea before the Results sections show what it found.
The prose explaining each diagram lives in the page caption, not inside the
image, which keeps the drawings uncluttered.

Writes ../docs/images/diag_*.png  (light and dark renderings)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, Arc
from scipy.spatial.distance import pdist

import theme

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "..", "docs", "images")


def save(fig, name):
    out = theme.out_name(name)
    fig.savefig(os.path.join(IMG, out)); plt.close(fig)
    print(f"    saved {out}")


def frame(ax, title, color=None):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_edgecolor(theme.GRID)
    ax.set_title(title, loc="left", fontsize=12, pad=10,
                 color=color or theme.INK)


def head(fig, text):
    fig.suptitle(text, x=0.012, ha="left", fontsize=14, fontweight="bold", y=0.975)


# ---------------------------------------------------------------------------

def diagram_clustering_types():
    rng = np.random.default_rng(7)
    centres = [(-1.7, 1.3), (1.8, 1.1), (0.1, -1.7)]
    pts, lab = [], []
    for i, c in enumerate(centres):
        pts.append(rng.normal(c, 0.60, size=(34, 2))); lab += [i] * 34
    P, lab = np.vstack(pts), np.array(lab)
    C = theme.C

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 5.0))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.86, bottom=0.04, wspace=0.11)

    for i in range(3):
        s = lab == i
        a1.scatter(P[s, 0], P[s, 1], s=26, color=C[i], alpha=0.75, linewidths=0)
    for i, (cx, cy) in enumerate(centres):
        a1.scatter([cx], [cy], marker="X", s=210, color=C[i],
                   edgecolor=theme.BG, linewidth=2.2, zorder=5)
    a1.annotate("centres chosen first,\nthen moved until stable",
                xy=(0, -3.95), ha="center", fontsize=9, color=theme.INK_SOFT)
    frame(a1, "Partitional  ·  k-means")

    a2.scatter(P[:, 0], P[:, 1], s=26, color=theme.INK_SOFT, alpha=0.55, linewidths=0)
    for i, (cx, cy) in enumerate(centres):
        a2.add_patch(Ellipse((cx, cy), 2.4, 2.2, facecolor="none",
                             edgecolor=C[i], linewidth=1.9))
    a2.add_patch(Ellipse((0.05, 1.2), 6.4, 3.1, facecolor="none",
                         edgecolor=theme.ACCENT, linewidth=1.6, linestyle="--"))
    a2.add_patch(Ellipse((0.1, 0.0), 7.9, 6.4, facecolor="none",
                         edgecolor=theme.ACCENT, linewidth=1.4, linestyle=":"))
    a2.annotate("merge 2", xy=(2.95, 2.5), fontsize=8.6, color=theme.ACCENT,
                fontweight="bold")
    a2.annotate("merge 3", xy=(3.55, 3.05), fontsize=8.6, color=theme.ACCENT,
                fontweight="bold")
    a2.annotate("cut the tree at any height\nto get any number of groups",
                xy=(0, -3.95), ha="center", fontsize=9, color=theme.INK_SOFT)
    frame(a2, "Hierarchical  ·  agglomerative")

    for ax in (a1, a2):
        ax.set_xlim(-4.3, 4.3); ax.set_ylim(-4.6, 3.7)
    head(fig, "Two ways of grouping the same points")
    save(fig, "diag_clustering_types.png")


def diagram_distance_metrics():
    C = theme.C
    A, B, D = np.array([1.1, 2.5]), np.array([3.1, 7.0]), np.array([5.6, 1.5])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 5.0))
    fig.subplots_adjust(left=0.045, right=0.985, top=0.86, bottom=0.09, wspace=0.13)

    for pt, col, name, off in [(A, C[0], "Quiet mix", (10, -14)),
                               (B, C[1], "Loud mix", (10, 4)),
                               (D, C[2], "Different mix", (-8, -18))]:
        a1.scatter(*pt, s=140, color=col, zorder=5,
                   edgecolor=theme.BG, linewidth=1.8)
        a1.annotate(name, xy=pt, xytext=off, textcoords="offset points",
                    fontsize=9.6, color=col, fontweight="bold")
    a1.plot([A[0], B[0]], [A[1], B[1]], color=theme.INK_SOFT,
            linewidth=1.8, linestyle="--")
    a1.annotate("counted as\nfar apart", xy=(1.55, 4.9), fontsize=9,
                color=theme.INK_SOFT, ha="right")
    frame(a1, "Euclidean  ·  how far apart")

    for pt, col, name, off in [(A, C[0], "Quiet mix", (12, -10)),
                               (B, C[1], "Loud mix", (10, 2)),
                               (D, C[2], "Different mix", (6, -4))]:
        a2.add_patch(FancyArrowPatch((0, 0), tuple(pt), color=col, linewidth=2.2,
                                     arrowstyle="-|>", mutation_scale=16, zorder=4))
        a2.annotate(name, xy=pt, xytext=off, textcoords="offset points",
                    fontsize=9.6, color=col, fontweight="bold")
    ang_a = np.degrees(np.arctan2(A[1], A[0]))
    ang_d = np.degrees(np.arctan2(D[1], D[0]))
    a2.add_patch(Arc((0, 0), 4.4, 4.4, theta1=ang_d, theta2=ang_a,
                     color=theme.ACCENT, linewidth=2.0))
    a2.annotate("this angle is\nthe distance", xy=(3.3, 2.15), fontsize=9,
                color=theme.ACCENT, ha="left", fontweight="bold")
    a2.annotate("same direction,\ndistance near zero", xy=(1.1, 7.4), fontsize=9,
                color=theme.INK_SOFT, ha="left")
    frame(a2, "Cosine  ·  which direction")

    for ax in (a1, a2):
        ax.set_xlim(-0.3, 8.6); ax.set_ylim(-0.3, 8.6)
        ax.set_xlabel("Loudness", fontsize=9.6, labelpad=2)
        ax.set_ylabel("Energy", fontsize=9.6, labelpad=2)
    head(fig, "Why the distance measure changes the answer")
    save(fig, "diag_distance_metrics.png")


def diagram_pca_concept():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, size=(240, 2)) @ np.array([[2.0, 0.0], [1.25, 0.80]])
    base -= base.mean(axis=0)
    C = theme.C

    vals, vecs = np.linalg.eigh(np.cov(base.T))
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # Sign is arbitrary in an eigen-decomposition; point PC1 to the right.
    for i in range(2):
        if vecs[0, i] < 0:
            vecs[:, i] *= -1

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 5.0))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.86, bottom=0.05, wspace=0.11)

    a1.scatter(base[:, 0], base[:, 1], s=20, color=theme.GRID, alpha=0.95, linewidths=0)
    for i, (lab, col, off) in enumerate([("PC1", C[1], (8, 10)), ("PC2", C[0], (8, 6))]):
        v = vecs[:, i] * np.sqrt(vals[i]) * 2.0
        a1.add_patch(FancyArrowPatch((0, 0), tuple(v), color=col, linewidth=2.8,
                                     arrowstyle="-|>", mutation_scale=18, zorder=5))
        a1.annotate(f"{lab}  ({vals[i]/vals.sum()*100:.0f}%)", xy=v, xytext=off,
                    textcoords="offset points", fontsize=10, color=col,
                    fontweight="bold")
    frame(a1, "Eigenvectors point along the spread")

    pc1 = vecs[:, 0]
    proj = np.outer(base @ pc1, pc1)
    lim = np.abs(base).max() * 1.08
    a2.plot([-pc1[0]*lim, pc1[0]*lim], [-pc1[1]*lim, pc1[1]*lim],
            color=C[1], linewidth=2.2, zorder=3)
    a2.scatter(base[:, 0], base[:, 1], s=18, color=theme.GRID, alpha=0.8, linewidths=0)
    for i in range(0, len(base), 4):
        a2.plot([base[i, 0], proj[i, 0]], [base[i, 1], proj[i, 1]],
                color=theme.INK_SOFT, linewidth=0.6, alpha=0.5)
    a2.scatter(proj[:, 0], proj[:, 1], s=15, color=C[1], alpha=0.9,
               linewidths=0, zorder=4)
    frame(a2, "Reducing dimensions is projecting")

    for ax in (a1, a2):
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim * 0.62, lim * 0.62)
    head(fig, "What PCA is doing")
    save(fig, "diag_pca_concept.png")


def diagram_dimensionality():
    C = theme.C
    rng = np.random.default_rng(11)
    dims = [2, 5, 10, 25, 50, 100, 200]
    spread = []
    for d in dims:
        X = rng.normal(size=(260, d))
        dd = pdist(X)
        spread.append(dd.std() / dd.mean())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.6))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.84, bottom=0.17, wspace=0.26)

    a1.plot(dims, spread, color=C[1], linewidth=2.4, marker="o", markersize=7,
            markeredgecolor=theme.BG, markeredgewidth=1.6)
    a1.set_xscale("log"); a1.set_xticks(dims, [str(d) for d in dims])
    a1.set_xlabel("Number of dimensions"); a1.set_ylabel("Spread of distances")
    a1.set_title("Distances stop meaning much", loc="left", fontsize=12, pad=10)
    for sp in ("top", "right"): a1.spines[sp].set_visible(False)

    labels = ["1", "2", "3", "5", "10\n(all)"]
    kept = [21.8, 37.2, 48.6, 68.3, 100.0]
    bars = a2.bar(range(len(kept)), kept, color=C[6], width=0.6,
                  edgecolor=theme.BG, linewidth=1.8)
    bars[1].set_color(C[1])
    for i, v in enumerate(kept):
        a2.text(i, v + 2.6, f"{v:.0f}%", ha="center", fontsize=9.4,
                fontweight="bold", color=theme.INK_SOFT)
    a2.set_xticks(range(len(kept)), labels)
    a2.set_ylim(0, 118); a2.grid(False)
    a2.set_xlabel("Components kept"); a2.set_ylabel("Variance retained (%)")
    a2.set_title("But compression has a price", loc="left", fontsize=12, pad=10)
    for sp in ("top", "right"): a2.spines[sp].set_visible(False)

    head(fig, "The problem dimensionality reduction is for")
    save(fig, "diag_dimensionality.png")


if __name__ == "__main__":
    print("Building concept diagrams")
    for mode in theme.MODES:
        theme.set_theme(mode)
        print(f"  --- {mode} ---")
        diagram_clustering_types()
        diagram_distance_metrics()
        diagram_pca_concept()
        diagram_dimensionality()
    print("Done.")
