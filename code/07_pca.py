"""
CSCI 5612 - Project Part 2
Script 07: Principal Component Analysis (and its relationship to SVD).

Reads  ../data/clean/spotify_songs_clean.csv
Writes ../data/clean/pca_input.csv          (the standardised numeric matrix)
       ../data/clean/pca_components.csv     (loadings: variables x components)
       ../data/clean/pca_projection.csv     (songs in component space)
       ../data/clean/pca_metrics.json       (every number quoted on the site)
       ../docs/images/fig2x_*.png

Run: python 07_pca.py
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import theme

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "data", "clean")
IMG = os.path.join(HERE, "..", "docs", "images")
SEED = 5612

FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "duration_min"]
NICE = {f: f.replace("_", " ").capitalize() for f in FEATURES}
NICE["duration_min"] = "Duration"

M = {}


def finish(ax, title, subtitle=None, xlabel=None, ylabel=None):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"{title}\n" if subtitle else title, loc="left")
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.8,
                color=theme.INK_SOFT, va="bottom")
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)


def save(fig, name):
    out = theme.out_name(name)
    fig.savefig(os.path.join(IMG, out)); plt.close(fig)
    print(f"    saved {out}")


def main():
    print("=" * 70)
    print("CSCI 5612 Project Part 2 - PCA")
    print("=" * 70)

    df = pd.read_csv(os.path.join(CLEAN, "spotify_songs_clean.csv"))
    X = df[FEATURES]
    assert X.isna().sum().sum() == 0

    # PCA needs numeric data on a common scale. Without standardising, tempo
    # (values near 120) would swamp danceability (values under 1) purely
    # because of the units each happens to be measured in.
    Xs = StandardScaler().fit_transform(X)
    pd.DataFrame(Xs, columns=FEATURES).round(5).to_csv(
        os.path.join(CLEAN, "pca_input.csv"), index=False)
    print(f"  standardised {Xs.shape[0]:,} songs x {Xs.shape[1]} numeric features")

    pca = PCA(random_state=SEED).fit(Xs)
    evr = pca.explained_variance_ratio_
    eig = pca.explained_variance_
    cum = np.cumsum(evr)

    print("\n  eigenvalues and variance explained")
    for i, (e, v, c) in enumerate(zip(eig, evr, cum), start=1):
        print(f"    PC{i:<2} eigenvalue {e:5.3f}   {v*100:5.2f}%   cumulative {c*100:5.1f}%")

    # PCA is an eigen-decomposition of the covariance matrix, and it is also a
    # singular value decomposition of the centred data. Confirm they agree.
    U, S, Vt = np.linalg.svd(Xs - Xs.mean(axis=0), full_matrices=False)
    eig_from_svd = (S ** 2) / (len(Xs) - 1)
    agree = float(np.max(np.abs(eig_from_svd - eig)))
    print(f"\n  SVD check: largest difference between SVD and PCA eigenvalues = {agree:.2e}")

    n95 = int(np.searchsorted(cum, 0.95) + 1)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    kaiser = int((eig > 1).sum())
    print(f"  components for 90% variance: {n90};  95%: {n95};  eigenvalue > 1: {kaiser}")

    P = pca.transform(Xs)
    load = pd.DataFrame(pca.components_.T, index=FEATURES,
                        columns=[f"PC{i+1}" for i in range(len(FEATURES))])
    load.round(4).to_csv(os.path.join(CLEAN, "pca_components.csv"))
    proj = pd.DataFrame(P[:, :3], columns=["PC1", "PC2", "PC3"]).round(4)
    proj["playlist_genre"] = df["playlist_genre"].values
    proj["track_popularity"] = df["track_popularity"].values
    proj.to_csv(os.path.join(CLEAN, "pca_projection.csv"), index=False)

    M.update({
        "n_songs": int(len(df)), "n_features": len(FEATURES),
        "eigenvalues": [round(float(e), 4) for e in eig],
        "explained": [round(float(v), 4) for v in evr],
        "cumulative": [round(float(c), 4) for c in cum],
        "pc1": round(float(evr[0]) * 100, 1), "pc2": round(float(evr[1]) * 100, 1),
        "pc3": round(float(evr[2]) * 100, 1),
        "first_two": round(float(cum[1]) * 100, 1),
        "first_three": round(float(cum[2]) * 100, 1),
        "n_for_90": n90, "n_for_95": n95, "kaiser": kaiser,
        "svd_max_diff": agree,
        "top_pc1": load.PC1.abs().sort_values(ascending=False).head(4).index.tolist(),
        "top_pc2": load.PC2.abs().sort_values(ascending=False).head(4).index.tolist(),
        "loadings": {c: {f: round(float(load.loc[f, c]), 3) for f in FEATURES}
                     for c in ["PC1", "PC2", "PC3"]},
    })

    rng = np.random.default_rng(SEED)
    draw = rng.choice(len(P), size=7000, replace=False)
    genres = ["pop", "rap", "rock", "latin", "r&b", "edm"]

    for mode in theme.MODES:
        theme.set_theme(mode)
        C, ACC = theme.C, theme.ACCENT
        print(f"\n  --- figures ({mode}) ---")

        # 26. eigenvalues and cumulative variance, two panels (never two y-axes)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.4, 6.6), sharex=True)
        xs = np.arange(1, len(eig) + 1)
        a1.bar(xs, eig, color=C[6], width=0.62, edgecolor=theme.BG, linewidth=1.6)
        a1.axhline(1.0, color=C[1], linestyle="--", linewidth=1.6)
        a1.annotate("eigenvalue = 1", xy=(len(eig), 1.0), xytext=(-4, 7),
                    textcoords="offset points", ha="right", color=C[1],
                    fontweight="bold", fontsize=9.3)
        for x, e in zip(xs, eig):
            a1.text(x, e + 0.06, f"{e:.2f}", ha="center", fontsize=8.6,
                    color=theme.INK_SOFT)
        a1.set_ylim(0, max(eig) * 1.2)
        finish(a1, "Only the first few components carry real weight",
               "Eigenvalue of each principal component; a value above 1 means the "
               "component explains more than a single original variable would",
               None, "Eigenvalue")
        a2.plot(xs, cum * 100, color=C[0], linewidth=2.4, marker="o", markersize=7,
                markeredgecolor=theme.BG, markeredgewidth=1.6)
        a2.axhline(95, color=C[1], linestyle=":", linewidth=1.6)
        a2.annotate(f"95% reached at {n95} components", xy=(n95, 95),
                    xytext=(8, -16), textcoords="offset points", color=C[1],
                    fontweight="bold", fontsize=9.3)
        a2.set_xticks(xs); a2.set_ylim(0, 105)
        finish(a2, "Reaching most of the variance takes most of the components",
               "Cumulative share of total variance captured as components are added",
               "Principal component", "Cumulative variance (%)")
        fig.tight_layout(h_pad=2.4)
        save(fig, "fig26_scree_cumulative.png")

        # 27. biplot: where the variables point in component space
        fig, ax = plt.subplots(figsize=(8.2, 6.4))
        ax.scatter(P[draw, 0], P[draw, 1], s=3, color=theme.GRID, alpha=0.55,
                   linewidths=0, rasterized=True)
        scale = np.abs(P[draw, :2]).max() * 0.78
        # A few loadings point almost the same way, so nudge those labels apart.
        nudge = {"instrumentalness": (-0.085, -0.045), "duration_min": (0.085, 0.030),
                 "liveness": (-0.055, 0.045), "tempo": (0.060, -0.040),
                 "speechiness": (-0.075, 0.0), "danceability": (0.0, 0.045)}
        for f in FEATURES:
            vx, vy = load.loc[f, "PC1"], load.loc[f, "PC2"]
            ax.arrow(0, 0, vx * scale, vy * scale, color=ACC, width=0.012,
                     head_width=0.17, length_includes_head=True, alpha=0.95)
            dx, dy = nudge.get(f, (0.0, 0.0))
            ax.text((vx + dx) * scale * 1.16, (vy + dy) * scale * 1.16, NICE[f],
                    fontsize=9.3, color=theme.INK, ha="center", va="center",
                    fontweight="bold")
        ax.axhline(0, color=theme.INK_SOFT, linewidth=0.8, alpha=0.5)
        ax.axvline(0, color=theme.INK_SOFT, linewidth=0.8, alpha=0.5)
        ax.grid(False)
        finish(ax, "Loud, energetic and acoustic point in opposite directions",
               f"Songs on the first two components, with each original variable "
               f"drawn as an arrow. Together these axes hold {cum[1]*100:.0f}% of the variance.",
               f"PC1 ({evr[0]*100:.1f}% of variance)",
               f"PC2 ({evr[1]*100:.1f}% of variance)")
        save(fig, "fig27_pca_biplot.png")

        # 28. loadings table as a heatmap
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        sub = load[["PC1", "PC2", "PC3"]]
        im = ax.imshow(sub.values, cmap=theme.DIV_CMAP, vmin=-0.65, vmax=0.65,
                       aspect="auto")
        ax.set_xticks(range(3), [f"PC{i+1}\n({evr[i]*100:.1f}%)" for i in range(3)])
        ax.set_yticks(range(len(FEATURES)), [NICE[f] for f in FEATURES])
        ax.grid(False)
        for i in range(len(FEATURES)):
            for j in range(3):
                v = sub.values[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8.8,
                        color="#ffffff" if abs(v) > 0.38 else theme.INK)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
        cb.set_label("Loading", color=theme.INK_SOFT, fontsize=9.3)
        cb.outline.set_visible(False)
        ax.set_title("What each component is made of\n", loc="left")
        ax.text(0, 1.02, "How strongly each original variable contributes to the "
                "first three components", transform=ax.transAxes, fontsize=9.8,
                color=theme.INK_SOFT, va="bottom")
        save(fig, "fig28_pca_loadings.png")

        # 29. does the reduced space separate genre?
        fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.6), sharex=True, sharey=True)
        for ax, gname in zip(axes.ravel(), genres):
            sel = df["playlist_genre"].values[draw] == gname
            ax.scatter(P[draw, 0], P[draw, 1], s=2.5, color=theme.GRID, alpha=0.5,
                       linewidths=0, rasterized=True)
            ax.scatter(P[draw][sel, 0], P[draw][sel, 1], s=3.5,
                       color=C[genres.index(gname) % len(C)], alpha=0.5,
                       linewidths=0, rasterized=True)
            ax.set_title(gname.upper(), loc="left", fontsize=10.5,
                         color=C[genres.index(gname) % len(C)])
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.suptitle("Reducing to two dimensions does not pull the genres apart",
                     x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.03)
        fig.text(0.01, 0.967, "Each panel highlights one genre in the same "
                 "two-component space used above", fontsize=9.8,
                 color=theme.INK_SOFT, ha="left")
        fig.supxlabel("PC1", fontsize=10.5, color=theme.INK_SOFT)
        fig.supylabel("PC2", fontsize=10.5, color=theme.INK_SOFT)
        fig.tight_layout(rect=[0.01, 0.01, 1, 0.94])
        save(fig, "fig29_pca_by_genre.png")

    with open(os.path.join(CLEAN, "pca_metrics.json"), "w") as fh:
        json.dump(M, fh, indent=2)

    print("\n" + "=" * 70)
    print(f"PCA complete. PC1 {evr[0]*100:.1f}%, PC1+PC2 {cum[1]*100:.1f}%, "
          f"{n95} components for 95%")
    print("=" * 70)


if __name__ == "__main__":
    main()
