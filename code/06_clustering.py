"""
CSCI 5612 - Project Part 2
Script 06: Clustering.

Two families of clustering are run on the same unlabelled, numeric slice of the
song data:

  * partitional  - k-means at several values of k, with k chosen by silhouette
  * hierarchical - agglomerative, using COSINE distance

Reads  ../data/clean/spotify_songs_clean.csv
Writes ../data/clean/clustering_input.csv        (the unlabelled numeric sample)
       ../data/clean/clustering_results.csv      (cluster assignments)
       ../data/clean/clustering_metrics.json     (every number quoted on the site)
       ../docs/images/fig2x_*.png                (light and dark renderings)

Run: python 06_clustering.py
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import theme

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "data", "clean")
IMG = os.path.join(HERE, "..", "docs", "images")
SEED = 5612

# Clustering sees ONLY these. No genre, no popularity, no title, no artist.
FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "duration_min"]

K_VALUES = [2, 3, 4, 5, 6, 7, 8]
SIL_SAMPLE = 6000      # silhouette is O(n^2); evaluate on a large random sample
HCLUST_SAMPLE = 1500   # a full 25k x 25k distance matrix will not fit in memory

M = {}                 # every number that ends up on the website


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
    fig.savefig(os.path.join(IMG, out))
    plt.close(fig)
    print(f"    saved {out}")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare():
    df = pd.read_csv(os.path.join(CLEAN, "spotify_songs_clean.csv"))
    print(f"  loaded {len(df):,} songs")

    X = df[FEATURES].copy()
    assert X.isna().sum().sum() == 0, "clustering input must have no missing values"
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in FEATURES), "all columns must be numeric"

    # Standardise: tempo runs to ~240 and loudness is negative, so without this
    # those two would dominate every distance calculation.
    Xs = StandardScaler().fit_transform(X)
    print(f"  standardised {Xs.shape[1]} numeric features, no labels included")

    # The exact unlabelled table handed to the algorithms, saved so the website
    # can link to the precise input rather than a description of it.
    out = pd.DataFrame(Xs, columns=FEATURES).round(5)
    out.to_csv(os.path.join(CLEAN, "clustering_input.csv"), index=False)

    M["n_songs"] = int(len(df))
    M["n_features"] = len(FEATURES)
    M["features"] = FEATURES
    return df, X, Xs


# ---------------------------------------------------------------------------
# k-means
# ---------------------------------------------------------------------------

def run_kmeans(Xs):
    print("\n  k-means across k =", K_VALUES)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xs), size=min(SIL_SAMPLE, len(Xs)), replace=False)

    rows, labels_by_k = [], {}
    for k in K_VALUES:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        lab = km.fit_predict(Xs)
        labels_by_k[k] = lab
        sil = silhouette_score(Xs[idx], lab[idx])
        rows.append({"k": k, "silhouette": round(float(sil), 4),
                     "inertia": round(float(km.inertia_), 1)})
        print(f"    k={k}  silhouette={sil:.4f}  inertia={km.inertia_:,.0f}")

    res = pd.DataFrame(rows)
    best = int(res.loc[res.silhouette.idxmax(), "k"])
    print(f"    best k by silhouette: {best}")

    M["kmeans"] = rows
    M["best_k"] = best
    M["best_silhouette"] = float(res.silhouette.max())
    M["sil_sample"] = int(len(idx))
    return labels_by_k, res, best, idx


# ---------------------------------------------------------------------------
# Hierarchical, with cosine distance
# ---------------------------------------------------------------------------

def run_hclust(Xs, best_k):
    print(f"\n  hierarchical clustering on a {HCLUST_SAMPLE:,}-song sample, cosine distance")
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(Xs), size=HCLUST_SAMPLE, replace=False)
    Xh = Xs[sub]

    d = pdist(Xh, metric="cosine")
    Z = linkage(d, method="average")
    print(f"    cosine distance: min {d.min():.3f}, median {np.median(d):.3f}, max {d.max():.3f}")

    # What k does the hierarchy itself suggest? The largest jump in merge
    # height is the usual reading of a dendrogram.
    heights = Z[:, 2]
    last = heights[-10:]
    jumps = np.diff(last)
    suggested = int(10 - np.argmax(jumps))
    print(f"    largest merge gap suggests k = {suggested}")

    h_labels = fcluster(Z, t=best_k, criterion="maxclust")
    M["hclust_sample"] = HCLUST_SAMPLE
    M["hclust_metric"] = "cosine"
    M["hclust_linkage"] = "average"
    M["hclust_suggested_k"] = suggested
    M["cosine_median"] = float(np.median(d))
    return Z, sub, h_labels


# ---------------------------------------------------------------------------

def main():
    os.makedirs(IMG, exist_ok=True)
    print("=" * 70)
    print("CSCI 5612 Project Part 2 - clustering")
    print("=" * 70)

    df, X, Xs = prepare()
    labels_by_k, res, best_k, sil_idx = run_kmeans(Xs)
    Z, sub, h_labels = run_hclust(Xs, best_k)

    # How well do the k-means clusters line up with the genre labels the
    # algorithm never saw? This is the question the whole tab is built around.
    ari_genre = adjusted_rand_score(df["playlist_genre"], labels_by_k[best_k])
    ari_hk = adjusted_rand_score(h_labels, labels_by_k[best_k][sub])
    print(f"\n  agreement with genre labels (ARI): {ari_genre:.4f}")
    print(f"  agreement hclust vs k-means  (ARI): {ari_hk:.4f}")
    M["ari_genre"] = float(ari_genre)
    M["ari_hclust_kmeans"] = float(ari_hk)

    # Two components purely for drawing the clusters on a page.
    pca2 = PCA(n_components=2, random_state=SEED)
    P = pca2.fit_transform(Xs)
    M["plot_pca_var"] = float(pca2.explained_variance_ratio_.sum())

    # Cluster profiles in the original units, so each cluster can be described.
    prof = X.copy()
    prof["cluster"] = labels_by_k[best_k]
    profile = prof.groupby("cluster").mean()
    sizes = pd.Series(labels_by_k[best_k]).value_counts().sort_index()
    M["cluster_sizes"] = {int(k): int(v) for k, v in sizes.items()}
    M["cluster_profile"] = {int(c): {f: round(float(v), 3) for f, v in row.items()}
                            for c, row in profile.iterrows()}

    # Save assignments alongside the labels we withheld, for the write-up.
    out = df[["track_name", "track_artist", "playlist_genre", "track_popularity"]].copy()
    out["kmeans_cluster"] = labels_by_k[best_k]
    out.to_csv(os.path.join(CLEAN, "clustering_results.csv"), index=False)

    # ---- figures, rendered once per site theme ----------------------------
    for mode in theme.MODES:
        theme.set_theme(mode)
        C, ACC = theme.C, theme.ACCENT
        print(f"\n  --- figures ({mode}) ---")

        # 20. silhouette score against k
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        ax.plot(res.k, res.silhouette, color=C[0], linewidth=2.4, marker="o",
                markersize=7, markeredgecolor=theme.BG, markeredgewidth=1.6)
        bi = res.silhouette.idxmax()
        ax.scatter([res.k[bi]], [res.silhouette[bi]], s=190, facecolor="none",
                   edgecolor=C[1], linewidth=2.2, zorder=5)
        ax.annotate(f"best: k = {best_k}\nsilhouette {res.silhouette[bi]:.3f}",
                    xy=(res.k[bi], res.silhouette[bi]), xytext=(18, -6),
                    textcoords="offset points", va="top", color=C[1],
                    fontweight="bold", fontsize=9.6)
        ax.set_xticks(K_VALUES)
        # headroom so the marker and its callout never reach the subtitle
        lo, hi = res.silhouette.min(), res.silhouette.max()
        ax.set_ylim(lo - (hi - lo) * 0.18, hi + (hi - lo) * 0.16)
        finish(ax, "The silhouette score peaks at a small number of clusters",
               f"Average silhouette width for k-means at each k, measured on "
               f"{len(sil_idx):,} sampled songs",
               "Number of clusters (k)", "Average silhouette width")
        save(fig, "fig20_silhouette_by_k.png")

        # 21. per-song silhouette at the chosen k
        lab = labels_by_k[best_k][sil_idx]
        sv = silhouette_samples(Xs[sil_idx], lab)
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        y = 0
        for c in range(best_k):
            vals = np.sort(sv[lab == c])
            ax.fill_betweenx(np.arange(y, y + len(vals)), 0, vals,
                             facecolor=C[c % len(C)], edgecolor="none", alpha=0.9)
            ax.text(-0.045, y + len(vals) / 2, f"cluster {c}", va="center",
                    ha="right", fontsize=9, color=theme.INK_SOFT)
            y += len(vals) + 90
        ax.axvline(sv.mean(), color=C[1], linestyle="--", linewidth=2)
        ax.set_ylim(-60, y * 1.10)
        ax.annotate(f"mean {M['best_silhouette']:.3f}", xy=(sv.mean(), y * 1.055),
                    xytext=(7, 0), textcoords="offset points", va="center",
                    color=C[1], fontweight="bold", fontsize=9.4)
        ax.set_yticks([])
        ax.grid(False)
        finish(ax, f"Most songs sit only weakly inside their cluster",
               f"Silhouette width of each sampled song at k = {best_k}; "
               f"values near zero mean a song is close to a boundary",
               "Silhouette width", None)
        save(fig, "fig21_silhouette_detail.png")

        # 22. the clusters themselves, at three values of k
        show_k = [3, best_k, 6] if best_k not in (3, 6) else [2, 3, 6]
        show_k = sorted(set(show_k))
        fig, axes = plt.subplots(1, len(show_k), figsize=(4.1 * len(show_k), 4.3),
                                 sharex=True, sharey=True)
        rng = np.random.default_rng(SEED)
        draw = rng.choice(len(P), size=7000, replace=False)
        for ax, k in zip(np.atleast_1d(axes), show_k):
            lb = labels_by_k[k][draw]
            for c in range(k):
                sel = lb == c
                ax.scatter(P[draw][sel, 0], P[draw][sel, 1], s=3.5,
                           color=C[c % len(C)], alpha=0.45, linewidths=0,
                           rasterized=True)
            ax.set_title(f"k = {k}" + ("   (chosen)" if k == best_k else ""),
                         loc="left", fontsize=11,
                         color=ACC if k == best_k else theme.INK)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.suptitle("The clusters carve a single cloud rather than finding separate islands",
                     x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.06)
        fig.text(0.01, 0.995,
                 f"k-means at three values of k, drawn on the first two principal "
                 f"components ({pca2.explained_variance_ratio_.sum()*100:.0f}% of variance)",
                 fontsize=9.8, color=theme.INK_SOFT, ha="left")
        fig.supxlabel("Principal component 1", fontsize=10.5, color=theme.INK_SOFT)
        fig.supylabel("Principal component 2", fontsize=10.5, color=theme.INK_SOFT)
        fig.tight_layout(rect=[0.01, 0.01, 1, 0.94])
        save(fig, "fig22_kmeans_clusters.png")

        # 23. what each cluster actually is
        z = (profile - X.mean()) / X.std()
        fig, ax = plt.subplots(figsize=(9.0, 0.62 * best_k + 2.6))
        im = ax.imshow(z.values, cmap=theme.DIV_CMAP, vmin=-1.6, vmax=1.6,
                       aspect="auto")
        ax.set_xticks(range(len(FEATURES)),
                      [f.replace("_", " ").capitalize() for f in FEATURES],
                      rotation=38, ha="right")
        ax.set_yticks(range(best_k),
                      [f"Cluster {c}\n({sizes[c]:,} songs)" for c in range(best_k)])
        ax.grid(False)
        for i in range(best_k):
            for j in range(len(FEATURES)):
                v = z.values[i, j]
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if abs(v) > 0.9 else theme.INK)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("Standard deviations from the overall average",
                     color=theme.INK_SOFT, fontsize=9)
        cb.outline.set_visible(False)
        ax.set_title("Each cluster is a recognisable kind of record\n", loc="left")
        ax.text(0, 1.02, f"How the average song in each cluster differs from the "
                f"catalogue average, at k = {best_k}",
                transform=ax.transAxes, fontsize=9.8, color=theme.INK_SOFT, va="bottom")
        save(fig, "fig23_cluster_profiles.png")

        # 24. dendrogram
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        dendrogram(Z, truncate_mode="lastp", p=28, ax=ax,
                   color_threshold=Z[-(best_k - 1), 2],
                   above_threshold_color=theme.INK_SOFT,
                   leaf_rotation=90, leaf_font_size=8)
        ax.axhline(Z[-(best_k - 1), 2], color=C[1], linestyle="--", linewidth=1.8)
        ax.annotate(f"cut for k = {best_k}", xy=(0.012, Z[-(best_k - 1), 2]),
                    xycoords=("axes fraction", "data"), xytext=(0, 6),
                    textcoords="offset points", color=C[1], fontweight="bold",
                    fontsize=9.4)
        ax.grid(False)
        finish(ax, "The hierarchy suggests three groups where k-means preferred two",
               f"Average-linkage dendrogram on {HCLUST_SAMPLE:,} songs using cosine "
               f"distance. Merge heights rise gradually, so where to cut is a judgement call.",
               "Groups of songs (leaf size in brackets)", "Cosine distance at merge")
        save(fig, "fig24_dendrogram.png")

        # 25. do the clusters correspond to genre?
        ct = pd.crosstab(pd.Series(labels_by_k[best_k], name="cluster"),
                         df["playlist_genre"])
        ct = ct.div(ct.sum(axis=1), axis=0) * 100
        order = ["pop", "rap", "rock", "latin", "r&b", "edm"]
        ct = ct[[c for c in order if c in ct.columns]]
        fig, ax = plt.subplots(figsize=(8.6, 0.55 * best_k + 2.8))
        left = np.zeros(len(ct))
        for i, gname in enumerate(ct.columns):
            v = ct[gname].values
            ax.barh(range(len(ct)), v, left=left, height=0.62,
                    color=C[i % len(C)], edgecolor=theme.BG, linewidth=1.5,
                    label=gname.upper())
            left += v
        ax.set_yticks(range(len(ct)), [f"Cluster {c}" for c in ct.index])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.grid(False)
        ax.legend(ncols=6, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  fontsize=9)
        finish(ax, "No cluster belongs to a single genre",
               f"Genre make-up of each k-means cluster. Agreement with the genre "
               f"labels is only {ari_genre:.3f} on a 0-to-1 scale.",
               "Share of the cluster (%)", None)
        save(fig, "fig25_clusters_vs_genre.png")

    with open(os.path.join(CLEAN, "clustering_metrics.json"), "w") as fh:
        json.dump(M, fh, indent=2)

    print("\n" + "=" * 70)
    print(f"Clustering complete. best k = {best_k}, "
          f"silhouette = {M['best_silhouette']:.3f}, ARI vs genre = {ari_genre:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
