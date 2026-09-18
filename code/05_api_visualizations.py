"""
CSCI 5612 - Project Part 1
Script 05: Visualisations of the API-collected data (Deezer + MusicBrainz).

Kept separate from 03 because it only runs once 01_collect_api_data.py has
finished. Writes ../docs/images/fig16..fig19.
"""
import os, sys, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "data", "clean")
IMG = os.path.join(HERE, "..", "docs", "images")

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ACCENT, INK, INK_SOFT, GRID = "#4a3aa7", "#1b1d21", "#4a4f57", "#e2e5ea"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 14, "axes.labelsize": 10.5, "axes.labelcolor": INK_SOFT,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0, "axes.grid": True,
    "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.9,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT, "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5, "legend.frameon": False, "text.color": INK,
})


def finish(ax, title, subtitle=None, xlabel=None, ylabel=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{title}\n" if subtitle else title, loc="left")
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.8,
                color=INK_SOFT, va="bottom")
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)


def save(fig, name):
    fig.savefig(os.path.join(IMG, name)); plt.close(fig); print(f"  saved {name}")


def main():
    tp = os.path.join(CLEAN, "deezer_tracks_clean.csv")
    if not os.path.exists(tp):
        sys.exit("Deezer data not found. Run 01_collect_api_data.py then 02_clean_data.py first.")

    t = pd.read_csv(tp)
    ap = os.path.join(CLEAN, "deezer_artists_clean.csv")
    gp = os.path.join(CLEAN, "musicbrainz_tags_clean.csv")

    # -------------------------------------------------------------- 16 ----
    if os.path.exists(ap):
        a = pd.read_csv(ap)
        j = t.merge(a[["artist_id", "nb_fan"]], on="artist_id", how="inner")
        j = j[(j.nb_fan > 0) & (j["rank"] > 0)]
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        hb = ax.hexbin(np.log10(j.nb_fan), j["rank"], gridsize=30, cmap="Purples",
                       mincnt=1, linewidths=0)
        cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label("Number of tracks", color=INK_SOFT, fontsize=9.5)
        cb.outline.set_visible(False)
        r = np.corrcoef(np.log10(j.nb_fan), j["rank"])[0, 1]
        ax.set_xticks([3, 4, 5, 6, 7], ["1K", "10K", "100K", "1M", "10M"])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v/1000:.0f}K"))
        ax.grid(False)
        finish(ax, "Existing audience predicts performance far better than sound does",
               f"Track popularity rank against the artist's follower count, {len(j):,} tracks "
               f"(correlation {r:.2f})",
               "Artist followers (log scale)", "Track popularity rank")
        save(fig, "fig16_artist_reach_vs_rank.png")

    # -------------------------------------------------------------- 17 ----
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    dec = t.groupby("release_decade").size()
    dec = dec[dec.index >= 1950]
    ax.bar([str(int(d)) + "s" for d in dec.index], dec.values, color=C[2],
           width=0.62, edgecolor="white", linewidth=1.6)
    for i, v in enumerate(dec.values):
        ax.text(i, v + max(dec.values) * 0.02, f"{v:,}", ha="center", fontsize=9,
                color=INK_SOFT)
    ax.set_ylim(0, max(dec.values) * 1.14)
    finish(ax, "The API sample spans seven decades, not just the present",
           f"Release decade of the {len(t):,} tracks gathered from the Deezer API",
           None, "Number of tracks")
    save(fig, "fig17_api_sample_coverage.png")

    # -------------------------------------------------------------- 18 ----
    if os.path.exists(gp):
        g = pd.read_csv(gp)
        top = g.tag.value_counts().head(22).sort_values()
        fig, ax = plt.subplots(figsize=(8.2, 6.2))
        ax.barh(range(len(top)), top.values, color=ACCENT, height=0.66)
        ax.set_yticks(range(len(top)), top.index)
        for i, v in enumerate(top.values):
            ax.text(v + max(top.values) * 0.012, i, str(v), va="center",
                    fontsize=8.8, color=INK_SOFT)
        ax.set_xlim(0, max(top.values) * 1.12)
        finish(ax, "Listeners overwhelmingly describe music by genre, not by feeling",
               f"The most common tags among {g.artist_name.nunique():,} artists, written "
               f"freely by listeners",
               "Number of artists carrying the tag", None)
        save(fig, "fig18_top_listener_tags.png")

        # ---------------------------------------------------------- 19 ----
        per_artist = g.groupby("artist_name").tag.nunique()
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        ax.hist(per_artist.values, bins=range(0, int(per_artist.max()) + 2),
                color=C[1], edgecolor="white", linewidth=0.6)
        ax.axvline(per_artist.median(), color=ACCENT, linewidth=2, linestyle="--")
        ax.annotate(f"median {per_artist.median():.0f} tags",
                    xy=(per_artist.median(), ax.get_ylim()[1] * 0.85),
                    xytext=(8, 0), textcoords="offset points", color=ACCENT,
                    fontweight="bold", fontsize=9.5)
        finish(ax, "Most artists carry only a handful of listener tags",
               f"Number of distinct tags per artist across {len(per_artist):,} artists",
               "Distinct tags", "Number of artists")
        save(fig, "fig19_tags_per_artist.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
