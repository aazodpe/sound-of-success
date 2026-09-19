"""
CSCI 5612 - Project Part 1
Script 03: Exploratory visualisations.

Reads  ../data/clean/
Writes ../docs/images/*.png

Run: python 03_eda_visualizations.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "data", "clean")
IMG = os.path.join(HERE, "..", "docs", "images")
os.makedirs(IMG, exist_ok=True)

import theme

GENRE_ORDER = ["pop", "rap", "rock", "latin", "r&b", "edm"]
GENRE_LABEL = {"pop": "Pop", "rap": "Rap", "rock": "Rock",
               "latin": "Latin", "r&b": "R&B", "edm": "EDM"}


def palette():
    """Colours for the mode currently set on the theme module."""
    return theme.C, theme.ACCENT, theme.INK, theme.INK_SOFT, theme.GRID


def genre_colors():
    return {g: theme.C[i] for i, g in enumerate(GENRE_ORDER)}


def on_color(swatch):
    """Pick black or white label text from a swatch's actual luminance."""
    r, g, b = matplotlib.colors.to_rgb(swatch)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#11111a" if lum > 0.55 else "#ffffff"


def finish(ax, title, subtitle=None, xlabel=None, ylabel=None, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)
    if subtitle:
        ax.set_title(f"{title}\n", loc="left")
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.8,
                color=theme.INK_SOFT, va="bottom", ha="left")
    else:
        ax.set_title(title, loc="left")
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)


def save(fig, name):
    path = os.path.join(IMG, theme.out_name(name))
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {theme.out_name(name)}")


# ============================================================================

def main():
    C, ACCENT, INK, INK_SOFT, GRID = palette()
    GENRE_COLOR = genre_colors()
    s = pd.read_csv(os.path.join(CLEAN, "spotify_songs_clean.csv"))
    m = pd.read_csv(os.path.join(CLEAN, "songs_with_chart_outcome.csv"))
    bw = pd.read_csv(os.path.join(CLEAN, "billboard_weekly_clean.csv"),
                     parse_dates=["chart_week"])
    bs = pd.read_csv(os.path.join(CLEAN, "billboard_songs_clean.csv"),
                     parse_dates=["debut_week"])
    print(f"loaded {len(s):,} songs / {len(bw):,} chart weeks\n")

    # ---------------------------------------------------------------- 01 ----
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.hist(s.track_popularity, bins=50, color=ACCENT, alpha=0.9,
            edgecolor=theme.BG, linewidth=0.4)
    med = s.track_popularity.median()
    ax.axvline(med, color=C[1], linewidth=2, linestyle="--")
    ax.annotate(f"median {med:.0f}", xy=(med, ax.get_ylim()[1] * 0.88),
                xytext=(8, 0), textcoords="offset points",
                color=C[1], fontweight="bold", fontsize=10)
    finish(ax, "Most songs are not popular, and a few are very popular",
           f"Popularity score of {len(s):,} songs (0 = never played, 100 = most played)",
           "Popularity score", "Number of songs")
    save(fig, "fig01_popularity_distribution.png")

    # ---------------------------------------------------------------- 02 ----
    feats = ["danceability", "energy", "valence", "acousticness",
             "speechiness", "instrumentalness"]
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.2))
    for ax, f in zip(axes.ravel(), feats):
        ax.hist(s[f], bins=40, color=ACCENT, alpha=0.88,
                edgecolor=theme.BG, linewidth=0.3)
        ax.set_title(f.capitalize(), loc="left", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:,.0f}"))
    fig.suptitle("Each measured quality has its own characteristic shape",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.06)
    fig.text(0.01, 0.995, "Distribution of six audio measures across all "
             f"{len(s):,} songs; each runs from 0 to 1",
             fontsize=9.8, color=INK_SOFT, ha="left")
    fig.supylabel("Number of songs", fontsize=10.5, color=INK_SOFT, x=0.005)
    fig.tight_layout(rect=[0.01, 0, 1, 0.94])
    save(fig, "fig02_feature_distributions.png")

    # ---------------------------------------------------------------- 03 ----
    corr_cols = ["track_popularity", "danceability", "energy", "loudness",
                 "speechiness", "acousticness", "instrumentalness",
                 "liveness", "valence", "tempo", "duration_min"]
    corr = s[corr_cols].corr()
    labels = [c.replace("_", " ").replace("track ", "").capitalize() for c in corr_cols]
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    im = ax.imshow(corr, cmap=theme.DIV_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.6,
                    color="#ffffff" if abs(v) > 0.55 else theme.INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
    cb.set_label("Correlation", color=INK_SOFT, fontsize=9.5)
    cb.outline.set_visible(False)
    ax.set_title("Loudness and energy move together; popularity moves with almost nothing\n",
                 loc="left")
    ax.text(0, 1.015, "Pairwise correlation between every measured quality and popularity",
            transform=ax.transAxes, fontsize=9.8, color=INK_SOFT, va="bottom")
    save(fig, "fig03_correlation_heatmap.png")

    # ---------------------------------------------------------------- 04 ----
    yr = s[s.release_year.between(1965, 2020)].groupby("release_year").agg(
        loudness=("loudness", "median"), n=("loudness", "size"))
    yr = yr[yr.n >= 15]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(yr.index, yr.loudness, color=C[1], linewidth=2.2)
    ax.fill_between(yr.index, yr.loudness, yr.loudness.min() - 1,
                    color=C[1], alpha=0.10)
    ax.annotate(f"{yr.loudness.iloc[0]:.1f} dB", xy=(yr.index[0], yr.loudness.iloc[0]),
                xytext=(6, -14), textcoords="offset points", color=C[1],
                fontweight="bold", fontsize=9.5)
    ax.annotate(f"{yr.loudness.iloc[-1]:.1f} dB", xy=(yr.index[-1], yr.loudness.iloc[-1]),
                xytext=(-16, 10), textcoords="offset points", color=C[1],
                fontweight="bold", fontsize=9.5)
    ax.set_ylim(yr.loudness.min() - 1, yr.loudness.max() + 1.2)
    finish(ax, "Recordings grew louder for fifty years, and have only just stopped",
           "Median loudness of songs by year of release (decibels; closer to zero is louder)",
           "Year released", "Median loudness (dB)")
    save(fig, "fig04_loudness_over_time.png")

    # ---------------------------------------------------------------- 05 ----
    yr2 = s[s.release_year.between(1965, 2020)].groupby("release_year").agg(
        dur=("duration_min", "median"), n=("duration_min", "size"))
    yr2 = yr2[yr2.n >= 15]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(yr2.index, yr2.dur, color=C[0], linewidth=2.2)
    peak_year = yr2.dur.idxmax()
    ax.scatter([peak_year], [yr2.dur.max()], s=55, color=C[0], zorder=5,
               edgecolor=theme.BG, linewidth=1.6)
    ax.annotate(f"peak: {yr2.dur.max():.1f} min in {peak_year}",
                xy=(peak_year, yr2.dur.max()), xytext=(8, 6),
                textcoords="offset points", color=C[0], fontweight="bold", fontsize=9.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, p: f"{int(v)}:{int(round((v % 1) * 60)):02d}"))
    finish(ax, "Songs grew longer through the album era, then began shrinking again",
           "Median song length by year of release",
           "Year released", "Median length (minutes:seconds)")
    save(fig, "fig05_duration_over_time.png")

    # ---------------------------------------------------------------- 06 ----
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.6), sharex=True, sharey=True)
    for ax, g in zip(axes.ravel(), GENRE_ORDER):
        sub = s[s.playlist_genre == g]
        ax.scatter(s.danceability, s.energy, s=2, color=theme.GRID, alpha=0.75,
                   linewidths=0, rasterized=True)
        ax.scatter(sub.danceability, sub.energy, s=3.5, color=GENRE_COLOR[g],
                   alpha=0.45, linewidths=0, rasterized=True)
        ax.set_title(f"{GENRE_LABEL[g]}  ({len(sub):,})", loc="left", fontsize=11,
                     color=GENRE_COLOR[g])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Genres occupy overlapping, not separate, regions of the song space",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.0)
    fig.text(0.01, 0.962, "Each panel highlights one genre against all other songs in grey",
             fontsize=9.8, color=INK_SOFT, ha="left")
    fig.supxlabel("Danceability", fontsize=10.5, color=INK_SOFT)
    fig.supylabel("Energy", fontsize=10.5, color=INK_SOFT)
    fig.tight_layout(rect=[0.01, 0.01, 1, 0.94])
    save(fig, "fig06_genre_space.png")

    # ---------------------------------------------------------------- 07 ----
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    data = [s.loc[s.playlist_genre == g, "track_popularity"].values for g in GENRE_ORDER]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops=dict(color=theme.BG, linewidth=2),
                    whiskerprops=dict(color=INK_SOFT, linewidth=1.1),
                    capprops=dict(color=INK_SOFT, linewidth=1.1))
    for patch, g in zip(bp["boxes"], GENRE_ORDER):
        patch.set_facecolor(GENRE_COLOR[g])
        patch.set_edgecolor(theme.BG)
        patch.set_linewidth(1.6)
    ax.set_xticks(range(1, len(GENRE_ORDER) + 1),
                  [f"{GENRE_LABEL[g]}\nmed {np.median(d):.0f}" for g, d in zip(GENRE_ORDER, data)])
    finish(ax, "Every genre contains both ignored songs and popular ones",
           "Distribution of popularity score within each genre; boxes span the middle half",
           None, "Popularity score")
    save(fig, "fig07_popularity_by_genre.png")

    # ---------------------------------------------------------------- 08 ----
    cmp_feats = ["danceability", "energy", "valence", "acousticness",
                 "speechiness", "liveness", "instrumentalness"]
    hit = m[m.charted == 1][cmp_feats].mean()
    non = m[m.charted == 0][cmp_feats].mean()
    diff = ((hit - non)).sort_values()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    colors = [C[0] if v > 0 else C[1] for v in diff.values]
    ax.barh(range(len(diff)), diff.values, color=colors, height=0.62)
    ax.axvline(0, color=INK_SOFT, linewidth=1.1)
    ax.set_yticks(range(len(diff)), [i.capitalize() for i in diff.index])
    for i, v in enumerate(diff.values):
        ax.text(v + (0.004 if v > 0 else -0.004), i, f"{v:+.3f}", va="center",
                ha="left" if v > 0 else "right", fontsize=9, color=INK_SOFT)
    ax.set_xlim(diff.min() * 1.5, diff.max() * 1.5)
    finish(ax, "Charting songs have vocals and a brighter mood; otherwise they barely differ",
           f"Average difference between the {int(m.charted.sum()):,} songs that reached the "
           f"Hot 100 and the {int((1-m.charted).sum()):,} that did not",
           "Difference in average value (charted minus not charted)", None)
    save(fig, "fig08_charted_vs_not.png")

    # ---------------------------------------------------------------- 09 ----
    per_year = bw.groupby("chart_year").title.nunique()
    per_year = per_year[(per_year.index >= 1959) & (per_year.index <= 2025)]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(per_year.index, per_year.values, color=C[6], linewidth=2.2)
    ax.fill_between(per_year.index, per_year.values, 0, color=C[6], alpha=0.10)
    lo = per_year.idxmin()
    ax.scatter([lo], [per_year[lo]], s=55, color=C[6], zorder=5,
               edgecolor=theme.BG, linewidth=1.6)
    ax.annotate(f"low point: {per_year[lo]} songs in {lo}", xy=(lo, per_year[lo]),
                xytext=(0, -26), textcoords="offset points", ha="center",
                fontsize=9.3, color=C[6], fontweight="bold")
    ax.axvline(2012, color=INK_SOFT, linewidth=1, linestyle=":")
    ax.annotate("streaming counted\ntoward the chart", xy=(2012, per_year.max() * 1.02),
                xytext=(-8, 0), textcoords="offset points", ha="right", va="top",
                fontsize=8.8, color=INK_SOFT)
    ax.set_ylim(0, per_year.max() * 1.20)
    finish(ax, "Chart variety shrank for forty years, then streaming reversed it",
           "Number of distinct songs appearing anywhere on the Billboard Hot 100 each year",
           "Year", "Distinct songs on the chart")
    save(fig, "fig09_chart_turnover.png")

    # ---------------------------------------------------------------- 10 ----
    dec = bs[bs.debut_decade.between(1960, 2020)]
    decades = sorted(dec.debut_decade.unique())
    med = [dec.loc[dec.debut_decade == d, "weeks_on_chart"].median() for d in decades]
    one = [(dec.loc[dec.debut_decade == d, "weeks_on_chart"] <= 1).mean() * 100 for d in decades]
    xs = np.arange(len(decades))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 6.4), sharex=True)
    ax1.plot(xs, med, color=C[2], linewidth=2.4, marker="o", markersize=7,
             markeredgecolor=theme.BG, markeredgewidth=1.6)
    peak = int(np.argmax(med))
    ax1.annotate(f"{med[peak]:.0f} weeks", xy=(xs[peak], med[peak]), xytext=(0, 11),
                 textcoords="offset points", ha="center", color=C[2],
                 fontweight="bold", fontsize=9.5)
    ax1.annotate(f"{med[-1]:.0f} weeks", xy=(xs[-1], med[-1]), xytext=(-4, 12),
                 textcoords="offset points", ha="right", color=C[2],
                 fontweight="bold", fontsize=9.5)
    ax1.set_ylim(0, max(med) * 1.35)
    finish(ax1, "Chart runs peaked in the 2000s and then collapsed",
           "Median number of weeks a song spends on the Hot 100, by the decade it debuted",
           None, "Median weeks on chart")

    bars = ax2.bar(xs, one, color=C[1], width=0.6, edgecolor=theme.BG, linewidth=1.8)
    for b, v in zip(bars, one):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.4, f"{v:.0f}%", ha="center",
                 fontsize=9.3, fontweight="bold", color=INK_SOFT)
    ax2.set_ylim(0, max(one) * 1.30)
    ax2.set_xticks(xs, [f"{int(d)}s" for d in decades])
    finish(ax2, "Because four in ten entries now last a single week",
           "Share of charting songs that appeared for exactly one week and then vanished",
           None, "Share of songs (%)")
    fig.tight_layout(h_pad=2.6)
    save(fig, "fig10_weeks_on_chart_by_decade.png")

    # ---------------------------------------------------------------- 11 ----
    tab = (s.groupby(["release_decade", "mode_name"]).size()
             .unstack(fill_value=0))
    tab = tab[tab.sum(axis=1) >= 60]
    share = tab.div(tab.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(share))
    ax.bar(x, share["Major"], color=C[0], width=0.62, label="Major key")
    ax.bar(x, share["Minor"], bottom=share["Major"] + 0.6, color=C[4],
           width=0.62, label="Minor key")
    for i, v in enumerate(share["Major"]):
        ax.text(i, v / 2, f"{v:.0f}%", ha="center", va="center",
                color=on_color(C[0]), fontsize=9, fontweight="bold")
    ax.set_xticks(x, [f"{int(d)}s" for d in share.index])
    ax.set_ylim(0, 104)
    ax.legend(loc="lower right", ncols=2)
    finish(ax, "Popular music has drifted away from major keys",
           "Share of songs written in a major versus a minor key, by decade of release",
           None, "Share of songs (%)")
    save(fig, "fig11_key_mode_by_decade.png")

    # ---------------------------------------------------------------- 12 ----
    v = s[s.release_year.between(1965, 2020)].groupby("release_year").agg(
        val=("valence", "median"), n=("valence", "size"))
    v = v[v.n >= 15]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(v.index, v.val, color=C[4], linewidth=2.2)
    ax.axhline(0.5, color=INK_SOFT, linewidth=1, linestyle=":")
    ax.text(v.index[2], 0.508, "neutral", fontsize=9, color=INK_SOFT)
    finish(ax, "Popular music has been getting less cheerful",
           "Median musical positivity by year of release (1 = upbeat, 0 = downbeat)",
           "Year released", "Median valence")
    save(fig, "fig12_valence_over_time.png")

    # ---------------------------------------------------------------- 13 ----
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    sub = bs[(bs.weeks_on_chart <= 40)]
    hb = ax.hexbin(sub.peak_position, sub.weeks_on_chart, gridsize=34,
                   cmap=theme.SEQ_CMAP, mincnt=1, linewidths=0)
    cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label("Number of songs", color=INK_SOFT, fontsize=9.5)
    cb.outline.set_visible(False)
    ax.set_xlim(0, 101)
    ax.grid(False)
    finish(ax, "Reaching a high position and lasting a long time are the same achievement",
           f"Every one of the {len(sub):,} songs that has charted, by best position and total weeks",
           "Best position reached (1 is the top)", "Total weeks on the chart")
    save(fig, "fig13_peak_vs_longevity.png")

    # ---------------------------------------------------------------- 14 ----
    band_order = ["Unheard", "Modest", "Popular", "Hit"]
    tab = (s.groupby(["playlist_genre", "popularity_band"]).size().unstack(fill_value=0))
    tab = tab.loc[GENRE_ORDER, band_order]
    share = tab.div(tab.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    left = np.zeros(len(share))
    shades = theme.SEQ_CMAP(np.linspace(0.25, 0.9, 4)) if not isinstance(theme.SEQ_CMAP, str) \
        else ["#ded9ec", "#b3a6d6", "#7a66bd", ACCENT]
    for band, col in zip(band_order, shades):
        vals = share[band].values
        ax.barh(range(len(share)), vals, left=left, color=col, height=0.62,
                edgecolor=theme.BG, linewidth=1.6, label=band)
        label_ink = on_color(col)
        for i, (val, l) in enumerate(zip(vals, left)):
            if val > 6:
                ax.text(l + val / 2, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=label_ink)
        left += vals + 0.5
    ax.set_yticks(range(len(share)), [GENRE_LABEL[g] for g in share.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 103)
    ax.grid(False)
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    finish(ax, "The mix of ignored and popular songs differs sharply by genre",
           "Share of each genre's songs falling into four popularity bands",
           "Share of songs (%)", None)
    save(fig, "fig14_popularity_bands_by_genre.png")

    # ---------------------------------------------------------------- 15 ----
    # Introduction figure: the odds facing a new release (Luminate 2025 figures)
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    cats = ["0 to 10\nstreams", "11 to 1,000\nstreams", "More than 1,000\nstreams"]
    vals = [120.5, 102.1, 30.4]
    bars = ax.bar(cats, vals, color=[C[1], "#f0a98a", C[0]], width=0.55,
                  edgecolor=theme.BG, linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.1f}M",
                ha="center", fontsize=11.5, fontweight="bold", color=INK)
    ax.set_ylim(0, 145)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.grid(False)
    finish(ax, "Nine in ten recordings are never really heard",
           "Of roughly 253 million recordings on streaming services, how many were played in 2025",
           None, None, spines=("top", "right", "left"))
    ax.text(0, -0.30, "Source: Luminate 2025 year-end report. On Spotify a track must reach "
            "1,000 plays in a year before it earns anything at all.",
            transform=ax.transAxes, fontsize=8.6, color=INK_SOFT, va="top")
    save(fig, "fig15_intro_odds.png")

    print(f"\nAll figures written to {os.path.abspath(IMG)}")


if __name__ == "__main__":
    for mode in theme.MODES:
        print(f"\n--- {mode} ---")
        theme.set_theme(mode)
        main()
