"""
CSCI 5612 - Project Part 1
Script 04: Render small before/after preview images of each dataset.

Writes ../docs/images/raw_*.png and clean_*.png
"""
import os, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap
import pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RAW   = os.path.join(HERE, "..", "data", "raw")
CLEAN = os.path.join(HERE, "..", "data", "clean")
IMG   = os.path.join(HERE, "..", "docs", "images")
os.makedirs(IMG, exist_ok=True)

INK, INK_SOFT, GRID = "#1b1d21", "#4a4f57", "#e2e5ea"
HDR_RAW, HDR_CLEAN = "#eb6834", "#4a3aa7"


def table_png(df, cols, title, subtitle, out, header_color, n=6, colw=None):
    d = df[cols].head(n).copy()
    for c in d.columns:
        d[c] = d[c].astype(str).str.slice(0, 22)
    ncol = len(cols)
    fig, ax = plt.subplots(figsize=(min(2.0 * ncol, 15), 0.30 * n + 0.95))
    ax.axis("off")
    heads = ["\n".join(textwrap.wrap(c.replace("_", " "), 13)[:2]) for c in d.columns]
    tbl = ax.table(cellText=d.values, colLabels=heads,
                   cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold", fontsize=7.6)
            cell.set_height(cell.get_height() * 1.7)
        else:
            cell.set_facecolor("#ffffff" if r % 2 else "#f7f8fa")
            cell.set_text_props(color=INK)
    if colw:
        for (r, c), cell in tbl.get_celld().items():
            cell.set_width(colw)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=20)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9,
            color=INK_SOFT, va="bottom")
    fig.savefig(os.path.join(IMG, out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    # ---- Spotify -----------------------------------------------------------
    r = pd.read_csv(os.path.join(RAW, "spotify_songs_raw.csv"))
    table_png(r, ["track_name", "track_artist", "track_album_release_date",
                  "playlist_genre", "track_popularity", "danceability", "tempo",
                  "duration_ms"],
              "RAW: Spotify track features",
              f"{len(r):,} rows before cleaning. Release dates arrive in three different "
              f"formats, songs repeat across playlists, and length is in milliseconds.",
              "raw_spotify.png", HDR_RAW)

    c = pd.read_csv(os.path.join(CLEAN, "spotify_songs_clean.csv"))
    table_png(c, ["track_name", "track_artist", "release_year", "playlist_genre",
                  "track_popularity", "popularity_band", "danceability",
                  "tempo_band", "duration_min"],
              "CLEAN: Spotify track features",
              f"{len(c):,} rows after cleaning. One row per song, a single numeric release "
              f"year, length in minutes, and new banded variables for later use.",
              "clean_spotify.png", HDR_CLEAN)

    # ---- Billboard ---------------------------------------------------------
    r = pd.read_csv(os.path.join(RAW, "billboard_hot100_raw.csv"))
    table_png(r.iloc[95:], ["chart_week", "current_week", "title", "performer",
                            "last_week", "peak_pos", "wks_on_chart"],
              "RAW: Billboard Hot 100 weekly entries",
              f"{len(r):,} weekly rows before cleaning. The last_week column is blank "
              f"wherever a song was not on the chart the previous week.",
              "raw_billboard.png", HDR_RAW)

    c = pd.read_csv(os.path.join(CLEAN, "billboard_songs_clean.csv"))
    table_png(c.sort_values("weeks_on_chart", ascending=False),
              ["title", "performer", "peak_position", "weeks_on_chart",
               "debut_year", "reached_top10", "reached_number1"],
              "CLEAN: Billboard, one row per song",
              f"{len(c):,} distinct songs after cleaning. Weekly entries are collapsed into "
              f"one record per song carrying its best position and total time on the chart.",
              "clean_billboard.png", HDR_CLEAN)

    # ---- Linked table ------------------------------------------------------
    c = pd.read_csv(os.path.join(CLEAN, "songs_with_chart_outcome.csv"))
    table_png(c.sort_values("weeks_on_chart", ascending=False),
              ["track_name", "track_artist", "playlist_genre", "track_popularity",
               "valence", "charted", "peak_position", "weeks_on_chart"],
              "CLEAN: audio qualities joined to chart outcome",
              f"{len(c):,} songs, each carrying both what it sounds like and how it "
              f"performed. Songs that never charted are coded 101, a real outcome not a gap.",
              "clean_linked.png", HDR_CLEAN)

    # ---- Deezer / MusicBrainz (only once the API script has run) -----------
    p = os.path.join(RAW, "deezer_tracks_raw.csv")
    if os.path.exists(p):
        r = pd.read_csv(p)
        table_png(r, ["title", "artist_name", "release_date", "duration_sec",
                      "bpm", "rank", "explicit_lyrics", "source_genre"],
                  "RAW: Deezer API track records",
                  f"{len(r):,} tracks as returned by the API. Tempo is reported as 0 when "
                  f"no analysis exists, and length is in seconds.",
                  "raw_deezer.png", HDR_RAW)
        c = pd.read_csv(os.path.join(CLEAN, "deezer_tracks_clean.csv"))
        table_png(c, ["title", "artist_name", "release_year", "duration_min", "bpm",
                      "bpm_known", "rank", "explicit_lyrics", "source_genre"],
                  "CLEAN: Deezer API track records",
                  f"{len(c):,} tracks after cleaning. Missing tempos are flagged and filled "
                  f"rather than silently kept as zero.",
                  "clean_deezer.png", HDR_CLEAN)

    p = os.path.join(RAW, "musicbrainz_tags_raw.csv")
    if os.path.exists(p):
        r = pd.read_csv(p)
        table_png(r, ["artist_name", "mb_country", "mb_type", "mb_begin_year",
                      "tag", "tag_count"],
                  "RAW: MusicBrainz listener tags",
                  f"{len(r):,} artist and tag pairs. Tags are typed freely by listeners, so "
                  f"casing, punctuation and blanks are all inconsistent.",
                  "raw_tags.png", HDR_RAW)
        c = pd.read_csv(os.path.join(CLEAN, "musicbrainz_tags_clean.csv"))
        table_png(c, ["artist_name", "mb_country", "mb_type", "mb_begin_year",
                      "tag", "tag_count"],
                  "CLEAN: MusicBrainz listener tags",
                  f"{len(c):,} usable pairs after cleaning. Tags are lowercased, stripped of "
                  f"punctuation, de-duplicated, and empty entries removed.",
                  "clean_tags.png", HDR_CLEAN)

    print("\nPreviews written to", os.path.abspath(IMG))


if __name__ == "__main__":
    main()
