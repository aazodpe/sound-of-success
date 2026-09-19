"""
CSCI 5612 - Project Part 1
Script 02: Clean and prepare every raw dataset.

Reads from  ../data/raw/
Writes to   ../data/clean/   plus small raw/clean preview images to ../docs/images/

Run: python 02_clean_data.py
"""

import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw")
CLEAN = os.path.join(HERE, "..", "data", "clean")
os.makedirs(CLEAN, exist_ok=True)

AUDIO_FEATURES = ["danceability", "energy", "loudness", "speechiness",
                  "acousticness", "instrumentalness", "liveness", "valence",
                  "tempo", "duration_min"]

KEY_NAMES = ["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"]


# Values pandas silently reads back as missing. A match key must never be one
# of these, or a perfectly good song reappears as a hole in the data.
NA_TOKENS = {"", "na", "n/a", "nan", "null", "none", "nat", "<na>", "#na",
             "#n/a", "-nan", "inf", "-inf", "1.#ind", "-1.#ind"}


def make_key(value, drop_featured=False):
    """
    Normalise a title or artist into a comparable key.

    Keeps letters and digits from any alphabet, so titles in Cyrillic or with
    accents do not collapse to nothing. Falls back progressively rather than
    ever returning an empty or NA-looking key.
    """
    original = str(value)
    s = original.casefold()
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)
    s = re.sub(r"\s*-\s*(remaster|remastered|live|radio edit|mono|stereo|version).*$", " ", s)
    if drop_featured:
        s = re.sub(r"\s+(featuring|feat\.?|ft\.?|with|x|&|and)\s+.*$", " ", s)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()

    if not s:                       # title was only punctuation, e.g. "+" or "!!!"
        s = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", original.casefold())).strip()
    if not s:                       # still nothing, so keep the original verbatim
        s = re.sub(r"\s+", " ", original.casefold()).strip()
    if not s:
        return "unknown"
    if s in NA_TOKENS:              # e.g. "Nanã" would otherwise become "nan"
        s = s + " k"
    return s


log = []


def step(msg):
    print(f"  {msg}")
    log.append(msg)


# ============================================================================
# 1. Spotify audio features
# ============================================================================

def clean_spotify():
    print("\n=== Spotify tracks ===")
    df = pd.read_csv(os.path.join(RAW, "spotify_songs_raw.csv"))
    n0 = len(df)
    step(f"loaded {n0:,} rows x {df.shape[1]} columns")

    # --- missing values -----------------------------------------------------
    missing = df[["track_name", "track_artist", "track_album_name"]].isna().any(axis=1).sum()
    df = df.dropna(subset=["track_name", "track_artist"])
    step(f"dropped {missing} rows missing a track name or artist")

    # --- duplicates ---------------------------------------------------------
    # The raw file is one row per (track, playlist), so a song that appears on
    # several playlists appears several times. Keep one row per track.
    n_before = len(df)
    df = df.sort_values("track_popularity", ascending=False)
    df = df.drop_duplicates(subset="track_id", keep="first")
    step(f"removed {n_before - len(df):,} duplicate track_id rows "
         f"(same song listed on multiple playlists)")

    # Remasters and re-releases repeat a song under a new id. Normalise the
    # title and artist and collapse those too, keeping the most popular version.
    df["title_key"] = df["track_name"].map(make_key)
    df["artist_key"] = df["track_artist"].map(make_key)
    n_before = len(df)
    df = df.drop_duplicates(subset=["title_key", "artist_key"], keep="first")
    step(f"removed {n_before - len(df):,} repeated recordings of the same song "
         f"(remasters, live versions, re-releases)")

    # --- release date: three different formats in one column ----------------
    raw_dates = df["track_album_release_date"].astype(str)
    fmt_counts = raw_dates.str.len().value_counts().to_dict()
    step(f"release dates arrived in mixed formats (string lengths {fmt_counts}); "
         f"parsed the leading year from each")
    df["release_year"] = pd.to_numeric(raw_dates.str.slice(0, 4), errors="coerce")

    n_before = len(df)
    df = df[df["release_year"].between(1950, 2026)]
    step(f"dropped {n_before - len(df):,} rows with an impossible release year")
    df["release_year"] = df["release_year"].astype(int)
    df["release_decade"] = (df["release_year"] // 10 * 10).astype(int)

    # --- impossible measurements -------------------------------------------
    n_before = len(df)
    df = df[df["tempo"] > 0]
    step(f"dropped {n_before - len(df):,} rows with a tempo of 0 BPM "
         f"(a recorded song cannot have no tempo)")

    df["duration_min"] = df["duration_ms"] / 60000.0
    n_before = len(df)
    df = df[df["duration_min"].between(0.5, 15)]
    step(f"dropped {n_before - len(df):,} rows shorter than 30s or longer than 15min "
         f"(interludes, DJ mixes and sound effects, not songs)")

    # Probabilistic features must lie in [0, 1].
    prob_cols = ["danceability", "energy", "speechiness", "acousticness",
                 "instrumentalness", "liveness", "valence"]
    bad = (~df[prob_cols].apply(lambda c: c.between(0, 1)).all(axis=1)).sum()
    df = df[df[prob_cols].apply(lambda c: c.between(0, 1)).all(axis=1)]
    step(f"dropped {bad} rows whose 0-1 audio measures fell outside that range")

    # --- readable categoricals ---------------------------------------------
    df["key_name"] = df["key"].map(lambda k: KEY_NAMES[int(k)] if 0 <= k <= 11 else np.nan)
    df["mode_name"] = df["mode"].map({1: "Major", 0: "Minor"})
    df = df.dropna(subset=["key_name", "mode_name"])

    # --- discretisation (adds variables, as the assignment allows) ----------
    df["popularity_band"] = pd.cut(
        df["track_popularity"], bins=[-0.1, 25, 50, 75, 100],
        labels=["Unheard", "Modest", "Popular", "Hit"])
    df["is_popular"] = (df["track_popularity"] >= 60).astype(int)
    df["tempo_band"] = pd.cut(
        df["tempo"], bins=[0, 90, 120, 150, 300],
        labels=["Slow (<90)", "Moderate (90-120)", "Fast (120-150)", "Very fast (150+)"])
    df["duration_band"] = pd.cut(
        df["duration_min"], bins=[0, 2.5, 3.5, 4.5, 15],
        labels=["Under 2:30", "2:30-3:30", "3:30-4:30", "Over 4:30"])
    step("added discretised variables: popularity_band, tempo_band, duration_band, "
         "and the binary label is_popular")

    keep = ["track_id", "track_name", "track_artist", "title_key", "artist_key",
            "track_album_name", "release_year", "release_decade",
            "playlist_genre", "playlist_subgenre", "track_popularity",
            "popularity_band", "is_popular",
            "danceability", "energy", "key", "key_name", "loudness",
            "mode", "mode_name", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "tempo_band",
            "duration_ms", "duration_min", "duration_band"]
    df = df[keep].reset_index(drop=True)

    assert df.isna().sum().sum() == 0, "cleaned Spotify data still contains NAs"
    step(f"FINAL: {len(df):,} rows x {df.shape[1]} columns, zero missing values "
         f"({n0 - len(df):,} rows removed in total)")

    df.to_csv(os.path.join(CLEAN, "spotify_songs_clean.csv"), index=False)

    # A separately normalised copy, for the distance-based methods later on.
    norm_df = df.copy()
    for col in AUDIO_FEATURES:
        lo, hi = norm_df[col].min(), norm_df[col].max()
        norm_df[col] = (norm_df[col] - lo) / (hi - lo)
    norm_df.to_csv(os.path.join(CLEAN, "spotify_songs_normalized.csv"), index=False)
    step("wrote a min-max normalised copy for the distance-based methods")

    return df


# ============================================================================
# 2. Billboard Hot 100
# ============================================================================

def clean_billboard():
    print("\n=== Billboard Hot 100 ===")
    df = pd.read_csv(os.path.join(RAW, "billboard_hot100_raw.csv"))
    n0 = len(df)
    step(f"loaded {n0:,} weekly chart entries")

    df["chart_week"] = pd.to_datetime(df["chart_week"], errors="coerce")
    df = df.dropna(subset=["chart_week"])
    df["chart_year"] = df["chart_week"].dt.year
    df["chart_decade"] = (df["chart_year"] // 10 * 10).astype(int)
    step(f"parsed chart dates: {df.chart_week.min():%Y-%m-%d} to {df.chart_week.max():%Y-%m-%d}")

    # `last_week` is blank whenever a song was not on the chart the week before.
    # That is not a data error, it is a debut, so it gets its own flag rather
    # than being dropped or silently imputed.
    n_blank = df["last_week"].isna().sum()
    df["is_debut"] = df["last_week"].isna().astype(int)
    df["last_week"] = df["last_week"].fillna(101).astype(int)   # 101 = "off the chart"
    step(f"{n_blank:,} blank last_week values were chart debuts, not missing data: "
         f"flagged as is_debut and coded 101 ('outside the Hot 100')")

    df["title"] = df["title"].str.strip()
    df["performer"] = df["performer"].str.strip()
    df["pos_change"] = df["last_week"] - df["current_week"]

    n_before = len(df)
    df = df.drop_duplicates(subset=["chart_week", "current_week"])
    step(f"removed {n_before - len(df):,} duplicated week/position entries")

    assert df.isna().sum().sum() == 0, "cleaned Billboard data still contains NAs"
    step(f"FINAL weekly table: {len(df):,} rows x {df.shape[1]} columns, zero missing values")
    df.to_csv(os.path.join(CLEAN, "billboard_weekly_clean.csv"), index=False)

    # --- one row per song ---------------------------------------------------
    songs = (df.groupby(["title", "performer"])
               .agg(peak_position=("peak_pos", "min"),
                    weeks_on_chart=("wks_on_chart", "max"),
                    debut_week=("chart_week", "min"),
                    last_seen=("chart_week", "max"))
               .reset_index())
    songs["debut_year"] = songs["debut_week"].dt.year
    songs["debut_decade"] = (songs["debut_year"] // 10 * 10).astype(int)
    songs["reached_top10"] = (songs["peak_position"] <= 10).astype(int)
    songs["reached_number1"] = (songs["peak_position"] == 1).astype(int)
    songs["title_key"] = songs["title"].map(make_key)
    songs["artist_key"] = songs["performer"].map(
        lambda v: make_key(v, drop_featured=True))

    assert songs.isna().sum().sum() == 0
    step(f"built a song-level table: {len(songs):,} distinct songs that have "
         f"charted since 1958")
    songs.to_csv(os.path.join(CLEAN, "billboard_songs_clean.csv"), index=False)

    return df, songs


# ============================================================================
# 3. Deezer + MusicBrainz (from the API script)
# ============================================================================

def clean_api_data():
    print("\n=== Deezer / MusicBrainz (API) ===")
    tracks_path = os.path.join(RAW, "deezer_tracks_raw.csv")
    if not os.path.exists(tracks_path):
        print("  (API files not found yet - run 01_collect_api_data.py first. Skipping.)")
        return None, None, None

    t = pd.read_csv(tracks_path)
    n0 = len(t)
    step(f"loaded {n0:,} Deezer tracks")

    t = t.dropna(subset=["track_id", "title", "artist_id"])
    t = t.drop_duplicates(subset="track_id")
    step(f"kept {len(t):,} unique tracks after dropping blanks and duplicates")

    t["release_year"] = pd.to_numeric(t["release_date"].astype(str).str.slice(0, 4),
                                      errors="coerce")
    t = t[t["release_year"].between(1950, 2026)]
    t["release_year"] = t["release_year"].astype(int)
    t["release_decade"] = (t["release_year"] // 10 * 10).astype(int)

    # Deezer reports bpm = 0 when it has no tempo analysis for a track.
    n_zero = (t["bpm"] <= 0).sum()
    t["bpm_known"] = (t["bpm"] > 0).astype(int)
    t.loc[t["bpm"] <= 0, "bpm"] = np.nan
    median_bpm = t["bpm"].median()
    t["bpm"] = t["bpm"].fillna(median_bpm)
    step(f"{n_zero:,} tracks had bpm = 0 (no tempo analysis available); flagged "
         f"with bpm_known and filled with the median of {median_bpm:.0f} BPM")

    t["duration_min"] = t["duration_sec"] / 60.0
    t = t[t["duration_min"].between(0.5, 15)]
    t["explicit_lyrics"] = t["explicit_lyrics"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    t["from_chart"] = (~t["source_genre"].astype(str).str.startswith("search_")).astype(int)
    t["rank"] = pd.to_numeric(t["rank"], errors="coerce").fillna(0).astype(int)

    keep = ["track_id", "title", "artist_id", "artist_name", "album_title",
            "release_year", "release_decade", "duration_min", "bpm", "bpm_known",
            "gain", "rank", "explicit_lyrics", "source_genre", "from_chart"]
    t = t[[c for c in keep if c in t.columns]].dropna().reset_index(drop=True)
    step(f"FINAL Deezer tracks: {len(t):,} rows x {t.shape[1]} columns, zero missing values")
    t.to_csv(os.path.join(CLEAN, "deezer_tracks_clean.csv"), index=False)

    # --- artists ------------------------------------------------------------
    a = None
    apath = os.path.join(RAW, "deezer_artists_raw.csv")
    if os.path.exists(apath):
        a = pd.read_csv(apath).drop_duplicates(subset="artist_id")
        a = a.dropna(subset=["artist_id", "artist_name"])
        a["nb_fan"] = pd.to_numeric(a["nb_fan"], errors="coerce").fillna(0).astype(int)
        a["nb_album"] = pd.to_numeric(a["nb_album"], errors="coerce").fillna(0).astype(int)
        a["fan_band"] = pd.cut(a["nb_fan"],
                               bins=[-1, 1000, 50_000, 500_000, 5_000_000, 10**9],
                               labels=["Under 1K", "1K-50K", "50K-500K",
                                       "500K-5M", "Over 5M"])
        a = a[["artist_id", "artist_name", "nb_album", "nb_fan", "fan_band"]].dropna()
        step(f"FINAL Deezer artists: {len(a):,} rows, zero missing values")
        a.to_csv(os.path.join(CLEAN, "deezer_artists_clean.csv"), index=False)

    # --- MusicBrainz tags (the qualitative slice) ---------------------------
    tg = None
    tpath = os.path.join(RAW, "musicbrainz_tags_raw.csv")
    if os.path.exists(tpath):
        tg = pd.read_csv(tpath)
        n0 = len(tg)
        tg = tg[tg["tag"].notna() & (tg["tag"].astype(str).str.strip() != "")]
        tg["tag"] = tg["tag"].str.lower().str.strip()
        tg["tag"] = tg["tag"].str.replace(r"[^a-z0-9 &/-]", "", regex=True)
        tg = tg[tg["tag"].str.len() > 1]
        tg["tag_count"] = pd.to_numeric(tg["tag_count"], errors="coerce").fillna(1).astype(int)
        tg["mb_begin_year"] = pd.to_numeric(tg["mb_begin_year"], errors="coerce")
        tg["mb_country"] = tg["mb_country"].fillna("Unknown")
        tg["mb_type"] = tg["mb_type"].fillna("Unknown")
        tg["mb_begin_year"] = tg["mb_begin_year"].fillna(tg["mb_begin_year"].median()).astype(int)
        tg = tg.drop_duplicates(subset=["artist_name", "tag"])
        step(f"cleaned free-text tags: {n0:,} raw rows -> {len(tg):,} usable "
             f"artist/tag pairs covering {tg.artist_name.nunique():,} artists")
        assert tg.isna().sum().sum() == 0
        tg.to_csv(os.path.join(CLEAN, "musicbrainz_tags_clean.csv"), index=False)

    return t, a, tg


# ============================================================================
# 4. Join Spotify audio features to Billboard chart outcomes
# ============================================================================

def build_matched(spotify, bb_songs):
    print("\n=== Linked table: audio features + chart outcome ===")
    charted = bb_songs.sort_values("peak_position").drop_duplicates(
        subset=["title_key", "artist_key"], keep="first")

    merged = spotify.merge(
        charted[["title_key", "artist_key", "peak_position", "weeks_on_chart",
                 "debut_year", "reached_top10", "reached_number1"]],
        on=["title_key", "artist_key"], how="left")

    merged["charted"] = merged["peak_position"].notna().astype(int)
    merged["peak_position"] = merged["peak_position"].fillna(101).astype(int)
    merged["weeks_on_chart"] = merged["weeks_on_chart"].fillna(0).astype(int)
    merged["debut_year"] = merged["debut_year"].fillna(0).astype(int)
    merged["reached_top10"] = merged["reached_top10"].fillna(0).astype(int)
    merged["reached_number1"] = merged["reached_number1"].fillna(0).astype(int)

    n_hit = merged["charted"].sum()
    step(f"matched {n_hit:,} of {len(merged):,} songs ({n_hit/len(merged)*100:.1f}%) to a "
         f"Billboard Hot 100 entry by normalised title and artist")
    step("songs with no match are coded charted = 0 and peak_position = 101 "
         "('never entered the Hot 100'), which is a real outcome rather than a gap")

    assert merged.isna().sum().sum() == 0
    merged.to_csv(os.path.join(CLEAN, "songs_with_chart_outcome.csv"), index=False)
    step(f"FINAL linked table: {len(merged):,} rows x {merged.shape[1]} columns")
    return merged


# ============================================================================


def verify_written_files():
    """
    Re-read every cleaned file exactly the way anyone else would and confirm
    it really is free of missing values. Asserting on the in-memory frame is
    not enough: a value like "NA" survives in memory and comes back as a hole.
    """
    print("\n=== Verifying written files ===")
    clean = True
    for name in sorted(os.listdir(CLEAN)):
        if not name.endswith(".csv"):
            continue
        d = pd.read_csv(os.path.join(CLEAN, name))
        n_na = int(d.isna().sum().sum())
        if n_na:
            clean = False
            cols = d.isna().sum()[lambda x: x > 0].to_dict()
            print(f"  FAIL {name}: {n_na} missing values on re-read -> {cols}")
        else:
            print(f"  ok   {name}: {len(d):,} rows, zero missing values on re-read")
    if not clean:
        raise SystemExit("Cleaned files still contain missing values when read back.")
    print("  every cleaned file re-reads with zero missing values")


def main():
    print("=" * 74)
    print("CSCI 5612 Project Part 1 - cleaning and preparation")
    print("=" * 74)

    spotify = clean_spotify()
    bb_weekly, bb_songs = clean_billboard()
    clean_api_data()
    build_matched(spotify, bb_songs)

    verify_written_files()

    with open(os.path.join(CLEAN, "cleaning_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("Cleaning log - CSCI 5612 Project Part 1\n")
        fh.write("=" * 60 + "\n\n")
        fh.write("\n".join(f"- {line}" for line in log))

    print("\n" + "=" * 74)
    print("Cleaning complete. Files written to ../data/clean/")
    print("=" * 74)


if __name__ == "__main__":
    main()
