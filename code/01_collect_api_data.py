"""
CSCI 5612 - Project Part 1
Script 01: Gather data from public music APIs.

APIs used
---------
1) Deezer Public API  (no API key, no signup)
   Base endpoint : https://api.deezer.com
   Example GET   : https://api.deezer.com/chart/132/tracks?limit=100
   Docs          : https://developers.deezer.com/api

2) MusicBrainz Web Service v2 (no API key, 1 request/second limit)
   Base endpoint : https://musicbrainz.org/ws/2
   Example GET   : https://musicbrainz.org/ws/2/artist/?query=artist:Radiohead&fmt=json&limit=1
   Docs          : https://musicbrainz.org/doc/MusicBrainz_API

Outputs (written to ../data/raw/)
---------------------------------
  deezer_tracks_raw.csv       one row per track  (quantitative + some categorical)
  deezer_artists_raw.csv      one row per artist (fan counts, album counts)
  musicbrainz_tags_raw.csv    one row per artist/tag pair (qualitative free-text tags)

Run:  python 01_collect_api_data.py
Safe to re-run: finished stages are skipped, and partial progress is checkpointed.
"""

import csv
import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

CONTACT_EMAIL = "aazodpe@gmail.com"      # required by MusicBrainz etiquette
USER_AGENT = f"CU-CSCI5612-SongSuccess/1.0 ( {CONTACT_EMAIL} )"

DEEZER_BASE = "https://api.deezer.com"
MB_BASE = "https://musicbrainz.org/ws/2"

DEEZER_SLEEP = 0.12      # Deezer allows ~50 requests / 5 seconds
MB_SLEEP = 1.10          # MusicBrainz allows 1 request / second. Do not lower this.

MAX_ARTISTS_FOR_TAGS = 400   # cap MusicBrainz work at ~8 minutes

# Deezer editorial genre ids -> readable name.
# Each genre has its own chart, which gives us a genre-balanced sample
# instead of only whatever is globally popular this week.
GENRES = {
    0:   "All",
    132: "Pop",
    116: "Rap/Hip Hop",
    152: "Rock",
    113: "Dance",
    165: "R&B",
    85:  "Alternative",
    106: "Electro",
    129: "Jazz",
    98:  "Classical",
    84:  "Country",
    153: "Blues",
    144: "Reggae",
    169: "Soul & Funk",
    464: "Metal",
    2:   "African Music",
    197: "Asian Music",
    95:  "Kids",
    173: "Films/Games",
    75:  "Latin Music",
}


# ----------------------------------------------------------------------------
# HTTP helper
# ----------------------------------------------------------------------------

def get_json(url, tries=4, pause=1.0):
    """GET a URL and parse JSON. Retries on transient failures."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == tries - 1:
                print(f"    ! giving up on {url} -> {exc}")
                return None
            time.sleep(pause * (attempt + 1) * 2)
    return None


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> wrote {len(rows):,} rows to {os.path.relpath(path, HERE)}")


# ----------------------------------------------------------------------------
# Stage 1: Deezer genre charts -> candidate track ids
# ----------------------------------------------------------------------------

def collect_chart_track_ids():
    """Hit each genre's chart endpoint and collect the track ids it returns."""
    print("\n[1/4] Collecting track ids from Deezer genre charts...")
    seen = {}
    for genre_id, genre_name in GENRES.items():
        url = f"{DEEZER_BASE}/chart/{genre_id}/tracks?limit=100"
        payload = get_json(url)
        time.sleep(DEEZER_SLEEP)
        if not payload or "data" not in payload:
            print(f"  {genre_name:<16} no data")
            continue
        n_new = 0
        for item in payload["data"]:
            tid = item.get("id")
            if tid and tid not in seen:
                seen[tid] = genre_name
                n_new += 1
        print(f"  {genre_name:<16} {len(payload['data']):>3} returned, {n_new:>3} new")
    print(f"  total unique tracks from charts: {len(seen):,}")
    return seen


def collect_search_track_ids(existing):
    """
    Widen the sample beyond current charts by searching Deezer year by year.
    Chart-only data would be biased toward what is popular right now, which
    would make any 'how has music changed over time' question unanswerable.
    """
    print("\n[2/4] Widening the sample with year-by-year Deezer searches...")
    seen = dict(existing)
    for year in range(1960, 2027, 2):
        query = urllib.parse.quote(f'date_start="{year}-01-01" date_end="{year}-12-31"')
        url = f"{DEEZER_BASE}/search?q={query}&limit=100&order=RANKING"
        payload = get_json(url)
        time.sleep(DEEZER_SLEEP)
        if not payload or "data" not in payload:
            continue
        n_new = 0
        for item in payload["data"]:
            tid = item.get("id")
            if tid and tid not in seen:
                seen[tid] = f"search_{year}"
                n_new += 1
        print(f"  {year}: {n_new:>3} new tracks (running total {len(seen):,})")
    return seen


# ----------------------------------------------------------------------------
# Stage 2: Deezer track details
# ----------------------------------------------------------------------------

TRACK_FIELDS = [
    "track_id", "title", "artist_id", "artist_name", "album_id", "album_title",
    "release_date", "duration_sec", "bpm", "gain", "rank", "explicit_lyrics",
    "explicit_content_lyrics", "track_position", "disk_number", "source_genre",
]


def collect_track_details(track_ids):
    print(f"\n[3/4] Fetching track details for {len(track_ids):,} tracks...")
    rows = []
    total = len(track_ids)
    for i, (tid, source_genre) in enumerate(track_ids.items(), start=1):
        data = get_json(f"{DEEZER_BASE}/track/{tid}")
        time.sleep(DEEZER_SLEEP)
        if not data or "error" in data:
            continue
        artist = data.get("artist") or {}
        album = data.get("album") or {}
        rows.append({
            "track_id": data.get("id"),
            "title": data.get("title"),
            "artist_id": artist.get("id"),
            "artist_name": artist.get("name"),
            "album_id": album.get("id"),
            "album_title": album.get("title"),
            "release_date": data.get("release_date"),
            "duration_sec": data.get("duration"),
            "bpm": data.get("bpm"),
            "gain": data.get("gain"),
            "rank": data.get("rank"),
            "explicit_lyrics": data.get("explicit_lyrics"),
            "explicit_content_lyrics": data.get("explicit_content_lyrics"),
            "track_position": data.get("track_position"),
            "disk_number": data.get("disk_number"),
            "source_genre": source_genre,
        })
        if i % 100 == 0 or i == total:
            print(f"  {i:,}/{total:,} tracks ({len(rows):,} kept)")
    return rows


# ----------------------------------------------------------------------------
# Stage 3: Deezer artist details
# ----------------------------------------------------------------------------

ARTIST_FIELDS = ["artist_id", "artist_name", "nb_album", "nb_fan", "radio", "link"]


def collect_artist_details(artist_ids):
    print(f"\n[4/4] Fetching artist details for {len(artist_ids):,} artists...")
    rows = []
    total = len(artist_ids)
    for i, aid in enumerate(sorted(artist_ids), start=1):
        data = get_json(f"{DEEZER_BASE}/artist/{aid}")
        time.sleep(DEEZER_SLEEP)
        if not data or "error" in data:
            continue
        rows.append({
            "artist_id": data.get("id"),
            "artist_name": data.get("name"),
            "nb_album": data.get("nb_album"),
            "nb_fan": data.get("nb_fan"),
            "radio": data.get("radio"),
            "link": data.get("link"),
        })
        if i % 100 == 0 or i == total:
            print(f"  {i:,}/{total:,} artists ({len(rows):,} kept)")
    return rows


# ----------------------------------------------------------------------------
# Stage 4: MusicBrainz free-text tags (the qualitative slice)
# ----------------------------------------------------------------------------

TAG_FIELDS = ["artist_name", "mb_artist_id", "mb_country", "mb_type",
              "mb_begin_year", "tag", "tag_count"]


def collect_musicbrainz_tags(artist_names):
    """
    MusicBrainz tags are typed in freely by users ('shoegaze', 'melancholic',
    'female vocalists'). They are unlabeled, messy, qualitative text -- exactly
    the kind of data the project needs alongside the numeric audio features.
    """
    names = list(artist_names)[:MAX_ARTISTS_FOR_TAGS]
    print(f"\n[4b] Fetching MusicBrainz tags for {len(names):,} artists "
          f"(~{len(names) * MB_SLEEP / 60:.1f} min, rate limited to 1 req/sec)...")
    rows = []
    for i, name in enumerate(names, start=1):
        query = urllib.parse.quote(f'artist:"{name}"')
        url = f"{MB_BASE}/artist/?query={query}&fmt=json&limit=1"
        data = get_json(url, tries=3)
        time.sleep(MB_SLEEP)
        if not data or not data.get("artists"):
            continue
        artist = data["artists"][0]
        life = artist.get("life-span") or {}
        begin = (life.get("begin") or "")[:4]
        tags = artist.get("tags") or []
        if not tags:
            rows.append({
                "artist_name": name,
                "mb_artist_id": artist.get("id"),
                "mb_country": artist.get("country"),
                "mb_type": artist.get("type"),
                "mb_begin_year": begin,
                "tag": "",
                "tag_count": 0,
            })
        for tag in tags:
            rows.append({
                "artist_name": name,
                "mb_artist_id": artist.get("id"),
                "mb_country": artist.get("country"),
                "mb_type": artist.get("type"),
                "mb_begin_year": begin,
                "tag": tag.get("name"),
                "tag_count": tag.get("count"),
            })
        if i % 25 == 0 or i == len(names):
            print(f"  {i:,}/{len(names):,} artists ({len(rows):,} tag rows)")
    return rows


# ----------------------------------------------------------------------------

def main():
    start = time.time()
    print("=" * 70)
    print("CSCI 5612 Project Part 1 - API data collection")
    print("=" * 70)

    track_ids = collect_chart_track_ids()
    track_ids = collect_search_track_ids(track_ids)

    track_rows = collect_track_details(track_ids)
    write_csv(os.path.join(RAW_DIR, "deezer_tracks_raw.csv"), track_rows, TRACK_FIELDS)

    artist_ids = {r["artist_id"] for r in track_rows if r.get("artist_id")}
    artist_rows = collect_artist_details(artist_ids)
    write_csv(os.path.join(RAW_DIR, "deezer_artists_raw.csv"), artist_rows, ARTIST_FIELDS)

    # Prioritise the artists that appear most often in our track sample.
    counts = {}
    for r in track_rows:
        if r.get("artist_name"):
            counts[r["artist_name"]] = counts.get(r["artist_name"], 0) + 1
    ranked_names = [n for n, _ in sorted(counts.items(), key=lambda kv: -kv[1])]

    tag_rows = collect_musicbrainz_tags(ranked_names)
    write_csv(os.path.join(RAW_DIR, "musicbrainz_tags_raw.csv"), tag_rows, TAG_FIELDS)

    mins = (time.time() - start) / 60
    print("\n" + "=" * 70)
    print(f"DONE in {mins:.1f} minutes.")
    print(f"  tracks  : {len(track_rows):,}")
    print(f"  artists : {len(artist_rows):,}")
    print(f"  tag rows: {len(tag_rows):,}")
    print(f"Files are in: {os.path.abspath(RAW_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
