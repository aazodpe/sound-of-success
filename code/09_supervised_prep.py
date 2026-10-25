"""
CSCI 5612 - Project Part 3
Script 09: Build the labelled train/test sets for the supervised methods.

Supervised learning needs two things unsupervised learning did not: a label to
predict, and a split between the data used to build a model and the data used
to judge it. This script creates both, for two separate prediction tasks.

  Task A - genre   : six classes, the question clustering failed to answer
  Task B - charted : binary, the question the whole project is about

Reads  ../data/clean/spotify_songs_clean.csv
       ../data/clean/songs_with_chart_outcome.csv
Writes ../data/clean/supervised_<task>_<train|test>.csv
       ../data/clean/split_summary.json

Run: python 09_supervised_prep.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "data", "clean")
SEED = 5612
TEST_SIZE = 0.20

FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "duration_min"]

S = {}


def build(df, label_col, task_name):
    """Stratified, disjoint split of one labelled table."""
    data = df[FEATURES + [label_col]].dropna().reset_index(drop=True)
    data = data.rename(columns={label_col: "label"})

    train, test = train_test_split(
        data, test_size=TEST_SIZE, random_state=SEED, stratify=data["label"])

    # Disjointness is guaranteed by construction, but assert it rather than
    # assume it: a row appearing in both sets would silently inflate accuracy.
    assert set(train.index).isdisjoint(set(test.index)), "train and test overlap"
    assert len(train) + len(test) == len(data), "rows lost in the split"

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train.to_csv(os.path.join(CLEAN, f"supervised_{task_name}_train.csv"), index=False)
    test.to_csv(os.path.join(CLEAN, f"supervised_{task_name}_test.csv"), index=False)

    tr_counts = train.label.value_counts().sort_index()
    te_counts = test.label.value_counts().sort_index()
    print(f"\n  {task_name}: {len(data):,} labelled rows "
          f"-> {len(train):,} train / {len(test):,} test "
          f"({TEST_SIZE:.0%} held out)")
    print(f"    classes: {list(tr_counts.index)}")
    print(f"    train balance: {dict(tr_counts)}")
    print(f"    test  balance: {dict(te_counts)}")

    S[task_name] = {
        "n_total": int(len(data)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_fraction": TEST_SIZE,
        "n_features": len(FEATURES),
        "classes": [str(c) for c in tr_counts.index],
        "train_counts": {str(k): int(v) for k, v in tr_counts.items()},
        "test_counts": {str(k): int(v) for k, v in te_counts.items()},
        "majority_class": str(te_counts.idxmax()),
        "baseline_accuracy": float(te_counts.max() / te_counts.sum()),
        "disjoint_verified": True,
    }
    return train, test


def main():
    print("=" * 70)
    print("CSCI 5612 Project Part 3 - supervised data preparation")
    print("=" * 70)

    songs = pd.read_csv(os.path.join(CLEAN, "spotify_songs_clean.csv"))
    outcome = pd.read_csv(os.path.join(CLEAN, "songs_with_chart_outcome.csv"))
    print(f"  loaded {len(songs):,} labelled songs")

    build(songs, "playlist_genre", "genre")
    build(outcome, "charted", "charted")

    S["features"] = FEATURES
    S["seed"] = SEED
    with open(os.path.join(CLEAN, "split_summary.json"), "w") as fh:
        json.dump(S, fh, indent=2)

    print("\n" + "=" * 70)
    print("Splits written. Both verified disjoint and stratified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
