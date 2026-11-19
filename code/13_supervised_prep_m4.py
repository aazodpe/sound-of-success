"""
CSCI 5612 - Project Part 4
Script 13: Datasets for regression and AdaBoost.

Three things are added on top of the Module 3 preparation:

  1. A VALIDATION set, carved out of the training data. The assignment now
     requires that settings be chosen without touching the final test set, so
     every tuning decision in this module is made on validation data and the
     test set is read exactly once, at the end.

  2. A NUMERIC target. Linear regression predicts a number, so it cannot use
     the charted flag; it predicts track_popularity (0-100) instead.

  3. ONE-HOT ENCODED categorical columns. AdaBoost in scikit-learn needs
     numeric inputs, so genre, musical key and mode are expanded into
     indicator columns rather than dropped.

The row split is reproduced with the same seed and the same stratification as
Module 3, so the held-out songs here are literally the same songs. That is what
makes the comparisons against Naive Bayes and the decision tree meaningful.

Writes ../data/clean/m4_*.csv  and  ../data/clean/m4_split_summary.json
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
VAL_SIZE = 0.25          # of the training set, i.e. 20% of everything

NUMERIC = ["danceability", "energy", "loudness", "speechiness", "acousticness",
           "instrumentalness", "liveness", "valence", "tempo", "duration_min"]
CATEGORICAL = ["playlist_genre", "key_name", "mode_name"]

S = {}


def main():
    print("=" * 72)
    print("CSCI 5612 Project Part 4 - data preparation")
    print("=" * 72)

    df = pd.read_csv(os.path.join(CLEAN, "songs_with_chart_outcome.csv"))
    df = df[NUMERIC + CATEGORICAL + ["charted", "track_popularity"]].dropna()
    df = df.reset_index(drop=True)
    print(f"  loaded {len(df):,} labelled songs")

    # --- the same split as Module 3, reproduced ----------------------------
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE,
                                      random_state=SEED, stratify=df["charted"])

    # Confirm this really is the Module 3 split rather than merely a similar
    # one, by checking the held-out set matches row for row.
    m3 = pd.read_csv(os.path.join(CLEAN, "supervised_charted_test.csv"))
    here = df.loc[te_idx, NUMERIC].round(6).reset_index(drop=True)
    there = m3[NUMERIC].round(6).reset_index(drop=True)
    same = here.equals(there)
    print(f"  test set identical to Module 3: {same}")
    S["matches_module3_test"] = bool(same)

    # --- carve validation out of training ----------------------------------
    tr_idx, va_idx = train_test_split(tr_idx, test_size=VAL_SIZE,
                                      random_state=SEED,
                                      stratify=df.loc[tr_idx, "charted"])

    for a, b, name in [(tr_idx, va_idx, "train/val"), (tr_idx, te_idx, "train/test"),
                       (va_idx, te_idx, "val/test")]:
        assert set(a).isdisjoint(set(b)), f"{name} overlap"
    assert len(tr_idx) + len(va_idx) + len(te_idx) == len(df)
    print(f"  split: {len(tr_idx):,} train / {len(va_idx):,} validation / "
          f"{len(te_idx):,} test  (all three verified disjoint)")

    # --- one-hot encode the categorical columns ----------------------------
    dummies = pd.get_dummies(df[CATEGORICAL], prefix=CATEGORICAL, dtype=int)
    print(f"  one-hot encoding: {len(CATEGORICAL)} categorical columns "
          f"-> {dummies.shape[1]} indicator columns")
    for c in CATEGORICAL:
        cols = [d for d in dummies.columns if d.startswith(c + "_")]
        print(f"    {c:<16} {df[c].nunique():>2} levels -> {len(cols):>2} columns")

    encoded = pd.concat([df[NUMERIC], dummies], axis=1)
    S["n_numeric"] = len(NUMERIC)
    S["n_onehot"] = int(dummies.shape[1])
    S["n_features_encoded"] = int(encoded.shape[1])
    S["onehot_columns"] = list(dummies.columns)

    # A small before/after sample so the encoding step can be shown.
    before = df[CATEGORICAL].head(8).reset_index(drop=True)
    after = dummies.head(8).reset_index(drop=True)
    before.to_csv(os.path.join(CLEAN, "m4_encoding_before.csv"), index=False)
    after.to_csv(os.path.join(CLEAN, "m4_encoding_after.csv"), index=False)

    # --- write every split ------------------------------------------------
    for part, ids in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
        # Regression: numeric features, numeric target
        reg = df.loc[ids, NUMERIC].copy()
        reg.insert(0, "track_popularity", df.loc[ids, "track_popularity"].values)
        reg.to_csv(os.path.join(CLEAN, f"m4_regression_{part}.csv"), index=False)

        # Logistic: numeric features, two-class target
        log = df.loc[ids, NUMERIC].copy()
        log.insert(0, "charted", df.loc[ids, "charted"].values)
        log.to_csv(os.path.join(CLEAN, f"m4_logistic_{part}.csv"), index=False)

        # AdaBoost: encoded features, two-class target
        ada = encoded.loc[ids].copy()
        ada.insert(0, "charted", df.loc[ids, "charted"].values)
        ada.to_csv(os.path.join(CLEAN, f"m4_adaboost_{part}.csv"), index=False)

        S[part] = {
            "n": int(len(ids)),
            "charted": int(df.loc[ids, "charted"].sum()),
            "charted_rate": float(df.loc[ids, "charted"].mean()),
            "popularity_mean": float(df.loc[ids, "track_popularity"].mean()),
            "popularity_sd": float(df.loc[ids, "track_popularity"].std()),
        }
        print(f"    {part:<6} n={len(ids):>6,}  charted={S[part]['charted']:>5,} "
              f"({S[part]['charted_rate']*100:.1f}%)  "
              f"popularity mean {S[part]['popularity_mean']:.1f}")

    S.update({"seed": SEED, "test_size": TEST_SIZE, "val_size_of_train": VAL_SIZE,
              "numeric_features": NUMERIC, "categorical_features": CATEGORICAL,
              "regression_target": "track_popularity",
              "classification_target": "charted",
              "baseline_accuracy": float(1 - df.loc[te_idx, "charted"].mean())})
    with open(os.path.join(CLEAN, "m4_split_summary.json"), "w") as fh:
        json.dump(S, fh, indent=2)

    print("\n" + "=" * 72)
    print("Done. Test set untouched from here until the final evaluation.")
    print("=" * 72)


if __name__ == "__main__":
    main()
