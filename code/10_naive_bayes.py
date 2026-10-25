"""
CSCI 5612 - Project Part 3
Script 10: Naive Bayes.

Three variants are fitted, because they differ in what they assume a feature
IS, not merely in their settings:

  MultinomialNB - treats each feature as a non-negative count or weight
  BernoulliNB   - treats each feature as present/absent
  GaussianNB    - treats each feature as a bell curve per class

Only Multinomial is required by the assignment; the other two are fitted so the
description of each can be checked rather than asserted.

Reads  ../data/clean/supervised_<task>_<train|test>.csv
Writes ../data/clean/nb_metrics.json
       ../data/clean/nb_input_sample.csv
       ../docs/images/fig3x_*.png
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.preprocessing import MinMaxScaler, Binarizer

import theme

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "data", "clean")
IMG = os.path.join(HERE, "..", "docs", "images")
SEED = 5612

TASKS = {
    "genre":   {"nice": "genre", "labels": None},
    "charted": {"nice": "charting", "labels": ["Did not chart", "Charted"]},
}

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


def load(task):
    tr = pd.read_csv(os.path.join(CLEAN, f"supervised_{task}_train.csv"))
    te = pd.read_csv(os.path.join(CLEAN, f"supervised_{task}_test.csv"))
    feats = [c for c in tr.columns if c != "label"]
    return tr[feats], tr["label"], te[feats], te["label"], feats


def fit_all(task):
    Xtr, ytr, Xte, yte, feats = load(task)

    # MultinomialNB cannot accept negative numbers, so the features are scaled
    # into [0, 1]. The scaler is fitted on the TRAINING data only and then
    # applied to the test data; fitting it on everything would leak information
    # about the held-out songs into the model.
    mm = MinMaxScaler().fit(Xtr)
    Xtr_mm, Xte_mm = mm.transform(Xtr), np.clip(mm.transform(Xte), 0, 1)

    # BernoulliNB wants present/absent, so each feature is split at its median
    # in the training data: "above typical" or not.
    thresh = np.median(Xtr.values, axis=0)
    Xtr_bin = (Xtr.values > thresh).astype(int)
    Xte_bin = (Xte.values > thresh).astype(int)

    runs = {
        "MultinomialNB": (MultinomialNB(), Xtr_mm, Xte_mm),
        "BernoulliNB":   (BernoulliNB(),   Xtr_bin, Xte_bin),
        "GaussianNB":    (GaussianNB(),    Xtr.values, Xte.values),
    }

    out = {}
    for name, (model, A, B) in runs.items():
        model.fit(A, ytr)
        pred = model.predict(B)
        acc = accuracy_score(yte, pred)
        out[name] = {
            "accuracy": float(acc),
            "confusion": confusion_matrix(yte, pred).tolist(),
            "classes": [str(c) for c in model.classes_],
            "report": classification_report(yte, pred, output_dict=True,
                                            zero_division=0),
            "predictions": pred,
        }
        print(f"    {name:<16} accuracy {acc:.4f}")

    # Keep a small image-able sample of exactly what MultinomialNB received.
    if task == "genre":
        samp = pd.DataFrame(Xtr_mm[:8], columns=feats).round(4)
        samp.insert(0, "label", ytr.values[:8])
        samp.to_csv(os.path.join(CLEAN, "nb_input_sample.csv"), index=False)

    return out, yte, feats


def plot_confusion(cm, classes, title, subtitle, name, pct=True):
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    n = len(classes)
    fig, ax = plt.subplots(figsize=(1.05 * n + 4.0, 0.95 * n + 3.1))
    im = ax.imshow(cmn, cmap=theme.SEQ_CMAP, vmin=0, vmax=100)
    ax.set_xticks(range(n), classes, rotation=35, ha="right")
    ax.set_yticks(range(n), classes)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            txt = f"{cmn[i, j]:.0f}%\n{cm[i, j]:,}" if pct else f"{cm[i, j]:,}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.6,
                    color="#ffffff" if cmn[i, j] > 55 else theme.INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("Share of the true class (%)", color=theme.INK_SOFT, fontsize=9.2)
    cb.outline.set_visible(False)
    ax.set_title(f"{title}\n", loc="left")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.8,
            color=theme.INK_SOFT, va="bottom")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    save(fig, name)


def main():
    print("=" * 70)
    print("CSCI 5612 Project Part 3 - Naive Bayes")
    print("=" * 70)

    split = json.load(open(os.path.join(CLEAN, "split_summary.json")))
    results = {}
    for task in TASKS:
        print(f"\n  task: {task}")
        results[task], yte, feats = fit_all(task)
        M[task] = {k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
                   for k, v in results[task].items()}
        M[task]["baseline"] = split[task]["baseline_accuracy"]
        M[task]["n_test"] = split[task]["n_test"]
        print(f"    always-guess-majority baseline: "
              f"{split[task]['baseline_accuracy']:.4f}")

    for mode in theme.MODES:
        theme.set_theme(mode)
        C = theme.C
        print(f"\n  --- figures ({mode}) ---")

        # 30. genre confusion matrix for the required Multinomial model
        g = results["genre"]["MultinomialNB"]
        plot_confusion(
            np.array(g["confusion"]), [c.upper() for c in g["classes"]],
            "Multinomial Naive Bayes struggles to tell the genres apart",
            f"Confusion matrix on {split['genre']['n_test']:,} held-out songs. "
            f"Overall accuracy {g['accuracy']*100:.1f}% against a "
            f"{split['genre']['baseline_accuracy']*100:.1f}% baseline.",
            "fig30_nb_confusion_genre.png")

        # 31. charted confusion matrix
        c = results["charted"]["MultinomialNB"]
        plot_confusion(
            np.array(c["confusion"]), TASKS["charted"]["labels"],
            "On charting, the model never once predicts a hit",
            f"Confusion matrix on {split['charted']['n_test']:,} held-out songs, "
            f"of which only {split['charted']['test_counts']['1']:,} actually charted.",
            "fig31_nb_confusion_charted.png")

        # 32. the three variants side by side, against the baseline
        fig, ax = plt.subplots(figsize=(8.6, 4.6))
        names = ["MultinomialNB", "BernoulliNB", "GaussianNB"]
        x = np.arange(len(names)); w = 0.38
        gvals = [results["genre"][n]["accuracy"] * 100 for n in names]
        cvals = [results["charted"][n]["accuracy"] * 100 for n in names]
        ax.bar(x - w/2, gvals, w, color=C[0], edgecolor=theme.BG, linewidth=1.6,
               label="Genre (6 classes)")
        ax.bar(x + w/2, cvals, w, color=C[2], edgecolor=theme.BG, linewidth=1.6,
               label="Charted (2 classes)")
        for xi, v in zip(x - w/2, gvals):
            ax.text(xi, v + 1.4, f"{v:.1f}", ha="center", fontsize=9,
                    fontweight="bold", color=theme.INK_SOFT)
        for xi, v in zip(x + w/2, cvals):
            ax.text(xi, v + 1.4, f"{v:.1f}", ha="center", fontsize=9,
                    fontweight="bold", color=theme.INK_SOFT)
        gb = split["genre"]["baseline_accuracy"] * 100
        cb = split["charted"]["baseline_accuracy"] * 100
        ax.axhline(gb, color=C[0], linestyle=":", linewidth=1.8,
                   label=f"Genre baseline ({gb:.0f}%)")
        ax.axhline(cb, color=C[2], linestyle="--", linewidth=1.8,
                   label=f"Charted baseline ({cb:.0f}%)")
        ax.set_xticks(x, names)
        ax.set_ylim(0, 112)
        ax.legend(ncols=2, loc="upper left", fontsize=8.8)
        finish(ax, "What a feature is assumed to be changes the answer",
               "Held-out accuracy of the three Naive Bayes variants on both tasks, "
               "against the accuracy of simply always guessing the commonest class",
               None, "Accuracy (%)")
        save(fig, "fig32_nb_variants.png")

        # 33. per-genre recall: which genres are recoverable at all?
        rep = results["genre"]["MultinomialNB"]["report"]
        cls = results["genre"]["MultinomialNB"]["classes"]
        rec = [rep[c]["recall"] * 100 for c in cls]
        order = np.argsort(rec)
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        cols = [C[i % len(C)] for i in range(len(cls))]
        ax.barh(range(len(cls)), [rec[i] for i in order], height=0.62,
                color=[cols[i] for i in order], edgecolor=theme.BG, linewidth=1.6)
        ax.set_yticks(range(len(cls)), [cls[i].upper() for i in order])
        for i, v in enumerate([rec[j] for j in order]):
            ax.text(v + 1.2, i, f"{v:.0f}%", va="center", fontsize=9,
                    color=theme.INK_SOFT, fontweight="bold")
        ax.axvline(100 / len(cls), color=theme.INK_SOFT, linestyle=":", linewidth=1.6)
        ax.annotate("random guessing", xy=(100/len(cls), len(cls)-0.4),
                    xytext=(5, 0), textcoords="offset points", fontsize=8.8,
                    color=theme.INK_SOFT)
        ax.set_xlim(0, max(rec) * 1.25)
        finish(ax, "Some genres are far easier to recognise than others",
               "Share of each genre's held-out songs that Multinomial Naive Bayes "
               "labelled correctly",
               "Recall (%)", None)
        save(fig, "fig33_nb_recall_by_genre.png")

    with open(os.path.join(CLEAN, "nb_metrics.json"), "w") as fh:
        json.dump(M, fh, indent=2, default=float)

    print("\n" + "=" * 70)
    print("Naive Bayes complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
