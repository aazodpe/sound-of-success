"""
CSCI 5612 - Project Part 4
Script 15: AdaBoost.

Every setting here is chosen on the VALIDATION set. The test set is read once,
at the very end, with the chosen configuration, so the final number is an
honest estimate rather than the best of many attempts.

Two feature sets are compared, because the one-hot encoding step the assignment
asks for is also a real experiment: does telling the model a song's genre, key
and mode help it predict charting?

Writes ../data/clean/adaboost_metrics.json
       ../docs/images/fig44..fig48
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score,
                             classification_report)
from sklearn.tree import DecisionTreeClassifier

import theme

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "data", "clean")
IMG = os.path.join(HERE, "..", "docs", "images")
SEED = 5612

NUMERIC = ["danceability", "energy", "loudness", "speechiness", "acousticness",
           "instrumentalness", "liveness", "valence", "tempo", "duration_min"]

ROUNDS = [1, 5, 10, 25, 50, 100, 200, 400]
RATES = [0.05, 0.1, 0.3, 0.5, 1.0]

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


def load(part, encoded=True):
    d = pd.read_csv(os.path.join(CLEAN, f"m4_adaboost_{part}.csv"))
    y = d["charted"]
    X = d.drop(columns=["charted"])
    if not encoded:
        X = X[NUMERIC]
    return X, y


def main():
    print("=" * 72)
    print("CSCI 5612 Project Part 4 - AdaBoost")
    print("=" * 72)

    Xtr, ytr = load("train")
    Xva, yva = load("val")
    Xte, yte = load("test")
    feats = list(Xtr.columns)
    baseline = float(1 - yte.mean())
    print(f"  {len(Xtr):,} train / {len(Xva):,} validation / {len(Xte):,} test")
    print(f"  {len(feats)} features ({len(NUMERIC)} numeric + "
          f"{len(feats)-len(NUMERIC)} one-hot)")
    print(f"  baseline accuracy (always 'did not chart'): {baseline:.4f}")

    # --- tune on validation, never on test --------------------------------
    print(f"\n  tuning on the validation set: {len(ROUNDS)} round counts "
          f"x {len(RATES)} learning rates")
    grid = []
    for lr in RATES:
        for n in ROUNDS:
            clf = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),  # a decision stump
                n_estimators=n, learning_rate=lr, random_state=SEED).fit(Xtr, ytr)
            pv = clf.predict(Xva)
            grid.append({
                "learning_rate": lr, "n_estimators": n,
                "val_accuracy": float(accuracy_score(yva, pv)),
                "val_auc": float(roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])),
            })
        print(f"    learning_rate {lr}: "
              f"best val AUC {max(g['val_auc'] for g in grid if g['learning_rate']==lr):.4f}")

    # Accuracy is useless for choosing here, because every setting can reach the
    # baseline by predicting 'no'. AUC measures whether the model ranks songs
    # correctly, which is the thing that can actually differ.
    best = max(grid, key=lambda g: g["val_auc"])
    print(f"\n  chosen on validation AUC: {best['n_estimators']} rounds, "
          f"learning rate {best['learning_rate']} (val AUC {best['val_auc']:.4f})")

    # --- the one-hot question ---------------------------------------------
    print("\n  does the one-hot encoded information help?")
    compare = {}
    for name, enc in [("numeric only", False), ("with one-hot", True)]:
        A, a = load("train", enc); B, b = load("val", enc)
        clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=best["n_estimators"], learning_rate=best["learning_rate"],
            random_state=SEED).fit(A, a)
        auc = float(roc_auc_score(b, clf.predict_proba(B)[:, 1]))
        compare[name] = {"n_features": int(A.shape[1]), "val_auc": auc}
        print(f"    {name:<14} {A.shape[1]:>2} features -> validation AUC {auc:.4f}")

    # --- final model, test set read once ----------------------------------
    final = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=best["n_estimators"], learning_rate=best["learning_rate"],
        random_state=SEED).fit(Xtr, ytr)
    pred = final.predict(Xte)
    prob = final.predict_proba(Xte)[:, 1]
    acc = float(accuracy_score(yte, pred))
    auc = float(roc_auc_score(yte, prob))
    rep = classification_report(yte, pred, output_dict=True, zero_division=0)
    print(f"\n  FINAL on the held-out test set:")
    print(f"    accuracy {acc:.4f}  (baseline {baseline:.4f})")
    print(f"    ROC AUC  {auc:.4f}")
    print(f"    recall on charted {rep['1']['recall']:.3f}, "
          f"precision {rep['1']['precision']:.3f}")

    # --- compare against the Module 3 decision tree -----------------------
    dt = json.load(open(os.path.join(CLEAN, "dt_metrics.json")))
    dt_best_key = max(["shallow", "medium", "full"],
                      key=lambda k: dt["charted"][k]["test_accuracy"])
    dt_best = dt["charted"][dt_best_key]
    print(f"\n  Module 3 best tree on the same songs: "
          f"{dt_best['name']} accuracy {dt_best['test_accuracy']:.4f}")
    print(f"  AdaBoost minus that tree: "
          f"{(acc - dt_best['test_accuracy'])*100:+.2f} percentage points")

    imp = pd.Series(final.feature_importances_, index=feats).sort_values(ascending=False)

    M.update({
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
        "n_features": len(feats), "n_numeric": len(NUMERIC),
        "n_onehot": len(feats) - len(NUMERIC),
        "baseline": baseline, "grid": grid, "best": best,
        "onehot_comparison": compare,
        "test_accuracy": acc, "test_auc": auc,
        "test_recall_charted": float(rep["1"]["recall"]),
        "test_precision_charted": float(rep["1"]["precision"]),
        "confusion": confusion_matrix(yte, pred).tolist(),
        "n_charted_test": int(yte.sum()),
        "dt_comparison": {"name": dt_best["name"],
                          "test_accuracy": dt_best["test_accuracy"],
                          "difference_pp": (acc - dt_best["test_accuracy"]) * 100},
        "importances": {f: float(v) for f, v in imp.items()},
        "stump_alphas": [float(w) for w in final.estimator_weights_[:60]],
    })

    for mode in theme.MODES:
        theme.set_theme(mode)
        C = theme.C
        print(f"\n  --- figures ({mode}) ---")

        # 44. validation curve: how accuracy and ranking change with rounds
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 6.8), sharex=True)
        for i, lr in enumerate(RATES):
            sub = [g for g in grid if g["learning_rate"] == lr]
            a1.plot([g["n_estimators"] for g in sub],
                    [g["val_accuracy"] * 100 for g in sub],
                    color=C[i % len(C)], linewidth=2.1, marker="o", markersize=5,
                    markeredgecolor=theme.BG, markeredgewidth=1.2,
                    label=f"rate {lr}")
            a2.plot([g["n_estimators"] for g in sub],
                    [g["val_auc"] for g in sub],
                    color=C[i % len(C)], linewidth=2.1, marker="o", markersize=5,
                    markeredgecolor=theme.BG, markeredgewidth=1.2)
        a1.axhline(baseline * 100, color=theme.INK_SOFT, linestyle="--", linewidth=1.6,
                   label=f"baseline ({baseline*100:.1f}%)")
        a1.set_xscale("log"); a1.set_xticks(ROUNDS, [str(r) for r in ROUNDS])
        a1.legend(ncols=3, fontsize=8.4, loc="lower right")
        finish(a1, "Adding rounds barely moves accuracy",
               "Validation accuracy as boosting rounds are added, at five learning "
               "rates. Everything sits on the baseline.",
               None, "Validation accuracy (%)")
        a2.scatter([best["n_estimators"]], [best["val_auc"]], s=170, facecolor="none",
                   edgecolor=C[1], linewidth=2.2, zorder=6)
        a2.annotate(f"chosen: {best['n_estimators']} rounds, rate {best['learning_rate']}",
                    xy=(best["n_estimators"], best["val_auc"]), xytext=(-18, -62),
                    textcoords="offset points", ha="right", va="top", color=C[1],
                    fontweight="bold", fontsize=9.3,
                    arrowprops=dict(arrowstyle="-", color=C[1], linewidth=1.1,
                                    shrinkA=2, shrinkB=10))
        a2.axhline(0.5, color=theme.INK_SOFT, linestyle=":", linewidth=1.5)
        a2.annotate("no better than chance", xy=(1.2, 0.5), xytext=(0, 5),
                    textcoords="offset points", fontsize=8.8, color=theme.INK_SOFT)
        finish(a2, "But the ranking does improve, up to a point",
               "Validation ROC AUC over the same grid. This is what the settings "
               "were chosen on, since accuracy cannot distinguish them.",
               "Boosting rounds (log scale)", "Validation ROC AUC")
        fig.tight_layout(h_pad=2.4)
        save(fig, "fig44_adaboost_validation.png")

        # 45. the stump weights
        al = M["stump_alphas"]
        fig, ax = plt.subplots(figsize=(8.4, 4.2))
        ax.plot(range(1, len(al) + 1), al, color=C[6], linewidth=2.0)
        ax.fill_between(range(1, len(al) + 1), al, 0, color=C[6], alpha=0.14)
        finish(ax, "Each stump earns a say proportional to how right it was",
               f"The weight alpha given to each of the first {len(al)} stumps. Early "
               f"stumps find the easy structure and are trusted most; later ones work "
               f"on the leftovers and count for less.",
               "Stump number", "Weight (alpha)")
        save(fig, "fig45_adaboost_alphas.png")

        # 46. confusion matrix
        cm = np.array(M["confusion"])
        cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        labels = ["Did not chart", "Charted"]
        fig, ax = plt.subplots(figsize=(6.2, 4.9))
        im = ax.imshow(cmn, cmap=theme.SEQ_CMAP, vmin=0, vmax=100)
        ax.set_xticks(range(2), labels); ax.set_yticks(range(2), labels)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cmn[i, j]:.0f}%\n{cm[i, j]:,}", ha="center",
                        va="center", fontsize=9.4,
                        color="#ffffff" if cmn[i, j] > 55 else theme.INK)
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
        cb.set_label("Share of the true class (%)", color=theme.INK_SOFT, fontsize=9.2)
        cb.outline.set_visible(False)
        ax.set_title("The strongest method so far lands in the same place\n", loc="left")
        ax.text(0, 1.02, f"Accuracy {acc*100:.1f}% against an {baseline*100:.1f}% "
                f"baseline, on the same {len(yte):,} held-out songs used throughout.",
                transform=ax.transAxes, fontsize=9.8, color=theme.INK_SOFT, va="bottom")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        save(fig, "fig46_adaboost_confusion.png")

        # 47. against the Module 3 tree
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        names = [f"Decision tree\n(Module 3)", "AdaBoost\n(this module)"]
        accs = [dt_best["test_accuracy"] * 100, acc * 100]
        ax.bar(range(2), accs, color=[C[3], C[0]], width=0.5,
               edgecolor=theme.BG, linewidth=1.8)
        for i, v in enumerate(accs):
            ax.text(i, v + 1.1, f"{v:.2f}%", ha="center", fontsize=10.5,
                    fontweight="bold", color=theme.INK_SOFT)
        ax.axhline(baseline * 100, color=C[1], linestyle="--", linewidth=1.8,
                   label=f"Baseline ({baseline*100:.2f}%)")
        ax.set_xticks(range(2), names)
        ax.set_ylim(80, 92)
        ax.legend(loc="upper right", fontsize=9)
        finish(ax, f"AdaBoost beats the single tree by "
                   f"{M['dt_comparison']['difference_pp']:+.2f} points, and the "
                   f"baseline by almost nothing",
               "Accuracy on the identical held-out songs. Note the axis starts at 80% "
               "to make a difference this small visible at all.",
               None, "Test accuracy (%)")
        save(fig, "fig47_adaboost_vs_tree.png")

        # 48. which features the stumps used
        top = imp[imp > 0].head(14).sort_values()
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        cols = [C[2] if ("genre" in f or "key" in f or "mode" in f) else C[6]
                for f in top.index]
        ax.barh(range(len(top)), top.values * 100, height=0.66, color=cols,
                edgecolor=theme.BG, linewidth=1.4)
        ax.set_yticks(range(len(top)),
                      [f.replace("playlist_genre_", "genre: ")
                        .replace("key_name_", "key: ")
                        .replace("mode_name_", "mode: ")
                        .replace("_", " ").capitalize() for f in top.index])
        for i, v in enumerate(top.values * 100):
            ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=8.6,
                    color=theme.INK_SOFT)
        ax.set_xlim(0, top.max() * 118)
        finish(ax, "The encoded categories earned nearly a third of the model",
               "Share of the boosted model's split quality, with one-hot encoded "
               "columns shown in green and the original measurements in purple",
               "Contribution (%)", None)
        save(fig, "fig48_adaboost_importance.png")

    with open(os.path.join(CLEAN, "adaboost_metrics.json"), "w") as fh:
        json.dump(M, fh, indent=2, default=float)

    print("\n" + "=" * 72)
    print("AdaBoost complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
