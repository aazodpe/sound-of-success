"""
CSCI 5612 - Project Part 5
Script 17: Gradient boosting.

AdaBoost in Module 4 corrected its mistakes by re-weighting the awkward rows.
Gradient boosting corrects them differently: each new tree is fitted to the
RESIDUAL left over by the running total so far, so the sequence is a series of
small additive corrections rather than a weighted vote.

Three settings are tuned, all of them on the VALIDATION set:

  learning_rate   how much of each tree's correction is actually added
  max_depth       how much interaction a single correction may express
  n_estimators    how many corrections to make, read off the staged
                  validation curve rather than guessed

The test set is read exactly once, at the end, with the chosen configuration.

Reads  ../data/clean/m4_adaboost_<train|val|test>.csv   (same rows as Module 4)
       ../data/clean/adaboost_metrics.json              (for the comparison)
Writes ../data/clean/gb_metrics.json
       ../docs/images/fig49..fig52
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, log_loss, roc_auc_score)

import theme

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "data", "clean")
IMG = os.path.join(HERE, "..", "docs", "images")
SEED = 5612

NUMERIC = ["danceability", "energy", "loudness", "speechiness", "acousticness",
           "instrumentalness", "liveness", "valence", "tempo", "duration_min"]

N_MAX = 300                       # trees fitted; the useful number is read off
RATES = [0.03, 0.1, 0.3]          # setting 1
DEPTHS = [1, 2, 3, 5]             # setting 2

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


def staged_curves(clf, X, y):
    """Validation AUC and log loss after every tree is added."""
    aucs, losses = [], []
    for p in clf.staged_predict_proba(X):
        aucs.append(float(roc_auc_score(y, p[:, 1])))
        losses.append(float(log_loss(y, p[:, 1], labels=[0, 1])))
    return np.array(aucs), np.array(losses)


def main():
    print("=" * 72)
    print("CSCI 5612 Project Part 5 - gradient boosting")
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

    # --- tune three settings on validation --------------------------------
    print(f"\n  tuning on validation: {len(RATES)} learning rates x "
          f"{len(DEPTHS)} depths, each fitted to {N_MAX} trees")
    grid = []
    curves = {}
    for lr in RATES:
        for d in DEPTHS:
            clf = GradientBoostingClassifier(
                n_estimators=N_MAX, learning_rate=lr, max_depth=d,
                random_state=SEED).fit(Xtr, ytr)
            va_auc, va_loss = staged_curves(clf, Xva, yva)
            k = int(va_auc.argmax()) + 1
            grid.append({
                "learning_rate": lr, "max_depth": d,
                "best_n_estimators": k,
                "val_auc": float(va_auc.max()),
                "val_auc_at_max_trees": float(va_auc[-1]),
                "val_loss_at_best": float(va_loss[k - 1]),
                "val_loss_at_max_trees": float(va_loss[-1]),
            })
            curves[(lr, d)] = {
                "val_auc": va_auc.tolist(),
                "val_loss": va_loss.tolist(),
                # train_score_ holds the binomial DEVIANCE, which is twice
                # the log loss, so it is halved to share an axis with the
                # validation curve computed by log_loss() above.
                "train_loss": (clf.train_score_ / 2).tolist(),
            }
            print(f"    rate {lr:<5} depth {d}:  best val AUC {va_auc.max():.4f} "
                  f"at {k:>3} trees   (at {N_MAX} trees: {va_auc[-1]:.4f})")

    best = max(grid, key=lambda g: g["val_auc"])
    # The configuration that loses the most by being run too long. Its curve is
    # the clearest illustration of what "too many trees" actually costs.
    worst = max(grid, key=lambda g: g["val_auc"] - g["val_auc_at_max_trees"])
    print(f"\n  chosen on validation AUC: rate {best['learning_rate']}, "
          f"depth {best['max_depth']}, {best['best_n_estimators']} trees "
          f"(val AUC {best['val_auc']:.4f})")

    # How much validation ranking is lost by running on to 300 trees instead
    # of stopping where the curve peaked. This is the overfitting cost, measured.
    overrun = {
        "val_auc_at_best": best["val_auc"],
        "val_auc_at_max_trees": best["val_auc_at_max_trees"],
        "auc_lost": best["val_auc"] - best["val_auc_at_max_trees"],
        "val_loss_at_best": best["val_loss_at_best"],
        "val_loss_at_max_trees": best["val_loss_at_max_trees"],
        "loss_gained": best["val_loss_at_max_trees"] - best["val_loss_at_best"],
    }
    print(f"  worst over-run: rate {worst['learning_rate']}, depth "
          f"{worst['max_depth']} peaks at {worst['best_n_estimators']} trees "
          f"({worst['val_auc']:.4f}) and falls to "
          f"{worst['val_auc_at_max_trees']:.4f} by {N_MAX}")
    print(f"  running on to {N_MAX} trees costs "
          f"{overrun['auc_lost']:.4f} AUC and adds "
          f"{overrun['loss_gained']:.4f} to validation log loss")

    # --- the one-hot question, same experiment as Module 4 ----------------
    print("\n  does the one-hot encoded information help here too?")
    compare = {}
    for name, enc in [("numeric only", False), ("with one-hot", True)]:
        A, a = load("train", enc); B, b = load("val", enc)
        clf = GradientBoostingClassifier(
            n_estimators=best["best_n_estimators"],
            learning_rate=best["learning_rate"], max_depth=best["max_depth"],
            random_state=SEED).fit(A, a)
        auc = float(roc_auc_score(b, clf.predict_proba(B)[:, 1]))
        compare[name] = {"n_features": int(A.shape[1]), "val_auc": auc}
        print(f"    {name:<14} {A.shape[1]:>2} features -> validation AUC {auc:.4f}")

    # --- final model, test set read once ----------------------------------
    final = GradientBoostingClassifier(
        n_estimators=best["best_n_estimators"], learning_rate=best["learning_rate"],
        max_depth=best["max_depth"], random_state=SEED).fit(Xtr, ytr)
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

    # --- against Module 4's AdaBoost, same songs --------------------------
    ada = json.load(open(os.path.join(CLEAN, "adaboost_metrics.json")))
    print(f"\n  Module 4 AdaBoost on the same songs: "
          f"accuracy {ada['test_accuracy']:.4f}, AUC {ada['test_auc']:.4f}")
    print(f"  gradient boosting minus AdaBoost: "
          f"{(acc - ada['test_accuracy'])*100:+.2f} pp accuracy, "
          f"{auc - ada['test_auc']:+.4f} AUC")

    imp = pd.Series(final.feature_importances_, index=feats).sort_values(ascending=False)

    M.update({
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
        "n_features": len(feats), "n_numeric": len(NUMERIC),
        "n_onehot": len(feats) - len(NUMERIC),
        "n_charted_test": int(yte.sum()),
        "baseline": baseline,
        "n_max_trees_fitted": N_MAX, "rates": RATES, "depths": DEPTHS,
        "grid": grid, "best": best, "overrun": overrun,
        "onehot_comparison": compare,
        "test_accuracy": acc, "test_auc": auc,
        "test_recall_charted": float(rep["1"]["recall"]),
        "test_precision_charted": float(rep["1"]["precision"]),
        "test_flagged": int((pred == 1).sum()),
        "confusion": confusion_matrix(yte, pred).tolist(),
        "adaboost_comparison": {
            "test_accuracy": ada["test_accuracy"], "test_auc": ada["test_auc"],
            "accuracy_difference_pp": (acc - ada["test_accuracy"]) * 100,
            "auc_difference": auc - ada["test_auc"],
        },
        "importances": {f: float(v) for f, v in imp.items()},
        "onehot_share": float(imp[[f for f in feats if f not in NUMERIC]].sum()),
        "best_curve": curves[(best["learning_rate"], best["max_depth"])],
        "overrun_worst": worst,
        "overrun_worst_curve": curves[(worst["learning_rate"], worst["max_depth"])],
    })

    for mode in theme.MODES:
        theme.set_theme(mode)
        C = theme.C
        print(f"\n  --- figures ({mode}) ---")

        # 49. the tuning grid
        fig, ax = plt.subplots(figsize=(8.6, 4.6))
        for i, lr in enumerate(RATES):
            sub = [g for g in grid if g["learning_rate"] == lr]
            ax.plot([g["max_depth"] for g in sub], [g["val_auc"] for g in sub],
                    color=C[i], linewidth=2.2, marker="o", markersize=6,
                    markeredgecolor=theme.BG, markeredgewidth=1.3,
                    label=f"learning rate {lr}")
        ax.scatter([best["max_depth"]], [best["val_auc"]], s=190, facecolor="none",
                   edgecolor=C[1], linewidth=2.3, zorder=6)
        ax.text(0.015, 0.97, f"circled: rate {best['learning_rate']}, depth "
                f"{best['max_depth']}, {best['best_n_estimators']} trees",
                transform=ax.transAxes, ha="left", va="top", color=C[1],
                fontweight="bold", fontsize=9.3)
        ax.set_xticks(DEPTHS, [str(d) for d in DEPTHS])
        ax.legend(loc="lower right", fontsize=9)
        finish(ax, "Depth buys more than the learning rate, and stops paying "
                   "around three",
               f"Best validation ROC AUC for each of the twelve configurations, "
               f"with the number of trees\nitself read off the staged curve below. "
               f"The whole grid spans {max(g['val_auc'] for g in grid) - min(g['val_auc'] for g in grid):.3f} AUC.",
               "Tree depth", "Best validation ROC AUC")
        save(fig, "fig49_gb_tuning.png")

        # 50. validation error as trees are added  (required by the assignment)
        cur = M["best_curve"]
        n = np.arange(1, len(cur["val_loss"]) + 1)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 7.0), sharex=True)
        a1.plot(n, cur["train_loss"], color=C[0], linewidth=2.1, label="Training")
        a1.plot(n, cur["val_loss"], color=C[1], linewidth=2.1, label="Validation")
        kk = best["best_n_estimators"]
        a1.axvline(kk, color=theme.INK_SOFT, linestyle="--", linewidth=1.5)
        a1.annotate(f"{kk} trees", xy=(kk, 0.45), xycoords=("data", "axes fraction"),
                    xytext=(-7, 0), textcoords="offset points", ha="right",
                    fontsize=9, color=theme.INK_SOFT)
        a1.legend(loc="upper right", fontsize=9)
        vmin = int(np.argmin(cur["val_loss"])) + 1
        a1.scatter([vmin], [min(cur["val_loss"])], s=150, facecolor="none",
                   edgecolor=C[1], linewidth=2.0, zorder=6)
        finish(a1, "Training error keeps falling. Validation error bottoms out "
                   "and turns back up.",
               f"Log loss after every tree is added. The two lines start together;\n"
               f"validation bottoms out at {vmin} trees, and the gap after that "
               f"is overfitting.",
               None, "Log loss")
        bad = M["overrun_worst"]; badc = M["overrun_worst_curve"]
        a2.plot(n, badc["val_auc"], color=C[1], linewidth=1.9,
                label=f"rate {bad['learning_rate']}, depth {bad['max_depth']} "
                      f"(aggressive)")
        a2.plot(n, cur["val_auc"], color=C[2], linewidth=2.3,
                label=f"rate {best['learning_rate']}, depth {best['max_depth']} "
                      f"(chosen)")
        a2.legend(loc="lower left", fontsize=8.6)
        a2.axvline(kk, color=theme.INK_SOFT, linestyle="--", linewidth=1.5)
        a2.scatter([kk], [best["val_auc"]], s=170, facecolor="none",
                   edgecolor=C[1], linewidth=2.2, zorder=6)
        a2.annotate(f"peak {best['val_auc']:.4f} at {kk} trees; "
                    f"{best['val_auc_at_max_trees']:.4f} by {N_MAX}",
                    xy=(kk, best["val_auc"]), xytext=(14, -34),
                    textcoords="offset points", ha="left", va="top", color=C[1],
                    fontweight="bold", fontsize=9.3,
                    arrowprops=dict(arrowstyle="-", color=C[1], linewidth=1.1,
                                    shrinkA=2, shrinkB=9))
        finish(a2, "Every setting peaks and then decays. Aggressive ones decay "
                   "hard.",
               f"Validation ROC AUC over the same sequence. The chosen setting "
               f"gives back {M['overrun']['auc_lost']:.4f} by {N_MAX} trees;\n"
               f"the most aggressive setting in the grid gives back "
               f"{bad['val_auc'] - bad['val_auc_at_max_trees']:.4f}. The tree count "
               f"was read off this curve.",
               "Trees added", "Validation ROC AUC")
        fig.tight_layout(h_pad=5.2)
        save(fig, "fig50_gb_validation_curve.png")

        # 51. confusion matrix
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
        ax.set_title("Better at ranking, no better at committing\n", loc="left")
        ax.text(0, 1.02, f"Accuracy {acc*100:.1f}% against an {baseline*100:.1f}% "
                f"baseline, on the same {len(yte):,} held-out songs used throughout.",
                transform=ax.transAxes, fontsize=9.8, color=theme.INK_SOFT, va="bottom")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        save(fig, "fig51_gb_confusion.png")

        # 52. against Module 4's AdaBoost
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.4))
        names = ["AdaBoost\n(Module 4)", "Gradient boosting\n(this module)"]
        accs = [ada["test_accuracy"] * 100, acc * 100]
        a1.bar(range(2), accs, color=[C[3], C[0]], width=0.5,
               edgecolor=theme.BG, linewidth=1.8)
        for i, v in enumerate(accs):
            a1.text(i, v + 0.25, f"{v:.2f}%", ha="center", fontsize=10.2,
                    fontweight="bold", color=theme.INK_SOFT)
        a1.axhline(baseline * 100, color=C[1], linestyle="--", linewidth=1.8,
                   label=f"Baseline ({baseline*100:.2f}%)")
        a1.set_xticks(range(2), names); a1.set_ylim(84, 89)
        a1.legend(loc="lower right", fontsize=8.6)
        a1.text(1, baseline * 100 - 0.42, "gradient boosting sits just below it",
                ha="center", fontsize=8.4, color=C[1])
        finish(a1, "Accuracy: both are pinned to the baseline",
               None, None, "Test accuracy (%)")
        aucs = [ada["test_auc"], auc]
        a2.bar(range(2), aucs, color=[C[3], C[2]], width=0.5,
               edgecolor=theme.BG, linewidth=1.8)
        for i, v in enumerate(aucs):
            a2.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=10.2,
                    fontweight="bold", color=theme.INK_SOFT)
        a2.axhline(0.5, color=theme.INK_SOFT, linestyle=":", linewidth=1.5)
        a2.set_xticks(range(2), names); a2.set_ylim(0.5, max(aucs) * 1.08)
        finish(a2, "Ranking: where the difference actually shows",
               None, None, "Test ROC AUC")
        fig.suptitle("Two boosting methods on the identical held-out songs",
                     x=0.012, ha="left", fontsize=13.5, fontweight="bold", y=0.985)
        fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=3.0)
        save(fig, "fig52_gb_vs_adaboost.png")

    with open(os.path.join(CLEAN, "gb_metrics.json"), "w") as fh:
        json.dump(M, fh, indent=2, default=float)

    print("\n" + "=" * 72)
    print("Gradient boosting complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
