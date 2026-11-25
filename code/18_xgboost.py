"""
CSCI 5612 - Project Part 5
Script 18: XGBoost, and the comparison across every supervised model on the site.

XGBoost is gradient boosting with two additions that matter here. The objective
it minimises is not the training loss alone but the training loss PLUS a price
for complexity: gamma for every leaf a tree grows, and lambda for the size of
each leaf's weight. And the number of rounds is not a setting to be guessed at
all: training watches the validation set and stops when it has not improved for
a fixed number of rounds.

Three settings are tuned on validation (max_depth, learning_rate, reg_lambda)
and the fourth, n_estimators, is chosen by early stopping. The test set is read
once.

The second half of this script assembles the comparison table the assignment
asks for. Accuracy and recall are taken from the stored metrics of each
module, so the table cannot disagree with the tab it came from; the two ROC
AUCs that were never stored (Naive Bayes and the decision tree) are computed
here by refitting those exact models, with an assertion that the refit
reproduces the published accuracy.

Reads  ../data/clean/m4_adaboost_<train|val|test>.csv
       ../data/clean/{nb,dt,regression,adaboost,gb}_metrics.json
Writes ../data/clean/xgb_metrics.json
       ../data/clean/model_comparison.csv
       ../docs/images/fig53..fig57
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

import theme

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "data", "clean")
IMG = os.path.join(HERE, "..", "docs", "images")
SEED = 5612

NUMERIC = ["danceability", "energy", "loudness", "speechiness", "acousticness",
           "instrumentalness", "liveness", "valence", "tempo", "duration_min"]

N_MAX = 800          # an upper bound; early stopping decides the real number
PATIENCE = 40
DEPTHS = [2, 3, 5, 7]            # setting 1
RATES = [0.03, 0.1, 0.3]         # setting 2
LAMBDAS = [1.0, 10.0]            # setting 3 - the regularisation XGBoost adds

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


def model(depth, rate, lam, n=N_MAX, patience=PATIENCE):
    return xgb.XGBClassifier(
        n_estimators=n, max_depth=depth, learning_rate=rate, reg_lambda=lam,
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        early_stopping_rounds=patience, random_state=SEED, n_jobs=4)


# --------------------------------------------------------------------------
# The comparison table: every supervised model on the site, same test set.
# --------------------------------------------------------------------------

def missing_aucs():
    """
    Naive Bayes and the decision tree were scored on accuracy in Module 3 and
    their ranking quality was never recorded. Both are refitted here exactly as
    they were fitted then, and each refit is checked against the published
    accuracy before its AUC is used.
    """
    tr = pd.read_csv(os.path.join(CLEAN, "supervised_charted_train.csv"))
    te = pd.read_csv(os.path.join(CLEAN, "supervised_charted_test.csv"))
    feats = [c for c in tr.columns if c != "label"]
    Xtr, ytr, Xte, yte = tr[feats], tr["label"], te[feats], te["label"]

    nb_pub = json.load(open(os.path.join(CLEAN, "nb_metrics.json")))["charted"]
    dt_pub = json.load(open(os.path.join(CLEAN, "dt_metrics.json")))["charted"]

    # MultinomialNB on min-max scaled features, scaler fitted on training only.
    mm = MinMaxScaler().fit(Xtr)
    nb = MultinomialNB().fit(mm.transform(Xtr), ytr)
    Xte_mm = np.clip(mm.transform(Xte), 0, 1)
    nb_acc = float(accuracy_score(yte, nb.predict(Xte_mm)))
    assert abs(nb_acc - nb_pub["MultinomialNB"]["accuracy"]) < 1e-12, \
        f"NB refit {nb_acc} does not reproduce published {nb_pub['MultinomialNB']['accuracy']}"
    nb_auc = float(roc_auc_score(yte, nb.predict_proba(Xte_mm)[:, 1]))

    # Tree B, the best of the three on accuracy: entropy, depth 6.
    dt = DecisionTreeClassifier(max_depth=6, criterion="entropy",
                                min_samples_leaf=1, random_state=SEED).fit(Xtr, ytr)
    dt_acc = float(accuracy_score(yte, dt.predict(Xte)))
    assert abs(dt_acc - dt_pub["medium"]["test_accuracy"]) < 1e-12, \
        f"tree refit {dt_acc} does not reproduce published {dt_pub['medium']['test_accuracy']}"
    dt_auc = float(roc_auc_score(yte, dt.predict_proba(Xte)[:, 1]))

    print(f"    Naive Bayes refit reproduces {nb_acc:.4f}; its AUC is {nb_auc:.4f}")
    print(f"    decision tree refit reproduces {dt_acc:.4f}; its AUC is {dt_auc:.4f}")
    return nb_auc, dt_auc


def comparison_table(xgb_row):
    nb = json.load(open(os.path.join(CLEAN, "nb_metrics.json")))["charted"]
    dt = json.load(open(os.path.join(CLEAN, "dt_metrics.json")))["charted"]
    lr = json.load(open(os.path.join(CLEAN, "regression_metrics.json")))["logistic"]
    ab = json.load(open(os.path.join(CLEAN, "adaboost_metrics.json")))
    gb = json.load(open(os.path.join(CLEAN, "gb_metrics.json")))
    nb_auc, dt_auc = missing_aucs()

    rows = [
        {"model": "Naive Bayes", "module": "Module 3", "variant": "Multinomial",
         "accuracy": nb["MultinomialNB"]["accuracy"], "auc": nb_auc,
         "recall": nb["MultinomialNB"]["report"]["1"]["recall"],
         "precision": nb["MultinomialNB"]["report"]["1"]["precision"],
         "flagged": int(np.array(nb["MultinomialNB"]["confusion"])[:, 1].sum())},
        {"model": "Decision tree", "module": "Module 3", "variant": "entropy, depth 6",
         "accuracy": dt["medium"]["test_accuracy"], "auc": dt_auc,
         "recall": dt["medium"]["report"]["1"]["recall"],
         "precision": dt["medium"]["report"]["1"]["precision"],
         "flagged": int(np.array(dt["medium"]["confusion"])[:, 1].sum())},
        {"model": "Logistic regression", "module": "Module 4", "variant": "default threshold",
         "accuracy": lr["default"]["accuracy"], "auc": lr["default"]["roc_auc"],
         "recall": lr["default"]["recall_charted"],
         "precision": lr["default"]["precision_charted"],
         "flagged": int(np.array(lr["default"]["confusion"])[:, 1].sum())},
        {"model": "AdaBoost", "module": "Module 4",
         "variant": f"{ab['best']['n_estimators']} stumps, rate {ab['best']['learning_rate']}",
         "accuracy": ab["test_accuracy"], "auc": ab["test_auc"],
         "recall": ab["test_recall_charted"], "precision": ab["test_precision_charted"],
         "flagged": int(np.array(ab["confusion"])[:, 1].sum())},
        {"model": "Gradient boosting", "module": "Module 5",
         "variant": f"{gb['best']['best_n_estimators']} trees, depth {gb['best']['max_depth']}",
         "accuracy": gb["test_accuracy"], "auc": gb["test_auc"],
         "recall": gb["test_recall_charted"], "precision": gb["test_precision_charted"],
         "flagged": int(np.array(gb["confusion"])[:, 1].sum())},
        xgb_row,
    ]
    df = pd.DataFrame(rows)
    df["accuracy_pp_over_baseline"] = (df["accuracy"] - M["baseline"]) * 100
    return df


# --------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("CSCI 5612 Project Part 5 - XGBoost")
    print("=" * 72)

    Xtr, ytr = load("train")
    Xva, yva = load("val")
    Xte, yte = load("test")
    feats = list(Xtr.columns)
    baseline = float(1 - yte.mean())
    M["baseline"] = baseline
    print(f"  {len(Xtr):,} train / {len(Xva):,} validation / {len(Xte):,} test")
    print(f"  {len(feats)} features ({len(NUMERIC)} numeric + "
          f"{len(feats)-len(NUMERIC)} one-hot)")
    print(f"  baseline accuracy (always 'did not chart'): {baseline:.4f}")
    print(f"  xgboost {xgb.__version__}")

    # --- tune three settings; early stopping chooses the fourth -----------
    print(f"\n  tuning on validation: {len(DEPTHS)} depths x {len(RATES)} rates "
          f"x {len(LAMBDAS)} lambdas, each capped at {N_MAX} rounds with "
          f"{PATIENCE} rounds of patience")
    grid = []
    for d in DEPTHS:
        for r in RATES:
            for lam in LAMBDAS:
                m = model(d, r, lam)
                m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
                grid.append({
                    "max_depth": d, "learning_rate": r, "reg_lambda": lam,
                    "stopped_at": int(m.best_iteration) + 1,
                    "val_auc": float(m.best_score),
                })
            best_d_r = max((g for g in grid if g["max_depth"] == d
                            and g["learning_rate"] == r), key=lambda g: g["val_auc"])
            print(f"    depth {d}, rate {r:<5}: best val AUC "
                  f"{best_d_r['val_auc']:.4f} (lambda {best_d_r['reg_lambda']}, "
                  f"stopped at {best_d_r['stopped_at']} rounds)")

    best = max(grid, key=lambda g: g["val_auc"])
    print(f"\n  chosen: depth {best['max_depth']}, rate {best['learning_rate']}, "
          f"lambda {best['reg_lambda']}, stopped at {best['stopped_at']} rounds "
          f"(val AUC {best['val_auc']:.4f})")

    # What the regularisation term is worth, holding everything else fixed.
    lam_effect = {}
    for lam in LAMBDAS:
        g = [x for x in grid if x["max_depth"] == best["max_depth"]
             and x["learning_rate"] == best["learning_rate"]
             and x["reg_lambda"] == lam][0]
        lam_effect[str(lam)] = {"val_auc": g["val_auc"], "stopped_at": g["stopped_at"]}
    print(f"  effect of lambda at the chosen depth and rate: " +
          ", ".join(f"lambda {k} -> {v['val_auc']:.4f}" for k, v in lam_effect.items()))

    # --- the chosen model, with its full early stopping history -----------
    final = model(best["max_depth"], best["learning_rate"], best["reg_lambda"])
    final.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
    hist = final.evals_result()
    train_curve = hist["validation_0"]["auc"]
    val_curve = hist["validation_1"]["auc"]
    print(f"  early stopping: fitted {len(val_curve)} rounds, kept round "
          f"{int(final.best_iteration)+1} of a possible {N_MAX}")

    # --- the one-hot question, for the third time -------------------------
    print("\n  does the one-hot encoded information help here too?")
    compare = {}
    for name, enc in [("numeric only", False), ("with one-hot", True)]:
        A, a = load("train", enc); B, b = load("val", enc)
        m = model(best["max_depth"], best["learning_rate"], best["reg_lambda"])
        m.fit(A, a, eval_set=[(B, b)], verbose=False)
        compare[name] = {"n_features": int(A.shape[1]), "val_auc": float(m.best_score)}
        print(f"    {name:<14} {A.shape[1]:>2} features -> validation AUC "
              f"{m.best_score:.4f}")

    # --- test set, read once ----------------------------------------------
    prob = final.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    acc = float(accuracy_score(yte, pred))
    auc = float(roc_auc_score(yte, prob))
    rep = classification_report(yte, pred, output_dict=True, zero_division=0)
    print(f"\n  FINAL on the held-out test set:")
    print(f"    accuracy {acc:.4f}  (baseline {baseline:.4f})")
    print(f"    ROC AUC  {auc:.4f}")
    print(f"    recall on charted {rep['1']['recall']:.3f}, "
          f"precision {rep['1']['precision']:.3f}")

    imp = pd.Series(final.feature_importances_, index=feats).sort_values(ascending=False)

    M.update({
        "xgboost_version": xgb.__version__,
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
        "n_features": len(feats), "n_numeric": len(NUMERIC),
        "n_onehot": len(feats) - len(NUMERIC), "n_charted_test": int(yte.sum()),
        "n_max_rounds": N_MAX, "patience": PATIENCE,
        "depths": DEPTHS, "rates": RATES, "lambdas": LAMBDAS,
        "grid": grid, "best": best, "lambda_effect": lam_effect,
        "rounds_fitted": len(val_curve),
        "best_round": int(final.best_iteration) + 1,
        "train_curve": [float(v) for v in train_curve],
        "val_curve": [float(v) for v in val_curve],
        "onehot_comparison": compare,
        "test_accuracy": acc, "test_auc": auc,
        "test_recall_charted": float(rep["1"]["recall"]),
        "test_precision_charted": float(rep["1"]["precision"]),
        "test_flagged": int(pred.sum()),
        "confusion": confusion_matrix(yte, pred).tolist(),
        "importances": {f: float(v) for f, v in imp.items()},
        "onehot_share": float(imp[[f for f in feats if f not in NUMERIC]].sum()),
    })

    # --- every supervised model on the site, one table --------------------
    print("\n  assembling the comparison across all six supervised models")
    xgb_row = {"model": "XGBoost", "module": "Module 5",
               "variant": f"depth {best['max_depth']}, rate {best['learning_rate']}, "
                          f"lambda {best['reg_lambda']}",
               "accuracy": acc, "auc": auc,
               "recall": float(rep["1"]["recall"]),
               "precision": float(rep["1"]["precision"]),
               "flagged": int(pred.sum())}
    table = comparison_table(xgb_row)
    table.to_csv(os.path.join(CLEAN, "model_comparison.csv"), index=False)
    print()
    print(table[["model", "accuracy", "auc", "recall", "flagged"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    win_acc = table.loc[table["accuracy"].idxmax()]
    win_auc = table.loc[table["auc"].idxmax()]
    runner = table.sort_values("auc", ascending=False).iloc[1]
    M["comparison"] = table.to_dict(orient="records")
    M["winner_accuracy"] = {"model": win_acc["model"], "accuracy": float(win_acc["accuracy"])}
    M["winner_auc"] = {
        "model": win_auc["model"], "auc": float(win_auc["auc"]),
        "runner_up": runner["model"], "runner_up_auc": float(runner["auc"]),
        "margin": float(win_auc["auc"] - runner["auc"]),
        "margin_over_worst": float(win_auc["auc"] - table["auc"].min()),
    }
    print(f"\n  best accuracy: {win_acc['model']} at {win_acc['accuracy']*100:.2f}% "
          f"({(win_acc['accuracy']-baseline)*100:+.2f} pp on the baseline)")
    print(f"  best ranking:  {win_auc['model']} at AUC {win_auc['auc']:.4f}, "
          f"{win_auc['auc']-runner['auc']:+.4f} over {runner['model']}")

    # ---------------------------- figures ---------------------------------
    for mode in theme.MODES:
        theme.set_theme(mode)
        C = theme.C
        print(f"\n  --- figures ({mode}) ---")

        # 53. the tuning grid
        fig, ax = plt.subplots(figsize=(8.8, 4.6))
        marks = {1.0: "o", 10.0: "s"}
        for i, r in enumerate(RATES):
            for lam in LAMBDAS:
                sub = sorted([g for g in grid if g["learning_rate"] == r
                              and g["reg_lambda"] == lam], key=lambda g: g["max_depth"])
                ax.plot([g["max_depth"] for g in sub], [g["val_auc"] for g in sub],
                        color=C[i], linewidth=2.0 if lam == 10.0 else 1.4,
                        linestyle="-" if lam == 10.0 else "--",
                        marker=marks[lam], markersize=5.5,
                        markeredgecolor=theme.BG, markeredgewidth=1.1,
                        label=f"rate {r}, lambda {lam:g}")
        ax.scatter([best["max_depth"]], [best["val_auc"]], s=190, facecolor="none",
                   edgecolor=C[1], linewidth=2.3, zorder=6)
        ax.text(0.015, 0.97, f"circled: depth {best['max_depth']}, rate "
                f"{best['learning_rate']}, lambda {best['reg_lambda']:g}",
                transform=ax.transAxes, ha="left", va="top", color=C[1],
                fontweight="bold", fontsize=9.3)
        ax.set_xticks(DEPTHS, [str(d) for d in DEPTHS])
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - (hi - lo) * 0.55, hi)        # headroom for the legend
        ax.legend(ncols=3, fontsize=8.0, loc="lower center")
        span = max(g["val_auc"] for g in grid) - min(g["val_auc"] for g in grid)
        finish(ax, "Twenty-four configurations, and a ceiling none of them pass",
               f"Validation ROC AUC for every depth, rate and lambda; solid lines "
               f"are the heavier regularisation.\nRead the axis carefully: the whole "
               f"grid spans {span:.3f} AUC, so these are large-looking differences "
               f"between near-identical models.",
               "Tree depth", "Validation ROC AUC")
        save(fig, "fig53_xgb_tuning.png")

        # 54. early stopping in action
        n = np.arange(1, len(val_curve) + 1)
        br = M["best_round"]
        fig, ax = plt.subplots(figsize=(9.0, 4.6))
        ax.plot(n, train_curve, color=C[0], linewidth=2.1, label="Training AUC")
        ax.plot(n, val_curve, color=C[1], linewidth=2.1, label="Validation AUC")
        ax.axvline(br, color=C[2], linewidth=1.8, linestyle="--")
        ax.scatter([br], [val_curve[br - 1]], s=170, facecolor="none",
                   edgecolor=C[2], linewidth=2.2, zorder=6)
        ax.annotate(f"best round {br}\nthis is the model kept",
                    xy=(br, val_curve[br - 1]), xytext=(-18, 48),
                    textcoords="offset points", ha="right", va="bottom", color=C[2],
                    fontweight="bold", fontsize=9.2,
                    arrowprops=dict(arrowstyle="-", color=C[2], linewidth=1.1,
                                    shrinkA=2, shrinkB=9))
        ax.axvspan(br, len(val_curve), color=C[1], alpha=0.09)
        ax.annotate(f"{PATIENCE} rounds of patience,\nthen training halts",
                    xy=((br + len(val_curve)) / 2, min(val_curve) + 0.03),
                    ha="center", fontsize=8.9, color=C[1], fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        finish(ax, f"Early stopping kept {br} rounds out of a possible {N_MAX}",
               f"Training and validation AUC at every round of the chosen "
               f"configuration.\nThe number of trees was decided by the data, not "
               f"by a setting.",
               "Boosting rounds", "ROC AUC")
        save(fig, "fig54_xgb_early_stopping.png")

        # 55. confusion matrix
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
        ax.set_title("The best method in the project, in the same corner\n", loc="left")
        ax.text(0, 1.02, f"Accuracy {acc*100:.1f}% against an {baseline*100:.1f}% "
                f"baseline, on the same {len(yte):,} held-out songs used throughout.",
                transform=ax.transAxes, fontsize=9.8, color=theme.INK_SOFT, va="bottom")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        save(fig, "fig55_xgb_confusion.png")

        # 56. feature importance
        top = imp[imp > 0].head(14).sort_values()
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
        cols = [C[2] if f not in NUMERIC else C[6] for f in top.index]
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
        finish(ax, f"The encoded categories take "
                   f"{M['onehot_share']*100:.0f}% of the finished model",
               "Importance by gain, with one-hot encoded columns in green and the\n"
               "original measurements in purple.",
               "Contribution (%)", None)
        save(fig, "fig56_xgb_importance.png")

        # 57. every supervised model, side by side
        t = table.sort_values("auc")
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)
        ypos = np.arange(len(t))
        a1.barh(ypos, t["accuracy"] * 100, height=0.6, color=C[6],
                edgecolor=theme.BG, linewidth=1.4)
        a1.axvline(baseline * 100, color=C[1], linestyle="--", linewidth=1.8)
        a1.text(baseline * 100 - 0.05, len(t) - 0.38,
                f"baseline {baseline*100:.2f}%", ha="right", va="center",
                fontsize=8.6, color=C[1], fontweight="bold")
        for i, v in enumerate(t["accuracy"] * 100):
            a1.text(v - 0.12, i, f"{v:.2f}%", va="center", ha="right", fontsize=8.6,
                    color="#ffffff", fontweight="bold")
        a1.set_xlim(84, 88)
        a1.set_yticks(ypos, list(t["model"]))
        finish(a1, "Accuracy: everything is the baseline", None,
               "Test accuracy (%)", None)
        a2.barh(ypos, t["auc"], height=0.6, color=C[2],
                edgecolor=theme.BG, linewidth=1.4)
        a2.axvline(0.5, color=theme.INK_SOFT, linestyle=":", linewidth=1.6)
        for i, v in enumerate(t["auc"]):
            a2.text(v - 0.004, i, f"{v:.3f}", va="center", ha="right", fontsize=8.6,
                    color="#ffffff", fontweight="bold")
        a2.set_xlim(0.5, max(t["auc"]) * 1.04)
        finish(a2, "Ranking: a clear and consistent order", None,
               "Test ROC AUC", None)
        fig.suptitle("Six supervised models, one held-out set of 5,164 songs",
                     x=0.012, ha="left", fontsize=13.5, fontweight="bold", y=0.985)
        fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.6)
        save(fig, "fig57_model_comparison.png")

    with open(os.path.join(CLEAN, "xgb_metrics.json"), "w") as fh:
        json.dump(M, fh, indent=2, default=float)

    print("\n" + "=" * 72)
    print("XGBoost complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
