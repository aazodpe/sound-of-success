"""
CSCI 5612 - Project Part 3
Script 11: Decision trees.

Three trees are grown on the same training data, differing in depth and in the
measure used to score a split, so the effect of each choice can be seen rather
than asserted. A depth sweep then shows where the trees stop learning the topic
and start memorising the training set.

The script also computes a real worked example of entropy, Gini and information
gain, taken from the project's own data, for the Overview section.

Reads  ../data/clean/supervised_<task>_<train|test>.csv
Writes ../data/clean/dt_metrics.json
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
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

import theme

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "data", "clean")
IMG = os.path.join(HERE, "..", "docs", "images")
SEED = 5612

# Three genuinely different trees, not three settings of one tree.
TREES = [
    {"key": "shallow", "name": "Tree A - shallow", "max_depth": 3,
     "criterion": "gini",    "min_samples_leaf": 1},
    {"key": "medium",  "name": "Tree B - medium",  "max_depth": 6,
     "criterion": "entropy", "min_samples_leaf": 1},
    {"key": "full",    "name": "Tree C - unpruned", "max_depth": None,
     "criterion": "gini",    "min_samples_leaf": 1},
]

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


def plot_confusion(cm, classes, title, subtitle, name):
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    n = len(classes)
    fig, ax = plt.subplots(figsize=(1.05 * n + 4.0, 0.95 * n + 3.1))
    im = ax.imshow(cmn, cmap=theme.SEQ_CMAP, vmin=0, vmax=100)
    ax.set_xticks(range(n), classes, rotation=35, ha="right")
    ax.set_yticks(range(n), classes)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cmn[i, j]:.0f}%\n{cm[i, j]:,}", ha="center",
                    va="center", fontsize=8.6,
                    color="#ffffff" if cmn[i, j] > 55 else theme.INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("Share of the true class (%)", color=theme.INK_SOFT, fontsize=9.2)
    cb.outline.set_visible(False)
    ax.set_title(f"{title}\n", loc="left")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.8,
            color=theme.INK_SOFT, va="bottom")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    save(fig, name)


# ---------------------------------------------------------------------------
# A real worked example of split quality, for the Overview section
# ---------------------------------------------------------------------------

def entropy(counts):
    n = sum(counts)
    if n == 0:
        return 0.0
    p = np.array([c / n for c in counts if c > 0])
    return float(-(p * np.log2(p)).sum())


def gini(counts):
    n = sum(counts)
    if n == 0:
        return 0.0
    p = np.array([c / n for c in counts])
    return float(1 - (p ** 2).sum())


def worked_example():
    """
    'Is this song rap?' split on speechiness, using the real training data.
    Two classes keeps the arithmetic small enough to follow by hand.
    """
    Xtr, ytr, *_ = load("genre")
    is_rap = (ytr == "rap").values
    sp = Xtr["speechiness"].values
    thresh = 0.15

    parent = [int(is_rap.sum()), int((~is_rap).sum())]
    left_mask = sp <= thresh                      # not very speech-like
    right_mask = ~left_mask                        # speech-like
    left = [int(is_rap[left_mask].sum()), int((~is_rap)[left_mask].sum())]
    right = [int(is_rap[right_mask].sum()), int((~is_rap)[right_mask].sum())]

    n = sum(parent)
    wl, wr = sum(left) / n, sum(right) / n
    h_parent, h_left, h_right = entropy(parent), entropy(left), entropy(right)
    g_parent, g_left, g_right = gini(parent), gini(left), gini(right)
    h_child = wl * h_left + wr * h_right
    g_child = wl * g_left + wr * g_right

    ex = {
        "question": "Is this song rap?",
        "split": f"speechiness <= {thresh}",
        "threshold": thresh,
        "parent": {"rap": parent[0], "not_rap": parent[1], "n": n,
                   "entropy": round(h_parent, 4), "gini": round(g_parent, 4)},
        "left":  {"rap": left[0], "not_rap": left[1], "n": sum(left),
                  "weight": round(wl, 4), "entropy": round(h_left, 4),
                  "gini": round(g_left, 4)},
        "right": {"rap": right[0], "not_rap": right[1], "n": sum(right),
                  "weight": round(wr, 4), "entropy": round(h_right, 4),
                  "gini": round(g_right, 4)},
        "weighted_child_entropy": round(h_child, 4),
        "weighted_child_gini": round(g_child, 4),
        "information_gain": round(h_parent - h_child, 4),
        "gini_gain": round(g_parent - g_child, 4),
    }
    print("\n  worked example - 'Is this song rap?' split at speechiness "
          f"<= {thresh}")
    print(f"    parent   : {parent[0]:,} rap / {parent[1]:,} not  "
          f"entropy {h_parent:.4f}  gini {g_parent:.4f}")
    print(f"    left  <= : {left[0]:,} rap / {left[1]:,} not  "
          f"entropy {h_left:.4f}  weight {wl:.3f}")
    print(f"    right >  : {right[0]:,} rap / {right[1]:,} not  "
          f"entropy {h_right:.4f}  weight {wr:.3f}")
    print(f"    weighted child entropy {h_child:.4f}  ->  "
          f"information gain {h_parent - h_child:.4f}")
    return ex


# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("CSCI 5612 Project Part 3 - decision trees")
    print("=" * 70)

    split = json.load(open(os.path.join(CLEAN, "split_summary.json")))
    M["worked_example"] = worked_example()

    fitted, results = {}, {}
    for task in ["genre", "charted"]:
        Xtr, ytr, Xte, yte, feats = load(task)
        print(f"\n  task: {task}")
        results[task] = {}
        for spec in TREES:
            clf = DecisionTreeClassifier(
                max_depth=spec["max_depth"], criterion=spec["criterion"],
                min_samples_leaf=spec["min_samples_leaf"], random_state=SEED)
            clf.fit(Xtr, ytr)
            tr_acc = accuracy_score(ytr, clf.predict(Xtr))
            te_acc = accuracy_score(yte, clf.predict(Xte))
            results[task][spec["key"]] = {
                "name": spec["name"], "criterion": spec["criterion"],
                "max_depth": spec["max_depth"],
                "actual_depth": int(clf.get_depth()),
                "n_leaves": int(clf.get_n_leaves()),
                "train_accuracy": float(tr_acc), "test_accuracy": float(te_acc),
                "overfit_gap": float(tr_acc - te_acc),
                "confusion": confusion_matrix(yte, clf.predict(Xte)).tolist(),
                "classes": [str(c) for c in clf.classes_],
                "importances": {f: float(v) for f, v
                                in zip(feats, clf.feature_importances_)},
                "report": classification_report(yte, clf.predict(Xte),
                                                output_dict=True, zero_division=0),
            }
            if task == "genre":
                fitted[spec["key"]] = (clf, feats)
            print(f"    {spec['name']:<20} depth {clf.get_depth():>2}  "
                  f"leaves {clf.get_n_leaves():>5,}  "
                  f"train {tr_acc:.4f}  test {te_acc:.4f}  "
                  f"gap {tr_acc-te_acc:+.4f}")
        M[task] = results[task]
        M[task + "_baseline"] = split[task]["baseline_accuracy"]
        M[task + "_n_test"] = split[task]["n_test"]

    # Depth sweep on the genre task, to locate where memorising begins.
    Xtr, ytr, Xte, yte, feats = load("genre")
    depths = list(range(1, 25))
    sweep = []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(Xtr, ytr)
        sweep.append({"depth": d,
                      "train": float(accuracy_score(ytr, clf.predict(Xtr))),
                      "test": float(accuracy_score(yte, clf.predict(Xte)))})
    best = max(sweep, key=lambda r: r["test"])
    M["depth_sweep"] = sweep
    M["best_depth"] = best["depth"]
    M["best_depth_test"] = best["test"]
    print(f"\n  depth sweep: best held-out accuracy {best['test']:.4f} "
          f"at depth {best['depth']}")

    for mode in theme.MODES:
        theme.set_theme(mode)
        C = theme.C
        print(f"\n  --- figures ({mode}) ---")

        # 34. the shallow tree, drawn
        clf, feats = fitted["shallow"]
        fig, ax = plt.subplots(figsize=(15.5, 7.2))
        plot_tree(clf, feature_names=[f.replace("_", " ") for f in feats],
                  class_names=[c.upper() for c in clf.classes_],
                  filled=True, rounded=True, impurity=True, proportion=True,
                  fontsize=7, ax=ax, precision=2)
        for t in ax.texts:
            t.set_color("#11111a")
        ax.set_title("Tree A, read from the top down\n", loc="left")
        ax.text(0, 1.015, "A depth-three tree on the genre task. Each box shows the "
                "split, the Gini impurity, the share of training songs reaching it, "
                "and the class it would predict.",
                transform=ax.transAxes, fontsize=9.8, color=theme.INK_SOFT,
                va="bottom")
        save(fig, "fig34_tree_shallow.png")

        # 35. the three trees against each other
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        keys = [t["key"] for t in TREES]
        names = [results["genre"][k]["name"] for k in keys]
        tr = [results["genre"][k]["train_accuracy"] * 100 for k in keys]
        te = [results["genre"][k]["test_accuracy"] * 100 for k in keys]
        x = np.arange(len(keys)); w = 0.38
        ax.bar(x - w/2, tr, w, color=C[3], edgecolor=theme.BG, linewidth=1.6,
               label="Training set")
        ax.bar(x + w/2, te, w, color=C[0], edgecolor=theme.BG, linewidth=1.6,
               label="Held-out test set")
        for xi, v in zip(x - w/2, tr):
            ax.text(xi, v + 1.6, f"{v:.0f}", ha="center", fontsize=9,
                    fontweight="bold", color=theme.INK_SOFT)
        for xi, v in zip(x + w/2, te):
            ax.text(xi, v + 1.6, f"{v:.0f}", ha="center", fontsize=9,
                    fontweight="bold", color=theme.INK_SOFT)
        base = split["genre"]["baseline_accuracy"] * 100
        ax.axhline(base, color=C[1], linestyle="--", linewidth=1.8,
                   label=f"Guessing the commonest genre ({base:.0f}%)")
        lbl = [f"{results['genre'][k]['name']}\n{results['genre'][k]['criterion']}, "
               f"depth {results['genre'][k]['actual_depth']}, "
               f"{results['genre'][k]['n_leaves']:,} leaves" for k in keys]
        ax.set_xticks(x, lbl, fontsize=8.6)
        ax.set_ylim(0, 112)
        ax.legend(ncols=3, loc="upper left", fontsize=8.8)
        finish(ax, "The deepest tree learns the training set almost perfectly, and "
                   "generalises worst",
               "Accuracy of three different trees on the genre task, on the data "
               "they were built from and on data they had never seen",
               None, "Accuracy (%)")
        save(fig, "fig35_three_trees.png")

        # 36. where memorising starts
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        d = [r["depth"] for r in sweep]
        ax.plot(d, [r["train"] * 100 for r in sweep], color=C[3], linewidth=2.3,
                label="Training set")
        ax.plot(d, [r["test"] * 100 for r in sweep], color=C[0], linewidth=2.3,
                label="Held-out test set")
        ax.scatter([best["depth"]], [best["test"] * 100], s=150, facecolor="none",
                   edgecolor=C[1], linewidth=2.2, zorder=5)
        ax.annotate(f"best held-out accuracy\n{best['test']*100:.1f}% at depth {best['depth']}",
                    xy=(best["depth"], best["test"] * 100), xytext=(14, -4),
                    textcoords="offset points", va="top", color=C[1],
                    fontweight="bold", fontsize=9.3)
        ax.set_ylim(0, 105)
        ax.legend(loc="center right", fontsize=9.3)
        finish(ax, "Past a certain depth the tree stops learning and starts memorising",
               "Training and held-out accuracy on the genre task as the tree is "
               "allowed to grow deeper",
               "Maximum tree depth", "Accuracy (%)")
        save(fig, "fig36_tree_depth.png")

        # 37. confusion matrix for the best-generalising tree
        bk = max(keys, key=lambda k: results["genre"][k]["test_accuracy"])
        r = results["genre"][bk]
        plot_confusion(np.array(r["confusion"]), [c.upper() for c in r["classes"]],
                       "The best tree recognises some genres and confuses others",
                       f"{r['name']} on {split['genre']['n_test']:,} held-out songs. "
                       f"Accuracy {r['test_accuracy']*100:.1f}% against a "
                       f"{split['genre']['baseline_accuracy']*100:.1f}% baseline.",
                       "fig37_dt_confusion_genre.png")

        # 38. which measurements the tree actually used
        imp = pd.Series(results["genre"][bk]["importances"]).sort_values()
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        ax.barh(range(len(imp)), imp.values * 100, height=0.64, color=C[6],
                edgecolor=theme.BG, linewidth=1.5)
        ax.set_yticks(range(len(imp)),
                      [i.replace("_", " ").capitalize() for i in imp.index])
        for i, v in enumerate(imp.values * 100):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8.8,
                    color=theme.INK_SOFT)
        ax.set_xlim(0, imp.max() * 115)
        finish(ax, "Four measurements carry almost the whole tree",
               f"How much each measurement contributed to the splits in {r['name']}",
               "Share of the tree's total split quality (%)", None)
        save(fig, "fig38_dt_importance.png")

    with open(os.path.join(CLEAN, "dt_metrics.json"), "w") as fh:
        json.dump(M, fh, indent=2, default=float)

    # A readable text version of the shallow tree, for the repository.
    clf, feats = fitted["shallow"]
    with open(os.path.join(CLEAN, "tree_a_rules.txt"), "w") as fh:
        fh.write(export_text(clf, feature_names=list(feats)))

    print("\n" + "=" * 70)
    print("Decision trees complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
