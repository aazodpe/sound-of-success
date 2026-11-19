"""
CSCI 5612 - Project Part 4
Script 14: Linear and logistic regression.

Linear regression predicts a NUMBER   -> track_popularity (0-100)
Logistic regression predicts a CLASS  -> charted (yes/no)

Both are fitted on the training split, inspected on validation, and scored once
on the same held-out test songs used in Module 3, so the logistic result can be
compared directly against that module's Naive Bayes.

Writes ../data/clean/regression_metrics.json
       ../docs/images/fig39..fig43
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score, classification_report,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler

import theme

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "data", "clean")
IMG = os.path.join(HERE, "..", "docs", "images")
SEED = 5612

NICE = {"danceability": "Danceability", "energy": "Energy", "loudness": "Loudness",
        "speechiness": "Speechiness", "acousticness": "Acousticness",
        "instrumentalness": "Instrumentalness", "liveness": "Liveness",
        "valence": "Valence", "tempo": "Tempo", "duration_min": "Duration"}

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


def load(kind, part):
    d = pd.read_csv(os.path.join(CLEAN, f"m4_{kind}_{part}.csv"))
    target = "track_popularity" if kind == "regression" else "charted"
    return d.drop(columns=[target]), d[target]


# ---------------------------------------------------------------------------

def linear():
    print("\n  LINEAR REGRESSION  -  predicting track_popularity")
    Xtr, ytr = load("regression", "train")
    Xva, yva = load("regression", "val")
    Xte, yte = load("regression", "test")
    feats = list(Xtr.columns)

    model = LinearRegression().fit(Xtr, ytr)
    pred_va, pred_te = model.predict(Xva), model.predict(Xte)

    rmse = float(np.sqrt(mean_squared_error(yte, pred_te)))
    mae = float(mean_absolute_error(yte, pred_te))
    r2 = float(r2_score(yte, pred_te))

    # The only honest reference point: predicting the training mean every time.
    base_pred = np.full(len(yte), ytr.mean())
    base_rmse = float(np.sqrt(mean_squared_error(yte, base_pred)))
    base_mae = float(mean_absolute_error(yte, base_pred))

    print(f"    test RMSE {rmse:.3f}   MAE {mae:.3f}   R2 {r2:.4f}")
    print(f"    always-guess-the-mean   RMSE {base_rmse:.3f}   MAE {base_mae:.3f}")
    print(f"    validation R2 {r2_score(yva, pred_va):.4f}")

    # Raw coefficients say "per one unit of this variable". Because the units
    # differ wildly (tempo in BPM, danceability 0-1), a standardised version is
    # also computed so the sizes can be compared with each other.
    sc = StandardScaler().fit(Xtr)
    std_model = LinearRegression().fit(sc.transform(Xtr), ytr)
    coefs = pd.DataFrame({
        "feature": feats,
        "raw": model.coef_,
        "standardised": std_model.coef_,
    }).sort_values("standardised", key=abs, ascending=False)
    print("    strongest standardised coefficients:")
    for _, r in coefs.head(4).iterrows():
        print(f"      {r['feature']:<18} raw {r['raw']:+8.4f}   "
              f"per SD {r['standardised']:+.3f}")

    M["linear"] = {
        "target": "track_popularity", "n_train": int(len(ytr)),
        "n_val": int(len(yva)), "n_test": int(len(yte)),
        "rmse": rmse, "mae": mae, "r2": r2,
        "val_r2": float(r2_score(yva, pred_va)),
        "baseline_rmse": base_rmse, "baseline_mae": base_mae,
        "rmse_improvement": base_rmse - rmse,
        "intercept": float(model.intercept_),
        "target_sd": float(ytr.std()),
        "coefficients": {r["feature"]: {"raw": float(r["raw"]),
                                        "standardised": float(r["standardised"])}
                         for _, r in coefs.iterrows()},
    }
    return yte, pred_te, coefs


def logistic():
    print("\n  LOGISTIC REGRESSION  -  predicting charted")
    Xtr, ytr = load("logistic", "train")
    Xva, yva = load("logistic", "val")
    Xte, yte = load("logistic", "test")
    feats = list(Xtr.columns)

    sc = StandardScaler().fit(Xtr)
    Atr, Ava, Ate = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)

    out = {}
    for key, kw in [("default", {}), ("balanced", {"class_weight": "balanced"})]:
        m = LogisticRegression(max_iter=2000, random_state=SEED, **kw).fit(Atr, ytr)
        pred = m.predict(Ate)
        prob = m.predict_proba(Ate)[:, 1]
        acc = accuracy_score(yte, pred)
        rep = classification_report(yte, pred, output_dict=True, zero_division=0)
        out[key] = {
            "accuracy": float(acc),
            "confusion": confusion_matrix(yte, pred).tolist(),
            "roc_auc": float(roc_auc_score(yte, prob)),
            "recall_charted": float(rep["1"]["recall"]),
            "precision_charted": float(rep["1"]["precision"]),
            "val_accuracy": float(accuracy_score(yva, m.predict(Ava))),
            "coefficients": {f: float(c) for f, c in zip(feats, m.coef_[0])},
            "odds_ratios": {f: float(np.exp(c)) for f, c in zip(feats, m.coef_[0])},
        }
        print(f"    {key:<9} accuracy {acc:.4f}  ROC AUC {out[key]['roc_auc']:.4f}  "
              f"recall(charted) {out[key]['recall_charted']:.3f}")

    base = float(1 - yte.mean())
    print(f"    always-guess-'did not chart' baseline: {base:.4f}")

    od = pd.Series(out["default"]["odds_ratios"]).sort_values(key=lambda s: abs(np.log(s)),
                                                              ascending=False)
    print("    strongest odds ratios (per standard deviation):")
    for f, v in od.head(4).items():
        direction = "increases" if v > 1 else "decreases"
        print(f"      {f:<18} x{v:.3f}  ({direction} the odds by "
              f"{abs(v-1)*100:.1f}%)")

    M["logistic"] = out
    M["logistic"]["baseline"] = base
    M["logistic"]["n_test"] = int(len(yte))
    M["logistic"]["n_charted_test"] = int(yte.sum())
    return yte, out


# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("CSCI 5612 Project Part 4 - regression")
    print("=" * 72)

    yte_lin, pred_lin, coefs = linear()
    yte_log, logout = logistic()

    nb = json.load(open(os.path.join(CLEAN, "nb_metrics.json")))
    M["nb_comparison"] = {
        "nb_multinomial_accuracy": nb["charted"]["MultinomialNB"]["accuracy"],
        "nb_multinomial_recall": nb["charted"]["MultinomialNB"]["report"]["1"]["recall"],
        "nb_gaussian_accuracy": nb["charted"]["GaussianNB"]["accuracy"],
        "nb_gaussian_recall": nb["charted"]["GaussianNB"]["report"]["1"]["recall"],
    }

    rng = np.random.default_rng(SEED)
    draw = rng.choice(len(yte_lin), size=4000, replace=False)

    for mode in theme.MODES:
        theme.set_theme(mode)
        C = theme.C
        print(f"\n  --- figures ({mode}) ---")

        # 39. predicted against actual
        fig, ax = plt.subplots(figsize=(7.8, 6.2))
        ax.scatter(np.asarray(yte_lin)[draw], pred_lin[draw], s=7, alpha=0.28,
                   color=C[0], linewidths=0, rasterized=True)
        lo, hi = 0, 100
        ax.plot([lo, hi], [lo, hi], color=C[1], linewidth=2, linestyle="--")
        ax.annotate("perfect prediction", xy=(78, 78), xytext=(-8, 10),
                    textcoords="offset points", ha="right", color=C[1],
                    fontweight="bold", fontsize=9.3)
        ax.axhline(float(np.mean(pred_lin)), color=theme.INK_SOFT, linewidth=1.3,
                   linestyle=":")
        ax.annotate(f"every prediction falls between {pred_lin.min():.0f} and "
                    f"{pred_lin.max():.0f},\nwhile the real values span 0 to 100",
                    xy=(96, 9), ha="right", va="bottom", fontsize=9,
                    color=theme.INK_SOFT)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        finish(ax, "The model predicts almost the same number for every song",
               f"Predicted against actual popularity for {len(draw):,} held-out songs. "
               f"R² = {M['linear']['r2']:.3f}, RMSE = {M['linear']['rmse']:.1f} points.",
               "Actual popularity", "Predicted popularity")
        save(fig, "fig39_linear_pred_vs_actual.png")

        # 40. the coefficients, in comparable units
        cf = coefs.sort_values("standardised")
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        cols = [C[1] if v < 0 else C[0] for v in cf["standardised"]]
        ax.barh(range(len(cf)), cf["standardised"], height=0.64, color=cols,
                edgecolor=theme.BG, linewidth=1.5)
        ax.axvline(0, color=theme.INK_SOFT, linewidth=1.1)
        ax.set_yticks(range(len(cf)), [NICE[f] for f in cf["feature"]])
        for i, v in enumerate(cf["standardised"]):
            ax.text(v + (0.06 if v > 0 else -0.06), i, f"{v:+.2f}", va="center",
                    ha="left" if v > 0 else "right", fontsize=8.8,
                    color=theme.INK_SOFT)
        ax.set_xlim(cf["standardised"].min() * 1.45, cf["standardised"].max() * 1.45)
        finish(ax, "Energy and loudness pull hard in opposite directions",
               "Change in predicted popularity for a one-standard-deviation rise in each "
               "measurement, holding the rest fixed. The two largest nearly cancel.",
               "Points of popularity per standard deviation", None)
        save(fig, "fig40_linear_coefficients.png")

        # 41. logistic confusion matrix
        cm = np.array(logout["default"]["confusion"])
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
        ax.set_title("Logistic regression reaches the same dead end\n", loc="left")
        ax.text(0, 1.02, f"Accuracy {logout['default']['accuracy']*100:.1f}% against a "
                f"{M['logistic']['baseline']*100:.1f}% baseline, on the same "
                f"{M['logistic']['n_test']:,} songs used in Module 3.",
                transform=ax.transAxes, fontsize=9.8, color=theme.INK_SOFT, va="bottom")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        save(fig, "fig41_logistic_confusion.png")

        # 42. against Naive Bayes, on accuracy and on recall
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.4))
        names = ["Multinomial\nNaive Bayes", "Gaussian\nNaive Bayes",
                 "Logistic\nregression"]
        accs = [M["nb_comparison"]["nb_multinomial_accuracy"] * 100,
                M["nb_comparison"]["nb_gaussian_accuracy"] * 100,
                logout["default"]["accuracy"] * 100]
        recs = [M["nb_comparison"]["nb_multinomial_recall"] * 100,
                M["nb_comparison"]["nb_gaussian_recall"] * 100,
                logout["default"]["recall_charted"] * 100]
        for ax, vals, lab, ttl in [
                (a1, accs, "Accuracy (%)", "Accuracy says they are all fine"),
                (a2, recs, "Hits actually found (%)", "Recall says none of them work")]:
            ax.bar(range(3), vals, color=[C[4], C[2], C[0]], width=0.6,
                   edgecolor=theme.BG, linewidth=1.6)
            for i, v in enumerate(vals):
                ax.text(i, v + 2.2, f"{v:.1f}", ha="center", fontsize=9.6,
                        fontweight="bold", color=theme.INK_SOFT)
            ax.set_xticks(range(3), names, fontsize=8.8)
            ax.set_ylim(0, 100)
            ax.set_title(ttl, loc="left", fontsize=11.5)
            ax.set_ylabel(lab)
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        a1.axhline(M["logistic"]["baseline"] * 100, color=C[1], linestyle="--",
                   linewidth=1.8, label=f"Baseline ({M['logistic']['baseline']*100:.0f}%)")
        a1.legend(loc="lower right", fontsize=8.8)
        fig.suptitle("Three models, one conclusion", x=0.008, ha="left",
                     fontsize=13.5, fontweight="bold", y=1.0)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        save(fig, "fig42_logistic_vs_nb.png")

        # 43. what re-weighting the classes buys, and costs
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        keys = ["default", "balanced"]
        lab = ["As fitted", "With classes re-weighted"]
        acc = [logout[k]["accuracy"] * 100 for k in keys]
        rec = [logout[k]["recall_charted"] * 100 for k in keys]
        prc = [logout[k]["precision_charted"] * 100 for k in keys]
        x = np.arange(2); w = 0.26
        for off, vals, c, nm in [(-w, acc, C[0], "Accuracy"),
                                 (0, rec, C[2], "Hits found (recall)"),
                                 (w, prc, C[3], "Of those flagged, share correct")]:
            ax.bar(x + off, vals, w, color=c, edgecolor=theme.BG, linewidth=1.5, label=nm)
            for xi, v in zip(x + off, vals):
                ax.text(xi, v + 1.8, f"{v:.0f}", ha="center", fontsize=8.8,
                        fontweight="bold", color=theme.INK_SOFT)
        ax.set_xticks(x, lab)
        ax.set_ylim(0, 100)
        ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.11), fontsize=8.8)
        finish(ax, "Forcing the model to guess hits trades away far more than it gains",
               "The same logistic model, fitted normally and again with the rare class "
               "weighted up to match the common one",
               None, "Percent")
        save(fig, "fig43_logistic_balanced.png")

    with open(os.path.join(CLEAN, "regression_metrics.json"), "w") as fh:
        json.dump(M, fh, indent=2, default=float)

    print("\n" + "=" * 72)
    print("Regression complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
