"""
CSCI 5612 - Project Part 5
Script 19: Explanatory diagrams for the GradientBoosting and XGBoost overviews.

The two gradient boosting diagrams are not drawn by hand: a small boosting loop
is actually run on a one-dimensional toy problem, so the residuals, the running
total and the overfitting curve are measured from a real sequence of trees. The
two XGBoost diagrams are schematics of the objective function and of the early
stopping rule, which are rules rather than results.

Writes ../docs/images/diag_gb_*.png and diag_xgb_*.png (light and dark)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from sklearn.tree import DecisionTreeRegressor

import theme

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "..", "docs", "images")
SEED = 5612


def save(fig, name):
    out = theme.out_name(name)
    fig.savefig(os.path.join(IMG, out)); plt.close(fig)
    print(f"    saved {out}")


def head(fig, text):
    fig.suptitle(text, x=0.012, ha="left", fontsize=14, fontweight="bold", y=0.975)


def panel(ax, title, color=None):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_edgecolor(theme.GRID)
    ax.spines["bottom"].set_edgecolor(theme.GRID)
    ax.set_title(title, loc="left", fontsize=11.0, pad=9, color=color or theme.INK)


def toy(n=90, noise=0.55, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, n))
    truth = np.sin(x) + 0.28 * x
    return x.reshape(-1, 1), truth + rng.normal(0, noise, n), truth


def boost(X, y, rate, rounds, depth=2):
    """A minimal gradient boosting loop; yields the running total each round."""
    pred = np.full(len(y), y.mean())
    trees = []
    for _ in range(rounds):
        resid = y - pred                       # the negative gradient of
        t = DecisionTreeRegressor(max_depth=depth,
                                  random_state=SEED).fit(X, resid)
        trees.append(t)
        pred = pred + rate * t.predict(X)
        yield pred.copy(), resid.copy(), t


# ---------------------------------------------------------------------------

def diagram_gb_residuals():
    """Four stages of one real boosting run: fit, residual, correct, repeat."""
    C = theme.C
    X, y, truth = toy()
    x = X.ravel()
    rate = 0.6

    steps = list(boost(X, y, rate=rate, rounds=30))
    p1, r1, t1 = steps[0]
    p2, r2, t2 = steps[1]
    p30 = steps[-1][0]

    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.9))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.76, bottom=0.17, wspace=0.26)
    a1, a2, a3, a4 = axes

    # 1. the starting guess
    a1.scatter(x, y, s=18, color=theme.GRID, linewidths=0)
    a1.axhline(y.mean(), color=C[0], linewidth=2.6)
    a1.annotate("one flat guess:\nthe average", xy=(6.4, y.mean()),
                xytext=(0, 16), textcoords="offset points", ha="center",
                fontsize=8.8, color=C[0], fontweight="bold")
    panel(a1, "1 · Start with the average", C[0])

    # 2. what is left over
    a2.axhline(0, color=theme.INK_SOFT, linewidth=1)
    a2.scatter(x, r1, s=18, color=C[1], linewidths=0, alpha=0.85)
    order = np.argsort(x)
    a2.step(x[order], (rate * t1.predict(X))[order], where="mid",
            color=C[2], linewidth=2.4)
    a2.annotate("a small tree fitted\nto the leftovers", xy=(3.0, 1.0),
                xytext=(0, 0), textcoords="offset points", ha="center",
                fontsize=8.8, color=C[2], fontweight="bold")
    panel(a2, "2 · Fit the leftovers", C[1])

    # 3. add it on
    a3.scatter(x, y, s=18, color=theme.GRID, linewidths=0)
    a3.step(x[order], p1[order], where="mid", color=C[0], linewidth=1.5,
            alpha=0.55, label="after 1 tree")
    a3.step(x[order], p2[order], where="mid", color=C[0], linewidth=2.6,
            label="after 2 trees")
    a3.legend(fontsize=8.2, loc="upper left")
    panel(a3, "3 · Add it to the running total", C[0])

    # 4. repeat
    a4.scatter(x, y, s=18, color=theme.GRID, linewidths=0)
    a4.plot(x, truth, color=theme.INK_SOFT, linewidth=1.6, linestyle="--",
            label="the real pattern")
    a4.step(x[order], p30[order], where="mid", color=C[5], linewidth=2.6,
            label="after 30 trees")
    a4.legend(fontsize=8.2, loc="upper left")
    panel(a4, "4 · Repeat thirty times", C[5])

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_xlabel("one measurement", fontsize=8.4)
    a2.set_ylabel("what is still wrong", fontsize=8.4)
    a1.set_ylabel("the thing being predicted", fontsize=8.4)

    fig.text(0.5, 0.035, "Each tree is fitted to the error the running total "
             "still has, then added on at a fraction of its full size. Nothing "
             "is re-weighted; the target itself changes every round.",
             ha="center", fontsize=9, color=theme.INK_SOFT)
    head(fig, "Gradient boosting chases what is left over")
    save(fig, "diag_gb_residuals.png")


def diagram_gb_learning_rate():
    """What the learning rate buys, and why too many trees hurts."""
    C = theme.C
    X, y, truth = toy()
    x = X.ravel()
    order = np.argsort(x)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.5))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.78, bottom=0.17, wspace=0.22)

    # --- left: three ways of spending the same corrections ---------------
    a1.scatter(x, y, s=18, color=theme.GRID, linewidths=0)
    for (rate, rounds, col, lab) in [
            (1.0, 6, C[1], "rate 1.0, 6 trees  ·  big steps, jumpy"),
            (0.15, 6, C[3], "rate 0.15, 6 trees  ·  barely started"),
            (0.15, 60, C[2], "rate 0.15, 60 trees  ·  small steps, arrives")]:
        p = list(boost(X, y, rate=rate, rounds=rounds))[-1][0]
        a1.step(x[order], p[order], where="mid", color=col, linewidth=2.3, label=lab)
    a1.plot(x, truth, color=theme.INK_SOFT, linewidth=1.5, linestyle="--",
            label="the real pattern")
    a1.legend(fontsize=8.4, loc="upper left")
    a1.set_xticks([]); a1.set_yticks([]); a1.grid(False)
    a1.set_xlabel("one measurement", fontsize=8.6)
    panel(a1, "The learning rate sets the size of each correction", C[2])

    # --- right: train and validation error as trees pile up --------------
    rng = np.random.default_rng(11)
    idx = rng.permutation(len(y))
    tr, va = idx[:60], idx[60:]
    Xtr, ytr = X[tr], y[tr]
    tr_err, va_err = [], []
    pred_tr = np.full(len(tr), ytr.mean())
    pred_va = np.full(len(va), ytr.mean())
    for _ in range(140):
        resid = ytr - pred_tr
        t = DecisionTreeRegressor(max_depth=3, random_state=SEED).fit(Xtr, resid)
        pred_tr = pred_tr + 0.3 * t.predict(Xtr)
        pred_va = pred_va + 0.3 * t.predict(X[va])
        tr_err.append(np.mean((ytr - pred_tr) ** 2))
        va_err.append(np.mean((y[va] - pred_va) ** 2))
    n = np.arange(1, 141)
    k = int(np.argmin(va_err)) + 1
    a2.plot(n, tr_err, color=C[0], linewidth=2.2, label="Training error")
    a2.plot(n, va_err, color=C[1], linewidth=2.2, label="Validation error")
    a2.axvline(k, color=theme.INK_SOFT, linestyle="--", linewidth=1.5)
    a2.annotate(f"best at {k} trees", xy=(k, max(va_err) * 0.92), xytext=(7, 0),
                textcoords="offset points", fontsize=8.8, color=theme.INK_SOFT)
    a2.fill_between(n[k:], 0, max(va_err), color=C[1], alpha=0.07)
    a2.annotate("every tree after this one\nis fitting noise", xy=(k + 55, max(va_err) * 0.55),
                ha="center", fontsize=8.8, color=C[1], fontweight="bold")
    a2.set_ylim(0, max(va_err) * 1.05)
    a2.legend(fontsize=8.6, loc="center right")
    a2.set_xlabel("Trees added", fontsize=8.6); a2.set_ylabel("Squared error", fontsize=8.6)
    a2.set_yticks([])
    panel(a2, "Too many trees: training keeps improving, validation does not", C[1])

    head(fig, "Two settings that decide whether boosting helps or hurts")
    save(fig, "diag_gb_learning_rate.png")


def diagram_xgb_objective():
    """Loss alone versus loss plus a penalty on complexity."""
    C = theme.C
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.7),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.035, right=0.975, top=0.77, bottom=0.12, wspace=0.18)

    # --- left: two candidate trees ---------------------------------------
    a1.set_xlim(0, 10); a1.set_ylim(0, 10)
    a1.axis("off")

    def node(ax, cx, cy, w, h, label, edge_col, fill=None, fs=8.2):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.08,rounding_size=0.16",
                                    linewidth=1.5, edgecolor=edge_col,
                                    facecolor=fill or "none", zorder=3))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fs, zorder=4,
                color="#ffffff" if fill else theme.INK, fontweight="bold")

    def edge(ax, x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color=theme.GRID, linewidth=1.4, zorder=0)

    # A simple tree: two leaves, cautious weights.
    a1.text(2.1, 9.4, "A simple tree", ha="center", fontsize=10.5,
            fontweight="bold", color=C[2])
    node(a1, 2.1, 8.2, 2.5, 0.85, "energy < 0.6", theme.GRID)
    edge(a1, 2.1, 7.78, 1.1, 6.5); edge(a1, 2.1, 7.78, 3.1, 6.5)
    node(a1, 1.1, 6.15, 1.0, 0.68, "+0.4", C[2], C[2], 8.6)
    node(a1, 3.1, 6.15, 1.0, 0.68, "-0.3", C[2], C[2], 8.6)
    a1.text(2.1, 5.35, "2 leaves  ·  small weights", ha="center", fontsize=8.6,
            color=theme.INK_SOFT)

    # An eager tree: six leaves, confident weights.
    a1.text(7.4, 9.4, "An eager tree", ha="center", fontsize=10.5,
            fontweight="bold", color=C[1])
    node(a1, 7.4, 8.2, 2.5, 0.85, "energy < 0.6", theme.GRID)
    edge(a1, 7.4, 7.78, 6.1, 7.5); edge(a1, 7.4, 7.78, 8.7, 7.5)
    node(a1, 6.1, 7.2, 1.85, 0.62, "tempo < 118", theme.GRID, None, 7.4)
    node(a1, 8.7, 7.2, 1.85, 0.62, "valence < 0.4", theme.GRID, None, 7.4)
    for px, lx, val in [(6.1, 5.4, "+3.1"), (6.1, 6.7, "-2.6"),
                        (8.7, 8.1, "+2.9"), (8.7, 9.4, "-3.4")]:
        edge(a1, px, 6.89, lx, 6.49)
        node(a1, lx, 6.15, 1.02, 0.68, val, C[1], C[1], 8.4)
    a1.text(7.4, 5.35, "4 leaves  ·  large weights", ha="center", fontsize=8.6,
            color=theme.INK_SOFT)

    a1.text(5.0, 4.45, "Ordinary boosting judges both on training error alone, so "
            "the eager tree wins.\nXGBoost puts the price of being eager into the "
            "thing it is minimising.",
            ha="center", va="center", fontsize=9.2, color=theme.INK_SOFT)
    a1.text(5.0, 2.55,
            r"objective  =  loss  +  $\gamma\,T$  +  $\frac{1}{2}\lambda\sum w^2$",
            ha="center", va="center", fontsize=13.5, color=theme.INK)
    a1.text(5.0, 1.35, "T is the number of leaves;  w is each leaf's weight;\n"
            r"$\gamma$ and $\lambda$ are how dearly each is charged",
            ha="center", va="center", fontsize=8.6, color=theme.INK_SOFT)

    # --- right: what the two terms actually charge -----------------------
    leaves = np.arange(1, 13)
    gamma = 0.08
    a2.bar(leaves - 0.19, gamma * leaves, width=0.36, color=C[4],
           edgecolor=theme.BG, linewidth=1.0, label=r"$\gamma T$  ·  per leaf")
    w = np.linspace(0.3, 3.2, 12)
    a2.bar(leaves + 0.19, 0.5 * 0.5 * w ** 2, width=0.36, color=C[6],
           edgecolor=theme.BG, linewidth=1.0,
           label=r"$\frac{1}{2}\lambda w^2$  ·  per unit of weight")
    a2.legend(fontsize=8.8, loc="upper left")
    a2.set_xticks([1, 4, 8, 12])
    a2.set_yticks([])
    a2.set_xlabel("Complexity  ·  leaves in the tree, or size of one leaf's weight",
                   fontsize=8.8)
    a2.set_ylabel("Penalty added to the objective", fontsize=8.8)
    panel(a2, "One term charges by the leaf, the other by the leap", C[6])
    a2.text(0.5, -0.30, "The leaf count is charged in a straight line; the leaf "
            "weights are charged by their square,\nso one very confident leaf "
            "costs far more than several cautious ones.",
            transform=a2.transAxes, ha="center", fontsize=8.6,
            color=theme.INK_SOFT)

    head(fig, "What XGBoost adds: a price on complexity, inside the objective")
    save(fig, "diag_xgb_objective.png")


def diagram_xgb_early_stopping():
    """The stopping rule, drawn on a curve shaped like the real ones."""
    C = theme.C
    n = np.arange(1, 221)
    rng = np.random.default_rng(3)
    # A curve with the shape every boosting run produces: fast gain, a peak,
    # then a slow decay as the later trees start memorising.
    val = 0.78 - 0.26 * np.exp(-n / 26) - 0.00034 * n + rng.normal(0, 0.0016, len(n))
    train = 0.80 + 0.19 * (1 - np.exp(-n / 60))
    best = int(val.argmax()) + 1
    patience = 30
    stop = best + patience

    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.17)

    ax.plot(n, train, color=C[0], linewidth=2.2, label="Training score")
    ax.plot(n, val, color=C[1], linewidth=2.2, label="Validation score")
    ax.axvline(best, color=C[2], linewidth=1.8, linestyle="--")
    ax.scatter([best], [val[best - 1]], s=170, facecolor="none", edgecolor=C[2],
               linewidth=2.2, zorder=6)
    ax.annotate(f"best round ({best})\nthis model is kept",
                xy=(best, val[best - 1]), xytext=(-18, 52),
                textcoords="offset points", ha="right", va="bottom", color=C[2],
                fontweight="bold", fontsize=9.2,
                arrowprops=dict(arrowstyle="-", color=C[2], linewidth=1.1,
                                shrinkA=2, shrinkB=9))
    ax.axvspan(best, stop, color=C[1], alpha=0.10)
    ax.annotate(f"{patience} rounds of patience:\nno improvement, so stop",
                xy=((best + stop) / 2, 0.585), ha="center", fontsize=9.0,
                color=C[1], fontweight="bold")
    ax.axvline(stop, color=theme.INK_SOFT, linewidth=1.6)
    ax.annotate("training halts here", xy=(stop, 0.90), xytext=(8, 0),
                textcoords="offset points", fontsize=8.8, color=theme.INK_SOFT)
    ax.axvspan(stop, n[-1], color=theme.GRID, alpha=0.35)
    ax.annotate("never fitted", xy=((stop + n[-1]) / 2, 0.555), ha="center",
                fontsize=8.8, color=theme.INK_SOFT)
    ax.set_ylim(0.50, 1.0)
    ax.legend(fontsize=9, loc="lower right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("Boosting rounds"); ax.set_ylabel("Score")
    ax.set_title("Early stopping: the validation set decides when to quit\n", loc="left")
    ax.text(0, 1.015, "Schematic, drawn to the shape the real curves take. The "
            "number of trees stops being a setting to guess and becomes "
            "something the data chooses.",
            transform=ax.transAxes, fontsize=9.8, color=theme.INK_SOFT, va="bottom")
    save(fig, "diag_xgb_early_stopping.png")


if __name__ == "__main__":
    print("Building Module 5 concept diagrams")
    for mode in theme.MODES:
        theme.set_theme(mode)
        print(f"  --- {mode} ---")
        diagram_gb_residuals()
        diagram_gb_learning_rate()
        diagram_xgb_objective()
        diagram_xgb_early_stopping()
    print("Done.")
