"""
Shared figure theming for the project.

Every figure is rendered twice, once for the light site and once for the dark
site, so that no chart is a glaring white rectangle on a dark page. Importing
this module and calling set_theme("light") or set_theme("dark") configures
matplotlib and exposes the colours the plotting scripts draw with.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MODES = ("light", "dark")

# Categorical palettes. The same eight hues, stepped for each surface, in a
# fixed order that is never cycled or reassigned between charts.
CAT = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500",
              "#d55181", "#4bb84b", "#9085e9", "#e66767"],
}

SURFACE = {
    "light": {"bg": "#ffffff", "ink": "#1b1d21", "soft": "#4a4f57",
              "grid": "#e2e5ea", "accent": "#4a3aa7"},
    "dark":  {"bg": "#16161a", "ink": "#e8e6e3", "soft": "#9b968f",
              "grid": "#2e2e36", "accent": "#9085e9"},
}

# Sequential: one hue, light surface to saturated. Diverging: two hues with a
# neutral midpoint that matches the surface it sits on.
SEQ = {
    "light": "Purples",
    "dark": LinearSegmentedColormap.from_list(
        "seq_dark", ["#16161a", "#3a3269", "#6a5cc4", "#a99df0", "#ded8fb"]),
}
DIV = {
    "light": "RdBu_r",
    "dark": LinearSegmentedColormap.from_list(
        "div_dark", ["#3987e5", "#2c4f80", "#2e2e36", "#8a4444", "#e66767"]),
}

# Populated by set_theme() so the plotting scripts can just read them.
C = list(CAT["light"])
BG = INK = INK_SOFT = GRID = ACCENT = None
SEQ_CMAP = DIV_CMAP = None
SUFFIX = ""


def set_theme(mode):
    """Configure matplotlib for one mode and expose that mode's colours."""
    global C, BG, INK, INK_SOFT, GRID, ACCENT, SEQ_CMAP, DIV_CMAP, SUFFIX
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")

    s = SURFACE[mode]
    C = list(CAT[mode])
    BG, INK, INK_SOFT, GRID, ACCENT = s["bg"], s["ink"], s["soft"], s["grid"], s["accent"]
    SEQ_CMAP, DIV_CMAP = SEQ[mode], DIV[mode]
    SUFFIX = "" if mode == "light" else "_dark"

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.titlecolor": INK,
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.facecolor": BG,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "legend.labelcolor": INK,
        "text.color": INK,
        "figure.edgecolor": BG,
    })
    return mode


def out_name(name):
    """figXX.png -> figXX.png (light) or figXX_dark.png (dark)."""
    stem, _, ext = name.rpartition(".")
    return f"{stem}{SUFFIX}.{ext}"
