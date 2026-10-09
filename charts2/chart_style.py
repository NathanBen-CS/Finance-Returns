"""Shared matplotlib style for the Phase 2 chart scripts (dataviz skill reference palette)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("../data_phase2/processed")
PHASE1 = Path("../data/processed")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

# categorical slots in fixed order; BLUE_RAMP is the sequential blue 100 -> 700
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = \
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SIZE_ORDER = ["Small", "Mid", "Large", "Mega-cap"]
VOL_ORDER = ["Low", "Medium", "High", "Very High"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True,  # gridlines behind marks
})


def subtitle(ax, text):
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")


def category_axis(ax):
    ax.grid(axis="x", visible=False)  # vertical rules add nothing on a category axis


def usd(x, _=None):
    for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if x >= div:
            v = x / div
            return f"${v:,.0f}{unit}" if v >= 10 or v == int(v) else f"${v:.1f}{unit}"
    return f"${x:,.0f}"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print("  wrote", name)
