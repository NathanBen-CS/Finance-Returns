"""Shared matplotlib style for the Phase 3 chart scripts: the same palette and look as phase2/charts2/chart_style.py."""
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_phase3" / "processed"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

# categorical slots in fixed order; the ramps run light -> dark
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
RED_RAMP = ["#fbe0df", "#f5b5b4", "#ee8988", "#e34948", "#c23433", "#962524", "#6b1817"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#eeede8"   # the gray midpoint of the diverging scale

REACTIONS = ["Strong Negative", "Negative", "Neutral", "Positive", "Strong Positive"]
SIZE_ORDER = ["Small", "Mid", "Large", "Mega-cap"]

# lift: red below 1 (less often than the base rate), gray at 1, blue above 1 (more often)
LIFT = LinearSegmentedColormap.from_list("lift", [RED_RAMP[4], RED_RAMP[2], RED_RAMP[0], NEUTRAL, BLUE_RAMP[0], BLUE_RAMP[2], BLUE_RAMP[4]])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True,
})


def lift_norm(values, floor=0.25):
    """Symmetric around 1, so equal colour means an equal distance from the base rate either way."""
    reach = max(abs(values.max() - 1), abs(values.min() - 1), floor)
    return TwoSlopeNorm(vmin=1 - reach, vcenter=1, vmax=1 + reach)


def headline(fig, title, sub):
    """Title and subtitle anchored to the figure's left edge (not the plot's), the subtitle wrapped to the figure width,
    then the plot laid out underneath them."""
    lines = textwrap.wrap(sub, width=int(fig.get_figwidth() * 13))
    height = fig.get_figheight()
    fig.text(0.012, 1 - 0.14 / height, title, fontsize=14, fontweight="bold", va="top")
    fig.text(0.012, 1 - 0.50 / height, "\n".join(lines), fontsize=9.5, color=INK2, va="top", linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.62 + 0.2 * len(lines)) / height))


def heatmap_frame(ax):
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save(fig, name):
    """headline() has already laid the figure out."""
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print(f"  wrote output/{name}")
