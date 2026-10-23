"""One visual style for every figure in the project.

Colours follow a colour-blind-validated categorical order (blue, orange, aqua),
a single-hue blue ramp for magnitude and a blue/red diverging pair with a grey
midpoint. Text always uses ink colours, never series colours.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# Surfaces and ink
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#c3c2b7"  # de-emphasised marks

# Categorical slots, fixed order
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
EQUIPMENT_COLORS = {"Dry Van": BLUE, "Reefer": ORANGE, "Flatbed": AQUA}
EQUIPMENT_ORDER = ["Dry Van", "Reefer", "Flatbed"]

# Blue ramp (light -> dark) for magnitude
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE_DARK = "#184f95"
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", ["#eef5fd"] + BLUE_RAMP)

# Diverging: cheaper (blue) <-> pricier (red), grey midpoint
RED = "#e34948"
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a99", "#c22f2f"]
)

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
SOURCE_NOTE = "Source: train_test.csv, 48,000 labelled loads, Jan–Oct 2025"


def apply_style() -> None:
    """Set matplotlib defaults shared by every figure."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10.5,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_2,
            "axes.labelsize": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "lines.linewidth": 1.8,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "legend.labelcolor": INK_2,
            "legend.handlelength": 1.4,
            "patch.linewidth": 0,
        }
    )


def headline(fig, title: str, subtitle: str | None = None, *, x: float = 0.012) -> None:
    """Figure-level headline (the finding) and a subtitle (what is plotted).

    Positions are fixed in inches from the top edge so every figure height
    gets the same header spacing.
    """
    h = fig.get_figheight()
    fig.text(x, 1 - 0.16 / h, title, ha="left", va="top", fontsize=15, fontweight="semibold", color=INK)
    if subtitle:
        fig.text(x, 1 - 0.50 / h, subtitle, ha="left", va="top", fontsize=10.5, color=INK_2,
                 linespacing=1.4)


def footnote(fig, text: str = SOURCE_NOTE, *, x: float = 0.012) -> None:
    fig.text(x, 0.10 / fig.get_figheight(), text, ha="left", va="bottom", fontsize=8.5, color=MUTED)


def top(fig, inches: float = 1.25) -> float:
    """Figure fraction where the plotting area starts, leaving room for the header."""
    return 1 - inches / fig.get_figheight()


def bottom(fig, inches: float = 0.75) -> float:
    """Figure fraction where the plotting area ends, leaving room for axis labels and footnote."""
    return inches / fig.get_figheight()


def panel_title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=11, fontweight="semibold", color=INK, pad=10)


def pct(v: float, decimals: int = 0, signed: bool = False) -> str:
    """0.05 -> '5%' (or '+5%'), with a true minus sign and an unsigned zero."""
    if round(v * 100, decimals) == 0:
        return f"{0:.{decimals}f}%"
    text = f"{v:{'+' if signed else ''}.{decimals}%}"
    return text.replace("-", "−")


def pct_axis(ax, axis: str = "y", decimals: int = 0, signed: bool = False) -> None:
    """Format an axis holding fractions (0.05) as percentages (5%)."""
    formatter = FuncFormatter(lambda v, _: pct(v, decimals, signed))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def _compact(v: float) -> str:
    return f"{v / 1000:g}K" if abs(v) >= 1000 else f"{v:g}"


def dollar_axis(ax, axis: str = "y") -> None:
    formatter = FuncFormatter(lambda v, _: f"${_compact(v)}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def thousands_axis(ax, axis: str = "y", unit: str = "") -> None:
    formatter = FuncFormatter(lambda v, _: f"{_compact(v)}{unit}".replace("-", "−"))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def month_axis(ax, months=range(1, 11)) -> None:
    """Month-name ticks (Jan, Feb, ...) on a date x-axis, limited to the months given."""
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=list(months)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def save(fig, name: str, subdir: str = "eda") -> Path:
    """Save to reports/figures/<subdir>/<name>.png and return the path."""
    out = FIG_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=200)
    return path
