"""Figures for the report that are not produced by a notebook.

Run from the repository root:  python -m freight.report_figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from freight import style as S
from freight.validation import DATA_END, FOLDS, TEST

ROOT = Path(__file__).resolve().parents[1]
SPOTTER_START, SPOTTER_END = pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31")
SPOTTER_LOADS = 12_000  # size of validation.csv, as stated in the brief

ROLE_COLORS = {"train": S.BLUE, "validate": S.ORANGE, "test": S.AQUA, "spotter": S.NEUTRAL}
ROLE_TEXT = {"train": S.SURFACE, "validate": S.INK, "test": S.INK, "spotter": S.INK}


def validation_scheme() -> Path:
    """Timeline of the walk-forward folds, the one-time test and the final model."""
    dates = pd.read_csv(ROOT / "data" / "train_test.csv", usecols=["date"], parse_dates=["date"]).date

    def count(start, end):
        return int(dates.between(start, end).sum())

    rows = [(fold.name, [("train", fold.train_start, fold.train_end), ("validate", fold.eval_start, fold.eval_end)])
            for fold in FOLDS]
    rows.append(("Test", [("train", TEST.train_start, TEST.train_end), ("test", TEST.eval_start, TEST.eval_end)]))
    rows.append(("Final model", [("train", TEST.train_start, DATA_END), ("spotter", SPOTTER_START, SPOTTER_END)]))
    notes = {"Fold 1": None, "Fold 4": None, "Test": "scored once, at the end",
             "Final model": "predicts Spotter's loads"}

    fig = plt.figure(figsize=(10, 4.7))
    ax = fig.add_axes([0.11, S.bottom(fig, 0.7), 0.68, S.top(fig, 1.55) - S.bottom(fig, 0.7)])
    ys = [6.4, 5.4, 4.4, 3.4, 1.9, 0.9]  # a gap separates the folds from the test and final rows
    for y, (name, segments) in zip(ys, rows):
        for role, start, end in segments:
            left = mdates.date2num(start)
            width = mdates.date2num(end + pd.Timedelta(days=1)) - left
            ax.barh(y, width, left=left, height=0.56, color=ROLE_COLORS[role], edgecolor=S.SURFACE, linewidth=2)
            n = SPOTTER_LOADS if role == "spotter" else count(start, end)
            ax.text(left + width / 2, y, f"{n:,}", ha="center", va="center", fontsize=9, color=ROLE_TEXT[role])

    ax.set_yticks(ys, [name for name, _ in rows])
    ax.tick_params(axis="y", labelsize=10, labelcolor=S.INK)
    ax.set_ylim(0.35, 7.0)
    ax.set_xlim(mdates.date2num(pd.Timestamp("2025-01-01")), mdates.date2num(pd.Timestamp("2026-01-01")))
    # gridlines on month boundaries, month names centred within each month
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=16))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(axis="x", which="minor", length=0, labelsize=9, labelcolor=S.INK_2, pad=6)
    ax.grid(True, axis="x", which="major")
    ax.grid(False, axis="y")
    ax.spines["left"].set_visible(False)

    trans = ax.get_yaxis_transform()
    ax.annotate("", xy=(1.03, ys[0] + 0.3), xytext=(1.03, ys[3] - 0.3), xycoords=trans, textcoords=trans,
                arrowprops=dict(arrowstyle="-", color=S.AXIS, lw=1))
    ax.text(1.055, (ys[0] + ys[3]) / 2, "averaged to make\nevery model choice", transform=trans, va="center",
            fontsize=9.5, color=S.INK_2, linespacing=1.4)
    for y, (name, _) in zip(ys, rows):
        if notes.get(name):
            ax.text(1.055, y, notes[name], transform=trans, va="center", fontsize=9.5, color=S.INK_2)

    ax.legend(handles=[Patch(color=ROLE_COLORS["train"], label="Train"),
                       Patch(color=ROLE_COLORS["validate"], label="Validate"),
                       Patch(color=ROLE_COLORS["test"], label="Test"),
                       Patch(color=ROLE_COLORS["spotter"], label="Spotter's prediction set")],
              loc="lower left", bbox_to_anchor=(0, 1.02), ncol=4, handlelength=1, handleheight=1,
              columnspacing=1.6, borderaxespad=0)

    S.headline(fig, "Every check trains on the past and predicts the next two months",
               "Four walk-forward folds inside January–August drive all model choices. September–October is scored "
               "once at the end,\nand the final model trains on all ten months. Numbers are loads in each period")
    S.footnote(fig, "Load counts from train_test.csv before corrupted prices are removed; "
                    "Spotter's set is the 12,000 loads in validation.csv.")
    return S.save(fig, "validation_scheme", subdir="validation")


def main() -> None:
    S.apply_style()
    print(validation_scheme())


if __name__ == "__main__":
    main()
