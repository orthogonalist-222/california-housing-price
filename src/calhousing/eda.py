"""Exploratory figures — regenerable, deterministic, and each one arguing a point.

Two rules this module follows.

**Every figure answers a question the modelling will later have to face.** A
gallery of plots nobody acts on is decoration; each function below is named for
the claim it supports, and each claim reappears in a later design decision
(ADR-001's censoring, ADR-002's encoder choice, the numeric block's de-skew).

**The output is byte-identical on a rerun.** Figures are cattle: they are
deleted and rebuilt by the recipe, and a build that cannot reproduce its own
output cannot be held to a staleness gate later. Matplotlib stamps its version
into PNG metadata by default, which makes every rebuild differ for no reason —
``_save`` strips it.

Run with::

    uv run python -m calhousing.eda
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on CI or in a Kaggle kernel

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .data import load_raw
from .splits import coverage_report, income_band, split_train_test

__all__ = ["build_all", "DEFAULT_OUTPUT_DIR"]

DEFAULT_OUTPUT_DIR = Path("artifacts/eda")

#: Figure style. Fixed here rather than inherited from a matplotlibrc, which
#: would make one machine's output differ from another's.
_STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

_WATER = "#2a6f97"
_LAND = "#bc6c25"
_ACCENT = "#9d0208"


def _save(fig: plt.Figure, path: Path) -> None:
    """Write a PNG with no environment-dependent bytes.

    ``metadata={"Software": None}`` is the load-bearing part: matplotlib
    otherwise writes its own version string into the PNG, so the same figure
    built by two different matplotlib versions — or the same one after an
    upgrade — differs in bytes while being identical as a picture. A
    determinism check would then fail for a reason that has nothing to do with
    the data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


# --- the figures -------------------------------------------------------------


def fig_target_is_censored(frame: pd.DataFrame, out: Path) -> Path:
    """The cap is not a tail, it is a wall. (ADR-001)"""
    path = out / "01-target-censoring.png"
    target = frame[config.TARGET]
    at_cap = int((target == config.TARGET_CAP).sum())
    at_floor = int((target == config.TARGET_FLOOR).sum())

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.hist(target, bins=120, color=_WATER)
    ax.axvline(config.TARGET_CAP, color=_ACCENT, lw=1.2, ls="--")
    ax.annotate(
        f"{at_cap:,} rows pinned at the ${config.TARGET_CAP:,.0f} cap",
        xy=(config.TARGET_CAP, ax.get_ylim()[1] * 0.85),
        xytext=(-12, 0),
        textcoords="offset points",
        ha="right",
        color=_ACCENT,
        fontsize=8.5,
    )
    ax.axvline(config.TARGET_FLOOR, color=_ACCENT, lw=1.2, ls=":")
    ax.annotate(
        f"{at_floor} at the ${config.TARGET_FLOOR:,.0f} floor",
        xy=(config.TARGET_FLOOR, ax.get_ylim()[1] * 0.55),
        xytext=(14, 0),
        textcoords="offset points",
        ha="left",
        color=_ACCENT,
        fontsize=8.5,
    )
    ax.set_xlabel("median_house_value ($)")
    ax.set_ylabel("block groups")
    ax.set_title("The target is censored at both ends", loc="left", fontsize=10.5)
    _save(fig, path)
    return path


def fig_rare_category(frame: pd.DataFrame, out: Path) -> Path:
    """Why the split collapses a level, and why the encoder handles unknowns."""
    path = out / "02-ocean-proximity-rarity.png"
    counts = frame["ocean_proximity"].value_counts()

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    colors = [_ACCENT if n < 100 else _WATER for n in counts]
    bars = ax.barh(list(counts.index)[::-1], counts.to_numpy()[::-1], color=colors[::-1])
    ax.set_xscale("log")
    ax.set_xlabel("block groups (log scale)")
    for bar, value in zip(bars, counts.to_numpy()[::-1]):
        ax.text(
            value * 1.15,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            fontsize=8.5,
        )
    ax.set_title(
        "One level has five rows — a log axis is the only way to see it",
        loc="left",
        fontsize=10.5,
    )
    ax.grid(axis="y", visible=False)
    _save(fig, path)
    return path


def fig_geography(frame: pd.DataFrame, out: Path) -> Path:
    """Price is spatial, which is what earns the KMeans-similarity feature."""
    path = out / "03-geography.png"
    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    scatter = ax.scatter(
        frame["longitude"],
        frame["latitude"],
        c=frame[config.TARGET],
        cmap="viridis",
        s=frame["population"] / 120,
        alpha=0.35,
        linewidths=0,
    )
    fig.colorbar(scatter, ax=ax, label="median_house_value ($)", shrink=0.8)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(
        "Value is spatial: the coast is expensive, the interior is not",
        loc="left",
        fontsize=10.5,
    )
    _save(fig, path)
    return path


def fig_correlations(frame: pd.DataFrame, out: Path) -> Path:
    """The raw counts correlate with each other, not with price — the argument
    for ratio features rather than raw totals."""
    path = out / "04-correlations.png"
    numeric = frame[config.NUMERIC_COLUMNS + [config.TARGET]]
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(value) > 0.55 else "black",
            )
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.grid(visible=False)
    ax.set_title(
        "total_rooms / bedrooms / population / households move together",
        loc="left",
        fontsize=10.5,
    )
    _save(fig, path)
    return path


def fig_missingness(frame: pd.DataFrame, out: Path) -> Path:
    """Where the nulls are, and whether they are patterned."""
    path = out / "05-missingness.png"
    missing = frame["total_bedrooms"].isna()
    by_level = (
        pd.DataFrame({"missing": missing, "level": frame["ocean_proximity"]})
        .groupby("level")["missing"]
        .agg(["sum", "mean", "count"])
        .sort_values("count", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar(by_level.index, by_level["mean"] * 100, color=_WATER)
    ax.set_ylabel("% of rows with total_bedrooms missing")
    overall = missing.mean() * 100
    ax.axhline(overall, color=_ACCENT, ls="--", lw=1.1)
    ax.annotate(
        f"overall {overall:.2f}%  ({int(missing.sum())} rows)",
        xy=(0.99, overall),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        color=_ACCENT,
        fontsize=8.5,
    )
    ax.set_title(
        "Missingness is spread across levels, not concentrated in one",
        loc="left",
        fontsize=10.5,
    )
    ax.tick_params(axis="x", labelsize=8)
    _save(fig, path)
    return path


def fig_skew(frame: pd.DataFrame, out: Path) -> Path:
    """The de-skew option in the numeric block is not speculative."""
    path = out / "06-heavy-tails.png"
    columns = config.HEAVY_TAILED_COLUMNS
    fig, axes = plt.subplots(
        2, len(columns), figsize=(2.6 * len(columns), 4.6), sharex="col"
    )
    for index, column in enumerate(columns):
        series = frame[column].dropna()
        axes[0, index].hist(series, bins=60, color=_LAND)
        axes[0, index].set_title(f"{column}\nskew {series.skew():.2f}", fontsize=9)
        axes[1, index].hist(np.log1p(series), bins=60, color=_WATER)
        axes[1, index].set_title(
            f"log1p — skew {np.log1p(series).skew():.2f}", fontsize=8.5
        )
        for row in (0, 1):
            axes[row, index].tick_params(labelsize=7)
    fig.suptitle(
        "Raw counts are heavily right-skewed; a log fixes it",
        x=0.01,
        ha="left",
        fontsize=10.5,
    )
    fig.tight_layout()
    _save(fig, path)
    return path


def fig_income_and_split(frame: pd.DataFrame, out: Path) -> Path:
    """What the stratified split preserves, shown rather than asserted."""
    path = out / "07-income-bands-and-split.png"
    train, test = split_train_test(frame)
    bands = config.INCOME_BAND_LABELS
    whole = income_band(frame).value_counts(normalize=True).reindex(bands).fillna(0)
    tr = income_band(train).value_counts(normalize=True).reindex(bands).fillna(0)
    te = income_band(test).value_counts(normalize=True).reindex(bands).fillna(0)

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    width = 0.27
    positions = np.arange(len(bands))
    left.bar(positions - width, whole.to_numpy(), width, label="whole", color="#8d99ae")
    left.bar(positions, tr.to_numpy(), width, label="train", color=_WATER)
    left.bar(positions + width, te.to_numpy(), width, label="test", color=_LAND)
    left.set_xticks(positions, [str(b) for b in bands])
    left.set_xlabel("income band")
    left.set_ylabel("share of rows")
    left.legend(fontsize=8, frameon=False)
    left.set_title("Stratification preserves the income mix", loc="left", fontsize=10.5)

    report = coverage_report(train, test)
    right.axis("off")
    table = right.table(
        cellText=[[f"{row.train:,}", f"{row.test:,}"] for row in report.itertuples()],
        rowLabels=list(report.index),
        colLabels=["train", "test"],
        loc="center",
        cellLoc="right",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.45)
    right.set_title("...and every level reaches both sides", loc="left", fontsize=10.5)
    fig.tight_layout()
    _save(fig, path)
    return path


#: Build order. Also the order the notebook shows them in.
_FIGURES = [
    fig_target_is_censored,
    fig_rare_category,
    fig_geography,
    fig_correlations,
    fig_missingness,
    fig_skew,
    fig_income_and_split,
]


def build_all(
    frame: pd.DataFrame | None = None,
    out: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    verbose: bool = True,
) -> list[Path]:
    """Build every figure into ``out`` and return the paths, in order.

    The digests printed here are the evidence for the determinism gate: run it
    twice and the two lists must match.
    """
    frame = load_raw() if frame is None else frame
    out = Path(out)
    with plt.rc_context(_STYLE):
        paths = [build(frame, out) for build in _FIGURES]
    if verbose:
        for path in paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            print(f"  {path}  sha256:{digest}")
        print(f"[calhousing] {len(paths)} figures -> {out}")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the exploratory figures.")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--csv", default=None, help="override the raw CSV path")
    args = parser.parse_args(argv)
    build_all(load_raw(args.csv), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
