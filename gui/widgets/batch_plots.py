"""
gui/widgets/batch_plots.py

Plot generation from batch analysis CSVs.

Aggregation rules (READ BEFORE EDITING):
    - lsbRandomSpatial / lsbCannySobel / pvdSequential are run with channel_idx=None
      and produce ONE CSV per place tagged Channel="default" (or whatever the pipeline
      writes for the no-channel case). Use the row as-is.
    - lsbmr is run per-channel and produces THREE CSVs per place tagged
      Channel="blue"/"green"/"red". For the main comparison plots, collapse the three
      channel rows into ONE row per (Place, ImageFile, ImageType, Payload_KB) using
      ITU-R BT.601 luminance weights:
              Y = 0.299*R + 0.587*G + 0.114*B
      so the four methods sit on a single fair axis.
    - The channel-breakdown figure deliberately KEEPS the three lsbmr rows separate.

The four 2x2 comparison figures (Quality / Chi2 / RS / Pair-diff) follow the same
layout: rows = (AI, Natural), columns = (Indoor, Outdoor), four method lines per
subplot, ONE shared figure-level legend, shared axes within sensible groups.

NOTE: fig_pairdiff_2x2 is now a SINGLE-IMAGE showcase (one image index per panel,
all methods overlaid, the cover image shown beside each plot, LSBMR = blue channel,
no caching). The old multi-image capped version was replaced.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import cv2
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from analysis_pipeline3 import RESULTS_ROOT


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_COLORS = {
    "lsbRandomSpatial": "#1f77b4",  # blue
    "lsbCannySobel":    "#ff7f0e",  # orange
    "pvdSequential":    "#2ca02c",  # green
    "lsbmr":            "#d62728",  # red
}

STRATEGY_LABELS = {
    "lsbRandomSpatial": "LSB Random Spatial",
    "lsbCannySobel":    "LSB Canny-Sobel",
    "pvdSequential":    "PVD Sequential",
    "lsbmr":            "LSBMR",
}

# ITU-R BT.601 luminance weights (Y = 0.299*R + 0.587*G + 0.114*B)
LUMINANCE_WEIGHTS = {"red": 0.299, "green": 0.587, "blue": 0.114}

# Numeric columns to luminance-collapse for LSBMR, per suffix.
LSBMR_COLLAPSE_COLS = {
    "metrics":    ["PSNR", "SSIM", "MSE", "Entropy", "BPP"],
    "chisquare":  ["Chi2_Statistic", "P_Value"],
    "rsanalysis": ["Estimated_Length_P", "RM", "SM", "RnM", "SnM"],
}

# Embedding folder layout (mirrors analysis_pipeline3._stego_folder).
EMBEDDINGS_ROOT = Path(RESULTS_ROOT).parent / "embeddings"
CHANNEL_FOLDER_DEFAULT = "channel_default"
CHANNEL_FOLDER_LSBMR = "channel_blue"  # representative channel for pair-diff


# ─────────────────────────────────────────────────────────────────────────────
#  CSV loading + LSBMR luminance collapse
# ─────────────────────────────────────────────────────────────────────────────

def _read_strategy_csvs(strategy: str, place: str, suffix: str) -> list[pd.DataFrame]:
    """
    Read every CSV matching <strategy>_<place>_*_<suffix>.csv under results/<place>/csv/.
    Returns a list of DataFrames (one per file found).
    """
    csv_dir = Path(RESULTS_ROOT) / place / "csv"
    if not csv_dir.exists():
        return []

    out: list[pd.DataFrame] = []
    prefix = f"{strategy}_{place}_"
    target = f"_{suffix}.csv"
    for fname in csv_dir.iterdir():
        name = fname.name
        if name.startswith(prefix) and name.endswith(target):
            try:
                out.append(pd.read_csv(fname))
            except Exception as exc:  # noqa: BLE001
                print(f"[batch_plots] Failed to read {fname}: {exc}")
    return out


def _collapse_lsbmr_luminance(df_lsbmr: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Collapse LSBMR's three per-channel rows into one luminance-weighted row per
    (Place, ImageFile, ImageType, Payload_KB).

    If only 1 or 2 channels are present for a group, fall back to plain mean
    of the available channels and log a warning once.
    """
    if df_lsbmr.empty:
        return df_lsbmr

    if "Channel" not in df_lsbmr.columns:
        # Nothing to collapse — return as-is.
        return df_lsbmr

    # Some CSVs may already be collapsed; only operate on rows whose Channel is one
    # of blue/green/red. Anything else (e.g. "default") passes through.
    is_channel = df_lsbmr["Channel"].isin(LUMINANCE_WEIGHTS.keys())
    df_chan = df_lsbmr[is_channel].copy()
    df_other = df_lsbmr[~is_channel].copy()

    if df_chan.empty:
        return df_other

    group_cols = [c for c in ["Place", "ImageFile", "ImageType", "Payload_KB"]
                  if c in df_chan.columns]
    if not group_cols:
        return df_lsbmr  # cannot collapse without grouping columns

    value_cols = [c for c in LSBMR_COLLAPSE_COLS.get(suffix, []) if c in df_chan.columns]
    if not value_cols:
        # Nothing numeric we know how to weight — average everything numeric.
        value_cols = df_chan.select_dtypes(include="number").columns.tolist()
        value_cols = [c for c in value_cols if c != "Payload_KB"]

    incomplete_groups = 0
    collapsed_rows: list[dict] = []

    for keys, group in df_chan.groupby(group_cols, dropna=False):
        present = {ch: group[group["Channel"] == ch] for ch in LUMINANCE_WEIGHTS}
        present_channels = [ch for ch, sub in present.items() if not sub.empty]

        if len(present_channels) == 3:
            # Full luminance collapse
            row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            for col in value_cols:
                weighted = 0.0
                for ch in present_channels:
                    weighted += float(present[ch][col].mean()) * LUMINANCE_WEIGHTS[ch]
                row[col] = weighted
        else:
            # Fallback: plain mean of whatever's present
            incomplete_groups += 1
            row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            for col in value_cols:
                vals = [float(present[ch][col].mean()) for ch in present_channels]
                row[col] = float(np.mean(vals)) if vals else np.nan

        row["Strategy"] = "lsbmr"
        row["Channel"] = "luminance"
        collapsed_rows.append(row)

    if incomplete_groups:
        print(f"[batch_plots] WARNING: {incomplete_groups} LSBMR groups had <3 "
              f"channels for suffix={suffix}; used plain mean of available channels.")

    df_collapsed = pd.DataFrame(collapsed_rows)
    return pd.concat([df_other, df_collapsed], ignore_index=True)


def load_metric_df(
    strategies: list[str],
    places: list[str],
    suffix: str,
    collapse_lsbmr: bool = True,
) -> pd.DataFrame:
    """
    Load CSVs for the given strategies/places/suffix. By default, LSBMR's three
    per-channel rows are luminance-collapsed; pass collapse_lsbmr=False to keep
    them (needed for the channel-breakdown figure).
    """
    frames: list[pd.DataFrame] = []

    for strategy in strategies:
        per_strategy: list[pd.DataFrame] = []
        for place in places:
            per_strategy.extend(_read_strategy_csvs(strategy, place, suffix))

        if not per_strategy:
            continue

        df = pd.concat(per_strategy, ignore_index=True)

        # Ensure Strategy column is set (some older CSVs may have it; force it)
        df["Strategy"] = strategy

        if strategy == "lsbmr" and collapse_lsbmr:
            df = _collapse_lsbmr_luminance(df, suffix)

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Final aggregation: mean across whatever rows remain per group (this averages
    # over distinct ImageFiles so one mean point per condition/payload).
    group_cols = [c for c in ["Strategy", "Place", "ImageType", "Payload_KB"]
                  if c in combined.columns]
    if group_cols:
        combined = combined.groupby(group_cols, as_index=False).median(numeric_only=True)

    return combined


# ─────────────────────────────────────────────────────────────────────────────
#  Shared plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _subplot_coords(place: str, img_type: str) -> tuple[int, int]:
    """(row, col) in the 2x2 grid: rows = AI/Natural, cols = Indoor/Outdoor."""
    row = 0 if img_type == "AI" else 1
    col = 0 if place == "indoor" else 1
    return row, col


def _strategy_present(df: pd.DataFrame, strategy: str) -> bool:
    """True if any rows for this strategy exist in df."""
    if df.empty or "Strategy" not in df.columns:
        return False
    return bool((df["Strategy"] == strategy).any())


def _plot_methods_on_ax(ax, df: pd.DataFrame, x_col: str, y_col: str,
                        strategies: list[str]) -> None:
    """Plot one line per strategy on a single axis. Skips strategies absent in df."""
    if df.empty or y_col not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    for strategy in strategies:
        sub = df[df["Strategy"] == strategy].sort_values(x_col)
        if sub.empty:
            continue
        ax.plot(
            sub[x_col],
            sub[y_col],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=STRATEGY_COLORS.get(strategy, "#000000"),
            label=STRATEGY_LABELS.get(strategy, strategy),
        )


def _legend_entries_for_strategies(strategies: list[str],
                                   present: set[str]) -> tuple[list, list]:
    """Build figure-legend handles/labels with '(no data)' suffix where missing."""
    from matplotlib.lines import Line2D
    handles, labels = [], []
    for s in strategies:
        color = STRATEGY_COLORS.get(s, "#000000")
        base = STRATEGY_LABELS.get(s, s)
        label = base if s in present else f"{base} (no data)"
        handles.append(Line2D([0], [0], color=color, linewidth=2.5, marker="o"))
        labels.append(label)
    return handles, labels


def _finalize_2x2(fig, axes, suptitle: str, xlabel: str, ylabel: str,
                  strategies: list[str], present: set[str]) -> None:
    """Apply consistent titles, axis labels, legend, and layout to a 2x2 figure."""
    titles = {
        (0, 0): "Indoor — AI",
        (0, 1): "Outdoor — AI",
        (1, 0): "Indoor — Natural",
        (1, 1): "Outdoor — Natural",
    }
    for (r, c), title in titles.items():
        ax = axes[r, c]
        ax.set_title(title, fontsize=11, pad=8)
        ax.grid(True, alpha=0.3)
        if r == 1:
            ax.set_xlabel(xlabel)
        if c == 0:
            ax.set_ylabel(ylabel)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    handles, labels = _legend_entries_for_strategies(strategies, present)
    fig.legend(handles, labels, loc="lower center", ncol=len(strategies),
               bbox_to_anchor=(0.5, 0.0), frameon=True)
    # Leave room for suptitle on top and the legend on the bottom.
    fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.94])


def _empty_figure(title: str, message: str = "No data available — run the pipeline first") -> Figure:
    fig = Figure(figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  2x2 comparison figures
# ─────────────────────────────────────────────────────────────────────────────

def fig_quality_2x2(strategies: list[str], places: list[str],
                    metric: str = "PSNR") -> Figure:
    """Quality metric (PSNR/SSIM/MSE/Entropy) vs Payload, 2x2 grid, 4 methods."""
    df = load_metric_df(strategies, places, "metrics")
    if df.empty or metric not in df.columns:
        return _empty_figure(f"Quality — {metric}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    # Share Y within each row only — AI and Natural have different ranges.
    for row in (0, 1):
        axes[row, 0].sharey(axes[row, 1])

    for place in places:
        for img_type in ("AI", "Natural"):
            r, c = _subplot_coords(place, img_type)
            sub = df[(df["Place"] == place) & (df["ImageType"] == img_type)]
            _plot_methods_on_ax(axes[r, c], sub, "Payload_KB", metric, strategies)

    present = set(df["Strategy"].unique()) if not df.empty else set()
    _finalize_2x2(fig, axes,
                  suptitle=f"Quality Metrics — {metric} vs Payload",
                  xlabel="Payload (KB)",
                  ylabel=metric,
                  strategies=strategies,
                  present=present)
    return fig


def fig_chi2_2x2(strategies: list[str], places: list[str]) -> Figure:
    """Chi-square statistic vs payload, 2x2 grid."""
    df = load_metric_df(strategies, places, "chisquare")
    if df.empty or "Chi2_Statistic" not in df.columns:
        return _empty_figure("Chi-square Steganalysis")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    # Don't share Y across rows — Indoor vs Outdoor can differ by an order of magnitude.

    for place in places:
        for img_type in ("AI", "Natural"):
            r, c = _subplot_coords(place, img_type)
            sub = df[(df["Place"] == place) & (df["ImageType"] == img_type)]
            _plot_methods_on_ax(axes[r, c], sub, "Payload_KB", "Chi2_Statistic", strategies)

    present = set(df["Strategy"].unique()) if not df.empty else set()
    _finalize_2x2(fig, axes,
                  suptitle="Chi-square Steganalysis — Statistic vs Payload",
                  xlabel="Payload (KB)",
                  ylabel="Chi² Statistic",
                  strategies=strategies,
                  present=present)
    return fig


def fig_rs_2x2(strategies: list[str], places: list[str]) -> Figure:
    """RS analysis Estimated_Length_P vs payload, 2x2 grid.

    Y axes are independent per subplot so very small panels (e.g. indoor-natural)
    stay readable.
    """
    df = load_metric_df(strategies, places, "rsanalysis")
    if df.empty or "Estimated_Length_P" not in df.columns:
        return _empty_figure("RS Steganalysis")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    # Intentionally no sharey — values can differ by 100x between panels.

    for place in places:
        for img_type in ("AI", "Natural"):
            r, c = _subplot_coords(place, img_type)
            sub = df[(df["Place"] == place) & (df["ImageType"] == img_type)]
            _plot_methods_on_ax(axes[r, c], sub, "Payload_KB",
                                "Estimated_Length_P", strategies)

    present = set(df["Strategy"].unique()) if not df.empty else set()
    _finalize_2x2(fig, axes,
                  suptitle="RS Steganalysis — Estimated Embedding Length",
                  xlabel="Payload (KB)",
                  ylabel="Estimated P̂  (↓ = stealthier)",
                  strategies=strategies,
                  present=present)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Pair-difference SHOWCASE (single image per panel, all methods, NO caching)
#
#  fig_pairdiff_2x2 now uses ONE image index across the four panels (= 4 images
#  total), draws all methods as (stego - cover) pair-difference densities, and shows
#  the cover image beside each plot. LSBMR uses its BLUE channel (via
#  _stego_folder_for_pair_diff). Reads only a handful of files -> no caching needed.
# ─────────────────────────────────────────────────────────────────────────────

def _stego_folder_for_pair_diff(strategy: str, place: str,
                                img_type_lower: str) -> Optional[Path]:
    """
    Resolve the stego folder for the pair-diff plot.
    LSB / Canny / PVD live under channel_default; LSBMR uses channel_blue
    as a representative slice.
    """
    base = EMBEDDINGS_ROOT / f"{strategy}_embeddings" / place
    if strategy == "lsbmr":
        candidate = base / CHANNEL_FOLDER_LSBMR / img_type_lower
    else:
        candidate = base / CHANNEL_FOLDER_DEFAULT / img_type_lower
    return candidate if candidate.exists() else None


def _luminance_diffs(img_bgr_or_gray: np.ndarray) -> np.ndarray:
    """Horizontal pair-difference of the luminance plane (|left - right|)."""
    if img_bgr_or_gray.ndim == 2:
        gray = img_bgr_or_gray
    else:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.int16)
    return np.abs(gray[:, 1:] - gray[:, :-1]).flatten()


def _pairdiff_density_from_path(path: Path, bins: np.ndarray) -> Optional[np.ndarray]:
    """
    Read one image as grayscale and return its luminance pair-difference DENSITY
    (matches np.histogram(..., density=True)). Returns None if unreadable/empty.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # grayscale: fast + no BGRA crash
    if img is None:
        return None
    diffs = _luminance_diffs(img)              # img is 2-D here, so no cvtColor happens
    counts, _ = np.histogram(diffs, bins=bins)
    total = counts.sum()
    if total == 0:
        return None
    bin_width = bins[1] - bins[0]
    return counts / (total * bin_width)


def _pairdiff_counts_from_path(path: Path, bins: np.ndarray) -> Optional[np.ndarray]:
    """
    Read one image as grayscale and return RAW luminance pair-difference COUNTS, so
    multiple payloads of the same image can be summed before normalising. None if
    the file is unreadable.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    diffs = _luminance_diffs(img)
    counts, _ = np.histogram(diffs, bins=bins)
    return counts.astype(np.float64)


def fig_pairdiff_2x2(strategies: list[str], places: list[str],
                     image_name: str = "1.png", payload_kb: Optional[int] = None,
                     cover_folders: Optional[dict] = None) -> Figure:
    """
    Single-image pair-difference showcase (qualitative; nothing is cached).

    One cover image per panel (place x image-type). Every method is overlaid as the
    (stego - cover) luminance pair-difference density, and the cover image is shown to
    the right of each plot. LSBMR uses its BLUE channel via _stego_folder_for_pair_diff.

    payload_kb:
        None (default) -> POOL all payload folders for this image into each method's
                          curve (1 KB ... 500 KB summed as raw counts, normalised once).
        an int         -> use only that single payload folder.

    Backward-compatible: callers that do fig_pairdiff_2x2(strategies, places) still work.
    """
    if cover_folders is None:
        try:
            from config import IMAGE_FOLDERS
            cover_folders = {}
            for place in places:
                if place in IMAGE_FOLDERS:
                    cover_folders[(place, "ai")] = Path(IMAGE_FOLDERS[place]["ai"])
                    cover_folders[(place, "natural")] = Path(IMAGE_FOLDERS[place]["natural"])
        except Exception as exc:  # noqa: BLE001
            return _empty_figure("Pair-Difference Showcase",
                                 f"Cannot resolve cover folders: {exc}")

    # Per-pixel difference bins: d = stego - cover, integer-centred and symmetric.
    diff_bins = np.arange(-32.5, 33.5, 1.0)
    diff_centers = np.arange(-32, 33)

    # Panels: image-type major, place minor (up to 4 rows; adapts if fewer places).
    panels = [(place, img_type)
              for img_type in ("AI", "Natural")
              for place in places]
    if not panels:
        return _empty_figure("Pair-Difference Showcase", "No places selected.")

    n_rows = len(panels)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 2.6 * n_rows),
                             gridspec_kw={"width_ratios": [3, 1]},
                             squeeze=False)
    present: set[str] = set()

    for row, (place, img_type) in enumerate(panels):
        ax_plot = axes[row, 0]
        ax_img = axes[row, 1]
        ax_img.axis("off")
        img_type_lower = img_type.lower()
        ax_plot.set_title(f"{place.capitalize()} — {img_type}", fontsize=11)
        ax_plot.grid(True, alpha=0.3, which="both")
        ax_plot.set_yscale("log")          # LSB footprints are tiny; log makes them visible
        ax_plot.set_ylabel("Density of (stego − cover)  [log]")
        if row == n_rows - 1:
            ax_plot.set_xlabel("Per-pixel change (stego − cover)")

        cover_folder = cover_folders.get((place, img_type_lower))
        cover_path = (cover_folder / image_name) if cover_folder else None
        if cover_path is None or not cover_path.exists():
            ax_plot.text(0.5, 0.5, f"cover {image_name} not found", ha="center",
                         va="center", transform=ax_plot.transAxes, color="gray")
            continue

        cover_gray = cv2.imread(str(cover_path), cv2.IMREAD_GRAYSCALE)
        if cover_gray is None:
            ax_plot.text(0.5, 0.5, "cover unreadable", ha="center", va="center",
                         transform=ax_plot.transAxes, color="gray")
            continue
        cover_gray = cover_gray.astype(np.int16)

        # Cover image (color) beside the plot for context.
        cov_color = cv2.imread(str(cover_path), cv2.IMREAD_COLOR)
        if cov_color is not None:
            ax_img.imshow(cv2.cvtColor(cov_color, cv2.COLOR_BGR2RGB))
            ax_img.set_title(image_name, fontsize=8)

        # One line per method: histogram of the PER-PIXEL change (stego - cover),
        # pooled across ALL payload folders for this image. This directly shows each
        # method's embedding footprint (LSB methods => ±1; PVD => much wider).
        for strategy in strategies:
            stego_folder = _stego_folder_for_pair_diff(strategy, place, img_type_lower)
            if stego_folder is None:
                continue

            kb_dirs = sorted((p for p in stego_folder.iterdir()
                              if p.is_dir() and p.name.isdigit()),
                             key=lambda p: int(p.name))

            pooled = np.zeros(len(diff_bins) - 1, dtype=np.float64)
            for kb_dir in kb_dirs:
                stego = cv2.imread(str(kb_dir / image_name), cv2.IMREAD_GRAYSCALE)
                if stego is None or stego.shape != cover_gray.shape:
                    continue
                d = stego.astype(np.int16) - cover_gray
                counts, _ = np.histogram(d, bins=diff_bins)
                pooled += counts

            total = pooled.sum()
            if total == 0:
                continue
            density = pooled / total                       # bin width = 1
            density = np.where(density > 0, density, np.nan)  # log skips empty bins
            ax_plot.plot(diff_centers, density,
                         color=STRATEGY_COLORS.get(strategy, "#000000"),
                         linewidth=1.8,
                         label=STRATEGY_LABELS.get(strategy, strategy))
            present.add(strategy)

    # Shared figure legend (method order preserved).
    if present:
        from matplotlib.lines import Line2D
        ordered = [s for s in strategies if s in present]
        handles = [Line2D([0], [0], color=STRATEGY_COLORS.get(s, "#000000"), linewidth=2.5)
                   for s in ordered]
        labels = [STRATEGY_LABELS.get(s, s) for s in ordered]
        fig.legend(handles, labels, loc="lower center", ncol=len(ordered),
                   bbox_to_anchor=(0.5, 0.0), frameon=True)

    fig.suptitle(f"Per-pixel Difference (stego − cover) — {image_name} @ all payloads pooled "
                 f"(LSBMR = blue channel)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.96])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: LSBMR per-channel breakdown
# ─────────────────────────────────────────────────────────────────────────────

def fig_channel_breakdown_2x2(strategies: list[str], places: list[str],
                              metric: str = "PSNR") -> Figure:
    """
    LSBMR per-channel PSNR (blue/green/red), averaged across ALL payloads.
    2x2 layout: rows = (AI, Natural), cols = (Indoor, Outdoor).

    Only LSBMR appears — the other methods were run with channel_idx=None and have
    no per-channel data to compare. This is noted on the figure.
    """
    # Load RAW (uncollapsed) LSBMR data — we deliberately want the three channel rows.
    df = load_metric_df(["lsbmr"], places, "metrics", collapse_lsbmr=False)
    if df.empty or metric not in df.columns or "Channel" not in df.columns:
        return _empty_figure(
            "LSBMR Per-Channel Sensitivity",
            "No LSBMR per-channel metrics CSVs found.",
        )

    df = df[df["Channel"].isin(LUMINANCE_WEIGHTS.keys())]
    if df.empty:
        return _empty_figure(
            "LSBMR Per-Channel Sensitivity",
            "LSBMR CSVs found but no blue/green/red rows.",
        )

    channel_order = ["blue", "green", "red"]
    channel_colors = {"blue": "#1f77b4", "green": "#2ca02c", "red": "#d62728"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)

    for place in places:
        for img_type in ("AI", "Natural"):
            r, c = _subplot_coords(place, img_type)
            ax = axes[r, c]
            sub = df[(df["Place"] == place) & (df["ImageType"] == img_type)]

            means, stds = [], []
            for ch in channel_order:
                ch_rows = sub[sub["Channel"] == ch]
                if ch_rows.empty:
                    means.append(np.nan)
                    stds.append(0.0)
                else:
                    means.append(float(ch_rows[metric].mean()))
                    stds.append(float(ch_rows[metric].std(ddof=0) or 0.0))

            x_pos = np.arange(len(channel_order))
            bar_colors = [channel_colors[ch] for ch in channel_order]
            ax.bar(x_pos, means, yerr=stds, color=bar_colors,
                   alpha=0.85, capsize=4, edgecolor="black", linewidth=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([ch.capitalize() for ch in channel_order])
            ax.set_title(f"{place.capitalize()} — {img_type}", fontsize=11, pad=8)
            ax.grid(True, alpha=0.3, axis="y")
            if c == 0:
                ax.set_ylabel(f"Mean {metric} (dB)" if metric == "PSNR"
                              else f"Mean {metric}")

    fig.suptitle(f"LSBMR Per-Channel {metric} Sensitivity (averaged across payloads)",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.02,
             "Only LSBMR shown — other methods were run with channel_idx=None "
             "and have no per-channel data in this experiment.",
             ha="center", fontsize=9, style="italic", color="gray")
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.94])
    return fig