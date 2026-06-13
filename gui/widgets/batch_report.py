"""
gui/widgets/batch_report.py

Verdict and ranking logic: aggregate metrics + RS, compute rankings, generate text.
NOTE on BPP: BPP is payload density (kb/(w*h)) and is identical across methods for
the same KB payload — treat it as "embedding density tested", not a ranking criterion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from gui.widgets.batch_plots import load_metric_df, STRATEGY_COLORS, STRATEGY_LABELS


def aggregate(strategies: list[str], places: list[str]) -> pd.DataFrame:
    """
    Aggregate metrics and RS data by (Strategy, Place, ImageType, Payload_KB).

    Returns:
        DataFrame with columns: Strategy, Place, ImageType, Payload_KB,
        PSNR, SSIM, MSE, Entropy, Estimated_Length_P, ...
    """
    metrics_df = load_metric_df(strategies, places, "metrics")
    rs_df = load_metric_df(strategies, places, "rsanalysis")

    if metrics_df.empty and rs_df.empty:
        return pd.DataFrame()

    if metrics_df.empty:
        return rs_df
    if rs_df.empty:
        return metrics_df

    # Merge on common columns
    merge_cols = ["Strategy", "Place", "ImageType", "Payload_KB"]
    agg = metrics_df.merge(rs_df, on=merge_cols, how="outer")

    return agg


def compute_rankings(agg: pd.DataFrame) -> dict:
    """
    Two separate rankings per condition:
      - imperceptibility: min-max normalized PSNR + SSIM (see compute_imperceptibility_ranking)
      - undetectability:  RS Estimated_Length_P, lower = better
    Returns {"imperceptibility": {cond: {strategy: rank}}, "undetectability": {...}}.
    """
    if agg.empty:
        return {"imperceptibility": {}, "undetectability": {}}

    rankings = {"imperceptibility": {}, "undetectability": {}}

    # Imperceptibility — computed across all conditions in ONE call
    rankings["imperceptibility"] = compute_imperceptibility_ranking(agg)

    # Undetectability — per condition, rank by RS estimated length (lower = safer)
    conditions = [f"{place}-{img_type}"
                  for place in agg["Place"].unique()
                  for img_type in agg["ImageType"].unique()]

    for cond in conditions:
        place, img_type = cond.split("-", 1)
        cond_data = agg[(agg["Place"] == place) & (agg["ImageType"] == img_type)]
        if cond_data.empty:
            continue

        undetectability_scores = {}
        for strategy in cond_data["Strategy"].unique():
            s_data = cond_data[cond_data["Strategy"] == strategy]
            avg_p_est = s_data["Estimated_Length_P"].mean() if "Estimated_Length_P" in s_data else 1.0
            undetectability_scores[strategy] = -avg_p_est   # lower P -> higher score

        undetectability_rank = {}
        for strategy, _ in sorted(undetectability_scores.items(), key=lambda x: x[1], reverse=True):
            undetectability_rank[strategy] = len(undetectability_rank) + 1

        rankings["undetectability"][cond] = undetectability_rank

    return rankings


def _minmax(values):
    """Scale to [0,1]. If all values are equal, treat as a tie (all 1.0)."""
    v = np.asarray(values, dtype=float)
    lo, hi = v.min(), v.max()
    return np.ones_like(v) if hi == lo else (v - lo) / (hi - lo)

def compute_imperceptibility_ranking(agg, w_psnr=0.5, w_ssim=0.5):
    """
    Per-condition imperceptibility ranking from min-max normalized PSNR + SSIM.
    For each (Place x ImageType): average PSNR & SSIM over payloads, scale each
    metric to [0,1] across the methods, score = w_psnr*PSNR_n + w_ssim*SSIM_n,
    rank (1 = best). Returns {condition: {strategy: rank}}.
    """
    rankings = {}
    if agg.empty:
        return rankings
    for place in agg["Place"].unique():
        for itype in agg["ImageType"].unique():
            c = agg[(agg["Place"] == place) & (agg["ImageType"] == itype)]
            if c.empty:
                continue
            strat = list(c["Strategy"].unique())
            psnr = [c[c.Strategy == s]["PSNR"].mean() for s in strat]
            ssim = [c[c.Strategy == s]["SSIM"].mean() for s in strat]
            score = w_psnr * _minmax(psnr) + w_ssim * _minmax(ssim)
            order = np.argsort(-score)
            rankings[f"{place}-{itype}"] = {strat[i]: r + 1 for r, i in enumerate(order)}
    return rankings


def fig_rank_heatmap(agg: pd.DataFrame, metric: str = "imperceptibility") -> Figure:
    """
    Heatmap: rows = strategies, columns = conditions (place-imagetype).
    Cell value = rank (1=best), colored from green (1) to red (4).
    """
    rankings = compute_rankings(agg)
    rank_dict = rankings.get(metric, {})

    if not rank_dict:
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"No {metric} rankings available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    strategies = sorted(set(s for ranks in rank_dict.values() for s in ranks.keys()))
    conditions = sorted(rank_dict.keys())
    rank_matrix = np.zeros((len(strategies), len(conditions)))

    for i, strat in enumerate(strategies):
        for j, cond in enumerate(conditions):
            rank_matrix[i, j] = rank_dict[cond].get(strat, 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(rank_matrix, cmap="RdYlGn_r", vmin=1, vmax=4)

    ax.set_xticks(range(len(conditions)))
    ax.set_yticks(range(len(strategies)))
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_yticklabels([STRATEGY_LABELS.get(s, s) for s in strategies])

    for i in range(len(strategies)):
        for j in range(len(conditions)):
            text = ax.text(j, i, int(rank_matrix[i, j]), ha="center", va="center",
                          color="black", fontweight="bold")

    fig.colorbar(im, ax=ax, label="Rank (1=best)")
    title_map = {"imperceptibility": "Imperceptibility Ranking",
                 "undetectability": "Undetectability Ranking"}
    ax.set_title(title_map.get(metric, "Ranking"), fontsize=12, fontweight="bold")

    plt.tight_layout()
    return fig


def build_verdict_text(agg: pd.DataFrame) -> str:
    """
    Generate human-readable verdict from rankings.
    Mention imperceptibility, undetectability, and tradeoff.
    """
    if agg.empty:
        return "No data available to generate verdict."

    rankings = compute_rankings(agg)
    imperceptible_ranks = rankings.get("imperceptibility", {})
    undetectable_ranks = rankings.get("undetectability", {})

    text_lines = [
        "📊 BATCH ANALYSIS VERDICT",
        "=" * 50,
        "",
    ]

    conditions = sorted(imperceptible_ranks.keys()) if imperceptible_ranks else []

    if not conditions:
        return "\n".join(text_lines + ["No conditions found."])

    for cond in conditions:
        imp_ranks = imperceptible_ranks.get(cond, {})
        und_ranks = undetectable_ranks.get(cond, {})

        if imp_ranks:
            best_imperceptible = min(imp_ranks, key=imp_ranks.get)
            text_lines.append(f"🎯 {cond.upper()}")
            text_lines.append(f"   Best imperceptibility: {STRATEGY_LABELS.get(best_imperceptible, best_imperceptible)}")

        if und_ranks:
            best_undetectable = min(und_ranks, key=und_ranks.get)
            text_lines.append(f"   Best undetectability: {STRATEGY_LABELS.get(best_undetectable, best_undetectable)}")

        if imp_ranks and und_ranks:
            if best_imperceptible != best_undetectable:
                text_lines.append(f"   ⚠ Tradeoff: imperceptibility ≠ undetectability")

        text_lines.append("")

    text_lines.append("=" * 50)
    text_lines.append("NOTE: Rankings kept separate (imperceptibility vs undetectability)")
    text_lines.append("to show method tradeoffs clearly.")

    return "\n".join(text_lines)
