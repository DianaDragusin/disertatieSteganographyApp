"""
gui/batch_results_index.py

Disk scanner to report which results already exist and audit pass-rates.
Each (method, place, channel) is a separate row.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List

from analysis_pipeline3 import RESULTS_ROOT
from config import IMAGE_FOLDERS


def get_channels_for_method(method: str) -> list[str]:
    """
    Determine which channels apply to a method.
    
    LSBMR is per-channel; all others are single-channel.
    Returns: ["blue", "green", "red"] for LSBMR, ["default"] for others.
    """
    if method == "lsbmr":
        return ["blue", "green", "red"]
    return ["default"]


def scan_results(strategies: list[str], places: list[str]) -> list[dict]:
    """
    Scan disk to report CSV existence and extraction audit pass-rates.
    One dict per (method, place, channel) combination.

    Returns:
        [
            {
                "method": str,
                "place": str,
                "channel": str,  # "default" | "blue" | "green" | "red"
                "display": str,  # method if channel=="default" else f"{method}_{channel}"
                "has_metrics": bool,
                "has_chi2": bool,
                "has_rs": bool,
                "has_audit": bool,
                "ai_pass": int,
                "ai_total": int,
                "nat_pass": int,
                "nat_total": int,
            },
            ...
        ]
    """
    results = []

    for strategy in strategies:
        channels = get_channels_for_method(strategy)
        for place in places:
            for channel in channels:
                display = strategy if channel == "default" else f"{strategy}_{channel}"

                try:
                    # Check which CSVs exist for this specific channel
                    has_metrics = _csv_exists(strategy, place, "metrics", channel)
                    has_chi2 = _csv_exists(strategy, place, "chisquare", channel)
                    has_rs = _csv_exists(strategy, place, "rsanalysis", channel)
                    has_audit = _audit_file_exists(strategy, place, channel)

                    # Load extraction audit data for this channel
                    audit_detail = load_audit_detail(strategy, place, channel) if has_audit else []
                    ai_pass, ai_total = 0, 0
                    nat_pass, nat_total = 0, 0

                    for payload_info in audit_detail:
                        ai_pass += payload_info.get("ai_pass", 0)
                        ai_total += payload_info.get("ai_total", 0)
                        nat_pass += payload_info.get("nat_pass", 0)
                        nat_total += payload_info.get("nat_total", 0)

                    results.append({
                        "method": strategy,
                        "place": place,
                        "channel": channel,
                        "display": display,
                        "has_metrics": has_metrics,
                        "has_chi2": has_chi2,
                        "has_rs": has_rs,
                        "has_audit": has_audit,
                        "ai_pass": ai_pass,
                        "ai_total": ai_total,
                        "nat_pass": nat_pass,
                        "nat_total": nat_total,
                    })
                except Exception as e:
                    # Add a blank row with status indicators for this channel
                    results.append({
                        "method": strategy,
                        "place": place,
                        "channel": channel,
                        "display": display,
                        "has_metrics": False,
                        "has_chi2": False,
                        "has_rs": False,
                        "has_audit": False,
                        "ai_pass": 0,
                        "ai_total": 0,
                        "nat_pass": 0,
                        "nat_total": 0,
                    })

    return results


def load_audit_detail(method: str, place: str, channel: str) -> list[dict]:
    """
    Read ONE extraction audit CSV for a specific (method, place, channel).

    Returns:
        [
            {"payload": int, "ai_pass": int, "ai_total": int, "nat_pass": int, "nat_total": int},
            ...
        ]
        sorted by payload ascending. Empty list if file not found or is malformed.
    """
    csv_dir = Path(RESULTS_ROOT) / place / "csv"
    if not csv_dir.exists():
        return []

    # Exact filename for this channel
    audit_path = csv_dir / f"{method}_{place}_{channel}_extraction_audit.csv"
    if not audit_path.exists():
        return []

    payload_stats: Dict[int, dict] = {}
    try:
        with open(audit_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    payload = int(row.get("payload", 0))
                    img_type = row.get("type", "").strip().lower()  # "ai" or "natural"
                    succeeded = row.get("succeeded", "").strip().lower()  # "true" or "false"

                    if payload not in payload_stats:
                        payload_stats[payload] = {
                            "ai_pass": 0, "ai_total": 0,
                            "nat_pass": 0, "nat_total": 0
                        }

                    if img_type == "ai":
                        payload_stats[payload]["ai_total"] += 1
                        if succeeded == "true":
                            payload_stats[payload]["ai_pass"] += 1
                    elif img_type == "natural":
                        payload_stats[payload]["nat_total"] += 1
                        if succeeded == "true":
                            payload_stats[payload]["nat_pass"] += 1
                except Exception:
                    continue  # Skip malformed rows
    except Exception:
        return []  # File unreadable

    # Return as sorted list
    result = [
        {"payload": kb, **stats}
        for kb, stats in sorted(payload_stats.items())
    ]
    return result


def _csv_exists(strategy: str, place: str, suffix: str, channel: str = "default") -> bool:
    """Check if a CSV with the given suffix exists for this strategy/place/channel."""
    csv_dir = Path(RESULTS_ROOT) / place / "csv"
    if not csv_dir.exists():
        return False

    expected_name = f"{strategy}_{place}_{channel}_{suffix}.csv"
    return (csv_dir / expected_name).exists()


def _audit_file_exists(strategy: str, place: str, channel: str) -> bool:
    """Check if an extraction audit CSV exists for this strategy/place/channel."""
    csv_dir = Path(RESULTS_ROOT) / place / "csv"
    if not csv_dir.exists():
        return False

    audit_path = csv_dir / f"{strategy}_{place}_{channel}_extraction_audit.csv"
    return audit_path.exists()


def get_status_icon(exists: bool) -> str:
    """Return a simple status indicator."""
    return "✓" if exists else "✗"


def format_audit_pass_rate(ai_pass: int, ai_total: int, nat_pass: int,
                           nat_total: int) -> tuple[str, str]:
    """
    Format extraction pass rates for display.

    Returns: (ai_display, nat_display)
        e.g., ("20/20 (100%)", "14/20 (70%)")
    """
    ai_pct = (ai_pass / ai_total * 100) if ai_total > 0 else 0
    nat_pct = (nat_pass / nat_total * 100) if nat_total > 0 else 0

    ai_str = f"{ai_pass}/{ai_total} ({ai_pct:.0f}%)" if ai_total > 0 else "—"
    nat_str = f"{nat_pass}/{nat_total} ({nat_pct:.0f}%)" if nat_total > 0 else "—"

    return ai_str, nat_str
