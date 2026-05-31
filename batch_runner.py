"""
batch_runner.py

Orchestration driver for batch steganography analysis.
Loops across strategies and places, calling analyzers without GUI dependencies.
"""

from __future__ import annotations
from config import PAYLOAD_LIST

import sys
from typing import Callable
from pathlib import Path

from analysis_pipeline3 import (
    LSBAnalyzer,
    LSBCannySobelAnalyzer,
    PVDAnalyzer,
    LSBMRAnalyzer,
    RESULTS_ROOT,
)
from config import IMAGE_FOLDERS, SECRET_FILE, NUM_IMAGES
from core.strategy_registry import StrategyRegistry


# ─────────────────────────────────────────────────────────────────────────────
#  Payload list and strategy table
# ─────────────────────────────────────────────────────────────────────────────

# TODO: Read PAYLOAD_LIST from config if available; default to [1, 2, 4, 8] KB

STRATEGY_TABLE = {
    "lsbRandomSpatial": (LSBAnalyzer, "LSB Random Spatial"),
    "lsbCannySobel": (LSBCannySobelAnalyzer, "LSB Canny-Sobel"),
    "pvdSequential": (PVDAnalyzer, "PVD Sequential"),
    "lsbmr": (LSBMRAnalyzer, "LSBMR"),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Main orchestration function
# ─────────────────────────────────────────────────────────────────────────────

def run_full_batch(
    strategies: list[str],
    places: list[str],
    steps: list[str],
    should_stop: Callable[[], bool] | None = None,
    progress_cb: Callable[[str, int, int], None] | None = None,
    verify_cb: Callable[[str, str, str, dict], None] | None = None,
) -> dict:
    """
    Run a full batch analysis across multiple strategies and places.

    Args:
        strategies: List of strategy internal names (e.g. ["lsbRandomSpatial", ...])
        places: List of places (e.g. ["indoor", "outdoor"])
        steps: List of pipeline steps to run (subset of embed/verify/metrics/chi2/rs)
        should_stop: Callable returning True to stop cleanly
        progress_cb: Callable(stage_label: str, current: int, total: int)
        verify_cb: Callable(strategy, place, img_type, tally_dict)

    Returns:
        Summary dict: {(strategy, place): "completed"|"stopped"|"error", ...}
    """
    should_stop = should_stop or (lambda: False)
    progress_cb = progress_cb or (lambda *_: None)
    verify_cb = verify_cb or (lambda *_: None)

    total_units = len(strategies) * len(places)
    current_unit = 0
    summary: dict = {}

    for strategy_key in strategies:
        if should_stop():
            break

        if strategy_key not in STRATEGY_TABLE:
            print(f"WARNING: Unknown strategy '{strategy_key}' — skipping.")
            continue

        analyzer_class, display_name = STRATEGY_TABLE[strategy_key]

        for place in places:
            if should_stop():
                break

            current_unit += 1
            stage_label = f"{strategy_key} | {place}"

            try:
                # Report progress
                progress_cb(stage_label, current_unit, total_units)

                # Resolve image folders for this place
                ai_folder = str(IMAGE_FOLDERS[place]["ai"])
                nat_folder = str(IMAGE_FOLDERS[place]["natural"])

                # Instantiate analyzer
                analyzer = analyzer_class(
                    stego_root=str(Path(RESULTS_ROOT).parent / "embeddings"),
                    should_stop=should_stop,
                )

                # Get strategy object for embedding/extraction
                strategy_obj = StrategyRegistry.get_strategy(display_name)

                # Run pipeline
                result = analyzer.run_pipeline(
                    ai_folder,
                    nat_folder,
                    place,
                    PAYLOAD_LIST,
                    strategy_obj,
                    str(SECRET_FILE),
                    channel_idx=None,
                    steps=steps,
                )

                # Capture extraction tally if verify was run
                if "verify" in steps:
                    # Re-run verify to get tally (or cache it in the analyzer)
                    # For now, call it again briefly to populate the tally
                    analyzer._verify_extraction(
                        place, PAYLOAD_LIST, strategy_obj, str(SECRET_FILE),
                        ai_folder, nat_folder, channel_idx=None
                    )

                summary[(strategy_key, place)] = result
                print(f"✓ {strategy_key} | {place}: {result}")

            except Exception as exc:
                summary[(strategy_key, place)] = "error"
                print(f"✗ {strategy_key} | {place}: ERROR — {exc}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  Headless test runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Running full batch analysis (headless)...\n")

    def on_progress(label: str, current: int, total: int) -> None:
        print(f"  [{current}/{total}] {label}")

    def on_verify(strategy: str, place: str, img_type: str, tally: dict) -> None:
        print(f"    [Verify] {strategy} | {place} | {img_type}: {tally}")

    summary = run_full_batch(
        strategies=list(STRATEGY_TABLE.keys()),
        places=["indoor", "outdoor"],
        steps=["embed", "verify", "metrics", "chi2", "rs"],
        progress_cb=on_progress,
        verify_cb=on_verify,
    )

    print("\n" + "=" * 60)
    print("Summary:")
    for (strat, place), result in summary.items():
        print(f"  {strat} | {place}: {result}")
