"""
analysis_pipeline3.py

Batch steganography analysis pipeline.

Conservative refactor of the original StegoAnalyzer:
  * Repeated scaffolding (channel labels, result paths, stego folders,
    CSV writing, the AI/Natural x payload loops) is factored into helpers.
  * A cooperative `should_stop` hook lets a GUI worker interrupt the
    pipeline cleanly between units of work (per category / per payload),
    without modifying the inner algorithms.
  * All PLOTTING has been removed — plots now live in the GUI layer
    (gui/widgets/batch_plots.py), which reads the CSVs written here.
  * Channel-correlation analysis has been dropped per scope trim.

Public surface kept compatible with the original:
    StegoAnalyzer.run_pipeline(...)
    LSBAnalyzer / LSBCannySobelAnalyzer / PVDAnalyzer / LSBMRAnalyzer

Steps retained: embed, verify extraction, compute metrics + CSV,
chi-square steganalysis, RS steganalysis.
"""

from __future__ import annotations

import os
import csv
from abc import ABC
from typing import Callable, Iterable

import cv2
import numpy as np

from ssimAndPsnr import (
    calculate_psnr,
    calculate_ssim,
    calculate_mse,
    calculate_entropy,
    calculate_bpp,
)
from steganalysis.chi_square import calculate_chi2
from steganalysis.rs_analysis import rs_analysis
from stego_processor import process_stego_dataset


# ─────────────────────────────────────────────────────────────────────────────
#  Paths / constants — edit only here
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_ROOT = r"C:\Users\Diana\Desktop\disertatieSteganographyApp\results"

CHANNEL_LABELS = {0: "blue", 1: "green", 2: "red"}
CHANNEL_FOLDERS = {0: "channel_blue", 1: "channel_green", 2: "channel_red"}
IMG_EXTS = (".png", ".jpg")


def _results_path(place: str, *parts: str) -> str:
    """Build a results sub-path that always includes the place folder."""
    return os.path.join(RESULTS_ROOT, place, *parts)


def _channel_label(channel_idx: int | None) -> str:
    """Human-readable channel tag used in filenames ('blue'/'green'/'red'/'default')."""
    return CHANNEL_LABELS.get(channel_idx, "default")


def _list_images(folder: str) -> list[str]:
    """Sorted list of image files in a folder."""
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


def _write_csv(out_path: str, fieldnames: list[str], rows: list[dict], tag: str) -> None:
    """Write rows to CSV; print a consistent status line. No-op on empty rows."""
    if not rows:
        print(f"  [{tag}] No rows generated — nothing written.")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [{tag}] Saved -> {out_path}  ({len(rows)} rows)")


class StopRequested(Exception):
    """Raised internally when the stop hook signals an interruption."""


# ─────────────────────────────────────────────────────────────────────────────
#  Base class
# ─────────────────────────────────────────────────────────────────────────────
class StegoAnalyzer(ABC):
    """
    Template for batch analysis of one embedding strategy.

    The GUI may pass a `should_stop` callable; it is polled between units of
    work. When it returns True, the current step exits cleanly at the next
    safe boundary (the algorithms themselves are never interrupted mid-call).
    """

    def __init__(
        self,
        strategy_name: str,
        stego_root: str = "embeddings",
        should_stop: Callable[[], bool] | None = None,
    ):
        self.strategy_name = strategy_name
        self.stego_root = stego_root
        self._should_stop = should_stop or (lambda: False)

    # ── stop hook ─────────────────────────────────────────────────────────────
    @property
    def is_pvd(self) -> bool:
        return self.strategy_name == "pvdSequential"

    def set_stop_hook(self, should_stop: Callable[[], bool]) -> None:
        """Allow a worker to (re)attach its stop predicate after construction."""
        self._should_stop = should_stop or (lambda: False)

    def _checkpoint(self) -> None:
        """Raise StopRequested if the GUI asked to stop. Call at safe boundaries."""
        if self._should_stop():
            raise StopRequested()

    # ── path helpers ──────────────────────────────────────────────────────────
    def _stego_folder(self, place: str, type_folder: str, channel_idx: int | None = None) -> str:
        """Path to embedded images for a given place, channel, and type."""
        channel_folder = CHANNEL_FOLDERS.get(channel_idx, "channel_default")
        return os.path.join(
            self.stego_root,
            f"{self.strategy_name}_embeddings",
            place,
            channel_folder,
            type_folder,
        )

    def _csv_path(self, place: str, channel_idx: int | None, suffix: str) -> str:
        """results/<place>/csv/<strategy>_<place>_<clabel>_<suffix>.csv"""
        c = _channel_label(channel_idx)
        return _results_path(place, "csv", f"{self.strategy_name}_{place}_{c}_{suffix}.csv")

    # ── data iteration helper ───────────────────────────────────────────────
    def _iter_categories(
        self, folder_ai: str, folder_nat: str
    ) -> list[tuple[str, list[str], str]]:
        """Return [(img_type, files, cover_folder), ...] for AI and Natural."""
        return [
            ("AI", _list_images(folder_ai), folder_ai),
            ("Natural", _list_images(folder_nat), folder_nat),
        ]

    def _read_cover(self, path: str):
        """Read a cover image, converting to grayscale for PVD."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if self.is_pvd else img

    def _read_stego(self, path: str):
        """Read a stego image with the flag appropriate to the strategy."""
        if not os.path.exists(path):
            return None
        flag = cv2.IMREAD_GRAYSCALE if self.is_pvd else cv2.IMREAD_COLOR
        return cv2.imread(path, flag)

    # ── template method ────────────────────────────────────────────────────────
    def run_pipeline(
        self,
        folder_ai: str,
        folder_nat: str,
        place: str,
        payload_list: Iterable[int],
        strategy_obj,
        secret_file: str,
        channel_idx: int | None = None,
        steps: Iterable[str] | None = None,
    ) -> str:
        """
        Run the analysis pipeline for one place.

        `steps` selects which stages run; default runs the full retained set.
        Returns "completed" or "stopped".

        Stages (in order):
            "embed"   -> _embed
            "verify"  -> _verify_extraction
            "metrics" -> _compute_metrics + _save_csv
            "chi2"    -> _run_chi_square_analysis
            "rs"      -> _run_rs_steganalysis
            "audit"   -> _run_extraction_audit
        """
        steps = list(steps) if steps else ["embed", "verify", "metrics", "chi2", "rs", "audit"]
        payload_list = sorted(payload_list)

        # Pre-load image file lists for audit step
        ai_files = _list_images(folder_ai)
        nat_files = _list_images(folder_nat)

        try:
            if "embed" in steps:
                self._embed(folder_ai, folder_nat, place, payload_list,
                            strategy_obj, secret_file, channel_idx)
                self._checkpoint()

            if "verify" in steps:
                self._verify_extraction(place, payload_list, strategy_obj,
                                        secret_file, folder_ai, folder_nat, channel_idx)
                self._checkpoint()

            if "metrics" in steps:
                metrics = self._compute_metrics(folder_ai, folder_nat, place,
                                                payload_list, channel_idx)
                self._save_csv(metrics, folder_ai, folder_nat, place,
                               payload_list, channel_idx)
                self._checkpoint()

            if "chi2" in steps:
                self._run_chi_square_analysis(folder_ai, folder_nat, place,
                                              payload_list, channel_idx)
                self._checkpoint()

            if "rs" in steps:
                self._run_rs_steganalysis(folder_ai, folder_nat, place,
                                          payload_list, channel_idx)
                self._checkpoint()

            if "audit" in steps:
                self._run_extraction_audit(folder_ai, folder_nat, ai_files, nat_files,
                                           place, payload_list, strategy_obj, secret_file,
                                           channel_idx)
                self._checkpoint()
        except StopRequested:
            print(f"  [STOP] Pipeline interrupted for {self.strategy_name} | {place}.")
            return "stopped"

        return "completed"

    # ── step 0: embed ───────────────────────────────────────────────────────
    def _embed(self, folder_ai, folder_nat, place, payload_list,
               strategy_obj, secret_file, channel_idx=None) -> None:
        print(f"\n=== Embed [{self.strategy_name}] | {place} | channel={channel_idx} ===")
        process_stego_dataset(folder_ai, "ai", place, payload_list, secret_file,
                              strategy_obj, self.stego_root, channel_idx=channel_idx)
        self._checkpoint()
        process_stego_dataset(folder_nat, "natural", place, payload_list, secret_file,
                              strategy_obj, self.stego_root, channel_idx=channel_idx)

    # ── step 1: compute metrics ──────────────────────────────────────────────
    def _compute_metrics(self, folder_ai, folder_nat, place, payload_list,
                         channel_idx=None) -> dict:
        """
        Returns metrics[img_type][idx][kb] = {psnr, ssim, mse, entropy, bpp}.
        """
        out: dict = {}
        for img_type, files, cover_folder in self._iter_categories(folder_ai, folder_nat):
            self._checkpoint()
            out[img_type] = {}
            stego_root = self._stego_folder(place, img_type.lower(), channel_idx)
            print(f"  [Metrics] {img_type}: {len(files)} images")

            for idx, fname in enumerate(files):
                cover = self._read_cover(os.path.join(cover_folder, fname))
                if cover is None:
                    continue
                h, w = cover.shape[:2]
                out[img_type][idx] = {}

                for kb in payload_list:
                    stego = self._read_stego(
                        os.path.join(stego_root, str(kb), fname))
                    if stego is None:
                        continue
                    out[img_type][idx][kb] = {
                        "psnr": calculate_psnr(cover, stego),
                        "ssim": calculate_ssim(cover, stego),
                        "mse": calculate_mse(cover, stego),
                        "entropy": calculate_entropy(stego),
                        "bpp": calculate_bpp(kb, w, h),
                    }
        return out

    # ── step 2: save metrics CSV ─────────────────────────────────────────────
    def _save_csv(self, metrics, folder_ai, folder_nat, place, payload_list,
                  channel_idx=None) -> None:
        c_label = _channel_label(channel_idx)
        files_by_type = {
            "AI": _list_images(folder_ai),
            "Natural": _list_images(folder_nat),
        }
        rows = []
        for img_type, files in files_by_type.items():
            for idx, fname in enumerate(files):
                for kb in payload_list:
                    m = metrics.get(img_type, {}).get(idx, {}).get(kb)
                    if not m:
                        continue
                    rows.append({
                        "Place": place,
                        "Strategy": self.strategy_name,
                        "Channel": c_label,
                        "ImageFile": fname,
                        "ImageType": img_type,
                        "Payload_KB": kb,
                        "PSNR": round(m["psnr"], 4),
                        "SSIM": round(m["ssim"], 6),
                        "MSE": round(m["mse"], 4),
                        "Entropy": round(m["entropy"], 4),
                        "BPP": round(m["bpp"], 4),
                    })

        fieldnames = ["Place", "Strategy", "Channel", "ImageFile", "ImageType",
                      "Payload_KB", "PSNR", "SSIM", "MSE", "Entropy", "BPP"]
        _write_csv(self._csv_path(place, channel_idx, "metrics"), fieldnames, rows, "Metrics CSV")

    # ── step 3: verify extraction ────────────────────────────────────────────
    def _verify_extraction(self, place, payload_list, strategy_obj, secret_file,
                           folder_ai, folder_nat, channel_idx=None) -> dict:
        """
        Verify extracted bits match the expected payload prefix.
        Returns a tally dict {img_type: {"pass": n, "fail": n, "error": n}}.
        """
        print(f"\n=== Verify [{self.strategy_name}] | {place} | channel={channel_idx} ===")
        with open(secret_file, "r", encoding="utf-8") as f:
            secret_text = f.read()
        full_bits = "".join(format(ord(ch), "08b") for ch in secret_text)

        tally: dict = {}
        for img_type, files, _ in self._iter_categories(folder_ai, folder_nat):
            self._checkpoint()
            stego_root = self._stego_folder(place, img_type.lower(), channel_idx)
            counts = {"pass": 0, "fail": 0, "error": 0}

            for kb in payload_list:
                n_bits = kb * 1024 * 8
                expected = full_bits[:n_bits]
                extra = {"channel_idx": channel_idx} if channel_idx is not None else {}

                for fname in files:
                    path = os.path.normpath(os.path.join(stego_root, str(kb), fname))
                    if not os.path.exists(path):
                        continue
                    try:
                        got = strategy_obj.extract(path, n_bits, **extra)
                        key = "pass" if got == expected else "fail"
                        counts[key] += 1
                    except Exception as exc:  # noqa: BLE001 — report, keep going
                        counts["error"] += 1
                        print(f"  [ERROR] {fname} ({kb} KB): {exc}")

            tally[img_type] = counts
            print(f"  [Verify] {img_type}: {counts}")
        return tally

    # ── step 4: chi-square steganalysis ──────────────────────────────────────
    def _run_chi_square_analysis(self, folder_ai, folder_nat, place, payload_list,
                                 channel_idx=None) -> None:
        print(f"\n--- Chi-square | {place} ---")
        c_label = _channel_label(channel_idx)
        rows = []

        for img_type, files, cover_folder in self._iter_categories(folder_ai, folder_nat):
            self._checkpoint()
            stego_root = self._stego_folder(place, img_type.lower(), channel_idx)
            for fname in files:
                cover = self._read_cover(os.path.join(cover_folder, fname))
                if cover is None:
                    continue
                for kb in payload_list:
                    stego = self._read_stego(os.path.join(stego_root, str(kb), fname))
                    if stego is None:
                        continue
                    results = calculate_chi2(cover, stego)
                    if not results:
                        continue
                    for ch_name, data in results.items():
                        rows.append({
                            "Place": place,
                            "Strategy": self.strategy_name,
                            "ExecutionChannel": c_label,
                            "ImageFile": fname,
                            "ImageType": img_type,
                            "Payload_KB": kb,
                            "AnalyzedChannel": ch_name,
                            "Chi2_Statistic": round(data["chi2_statistic"], 4),
                            "P_Value": round(data["p_value"], 6),
                            "Detected": "YES" if data["detected"] else "NO",
                        })

        fieldnames = ["Place", "Strategy", "ExecutionChannel", "ImageFile", "ImageType",
                      "Payload_KB", "AnalyzedChannel", "Chi2_Statistic", "P_Value", "Detected"]
        _write_csv(self._csv_path(place, channel_idx, "chisquare"), fieldnames, rows, "Chi-Square CSV")

    # ── step 5: RS steganalysis ──────────────────────────────────────────────
    def _run_rs_steganalysis(self, folder_ai, folder_nat, place, payload_list,
                             channel_idx=None) -> None:
        print(f"\n--- RS analysis | {place} ---")
        c_label = _channel_label(channel_idx)
        rows = []

        for img_type, files, _ in self._iter_categories(folder_ai, folder_nat):
            self._checkpoint()
            stego_root = self._stego_folder(place, img_type.lower(), channel_idx)
            for fname in files:
                for kb in payload_list:
                    path = os.path.join(stego_root, str(kb), fname)
                    if not os.path.exists(path):
                        continue
                    stego = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if stego is None:
                        continue
                    target = self._rs_target_channel(stego, channel_idx)
                    p_est, RM, SM, RnM, SnM = rs_analysis(target, group_size=4)
                    rows.append({
                        "Place": place,
                        "Strategy": self.strategy_name,
                        "Channel": c_label,
                        "ImageFile": fname,
                        "ImageType": img_type,
                        "Payload_KB": kb,
                        "Estimated_Length_P": round(p_est, 6),
                        "RM": round(RM, 4),
                        "SM": round(SM, 4),
                        "RnM": round(RnM, 4),
                        "SnM": round(SnM, 4),
                    })

        fieldnames = ["Place", "Strategy", "Channel", "ImageFile", "ImageType",
                      "Payload_KB", "Estimated_Length_P", "RM", "SM", "RnM", "SnM"]
        _write_csv(self._csv_path(place, channel_idx, "rsanalysis"), fieldnames, rows, "RS Analysis CSV")

    @staticmethod
    def _rs_target_channel(stego, channel_idx: int | None):
        """Isolate the channel array RS analysis should run on."""
        if stego.ndim == 2:
            return stego
        if channel_idx is not None:
            return stego[:, :, channel_idx]
        return cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)

    # ── step 6: extraction audit ──────────────────────────────────────────────
    def _run_extraction_audit(self, folder_ai, folder_nat, ai_files, nat_files,
                              place, payload_list, strategy_obj, secret_file, channel_idx=None) -> None:
        """
        Execute bit extraction checks on all embedded variations and log outcomes.

        Columns: method, category, type, payload, channel, image_name, succeeded
        """
        print(f"\n--- Extraction Audit | {place} | Channel: {channel_idx} ---")

        # Build output path and check for existence
        out_dir = _results_path(place, "csv")
        os.makedirs(out_dir, exist_ok=True)
        c_label = _channel_label(channel_idx)
        out_path = os.path.join(out_dir, f"{self.strategy_name}_{place}_{c_label}_extraction_audit.csv")

        if os.path.exists(out_path):
            print(f"  [Audit] Exists, skipping → {out_path}")
            return

        # Read secret file
        with open(secret_file, "r", encoding="utf-8") as f:
            secret_text = f.read()
        full_secret_bits = "".join(format(ord(ch), "08b") for ch in secret_text)

        # Stego roots for AI and Natural
        stego_ai_root = self._stego_folder(place, "ai", channel_idx=channel_idx)
        stego_nat_root = self._stego_folder(place, "natural", channel_idx=channel_idx)

        rows = []

        # Process AI and Natural categories
        for img_type, files, stego_folder in [
            ("ai", ai_files, stego_ai_root),
            ("natural", nat_files, stego_nat_root),
        ]:
            self._checkpoint()
            for kb in sorted(payload_list):
                expected_bits_length = kb * 1024 * 8
                expected_bit_string = full_secret_bits[:expected_bits_length]

                for fname in files:
                    stego_path = os.path.join(stego_folder, str(kb), fname)
                    if not os.path.exists(stego_path):
                        continue

                    # Prepare extraction kwargs (pass channel_idx if not None)
                    extra_kwargs = {}
                    if channel_idx is not None:
                        extra_kwargs["channel_idx"] = channel_idx

                    succeeded = "false"
                    try:
                        extracted = strategy_obj.extract(
                            stego_path, expected_bits_length, **extra_kwargs
                        )
                        if extracted == expected_bit_string:
                            succeeded = "true"
                    except Exception as e:
                        print(f"  [Audit Error] {fname} ({kb} KB): {str(e)}")
                        succeeded = "false"

                    rows.append({
                        "method": self.strategy_name,
                        "category": place,
                        "type": img_type,
                        "payload": kb,
                        "channel": c_label,
                        "image_name": fname,
                        "succeeded": succeeded,
                    })

        # Write CSV
        fieldnames = ["method", "category", "type", "payload", "channel", "image_name", "succeeded"]
        _write_csv(out_path, fieldnames, rows, "Extraction Audit CSV")


# ─────────────────────────────────────────────────────────────────────────────
#  Concrete analyzers
# ─────────────────────────────────────────────────────────────────────────────
class LSBAnalyzer(StegoAnalyzer):
    def __init__(self, stego_root="embeddings", should_stop=None):
        super().__init__("lsbRandomSpatial", stego_root, should_stop)


class LSBCannySobelAnalyzer(StegoAnalyzer):
    def __init__(self, stego_root="embeddings", should_stop=None):
        super().__init__("lsbCannySobel", stego_root, should_stop)


class PVDAnalyzer(StegoAnalyzer):
    def __init__(self, stego_root="embeddings", should_stop=None):
        super().__init__("pvdSequential", stego_root, should_stop)


class LSBMRAnalyzer(StegoAnalyzer):
    def __init__(self, stego_root="embeddings", should_stop=None):
        super().__init__("lsbmr", stego_root, should_stop)