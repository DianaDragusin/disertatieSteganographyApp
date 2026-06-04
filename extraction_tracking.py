"""
extraction_tracking.py
=======================

Coordinate-level extraction tracking for the four embedding strategies.

For each method this module provides a *tracked* extractor that returns, in
extraction order:

    extracted_bits   : list[int]            – the bit read at each step
    per_bit_coords   : list[(row, col)]     – the pixel that bit was read from

A single verification layer (`verify_against_message`) then compares those bits
against the known ground-truth message and splits the coordinates into two
lists — pixels that were read CORRECTLY and pixels that were read WRONG —
while counting both. `plot_extraction_map` draws those two lists over the
grayscale image (green = correct, red = wrong) and labels the figure with
"correct / total (percent%)".

The heavy logic lives here; each strategy class only needs a thin
`extract_tracked(...)` wrapper that calls the matching function below.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

import numpy as np
import cv2

# ───────────────────────────────────────────────────────────────────────────
#  Small bit helpers (kept local so this file has no hard import on the
#  strategy modules and can be unit-tested on its own).
# ───────────────────────────────────────────────────────────────────────────

def message_to_bits(message: str) -> List[int]:
    """Text → list of bits, 8 bits per character (matches every core file)."""
    bits: List[int] = []
    for ch in message:
        bits.extend(int(b) for b in format(ord(ch), "08b"))
    return bits


def bitstring_to_list(bits: Union[str, List[int]]) -> List[int]:
    """Accept '0101...' or [0,1,0,1,...] and always return list[int]."""
    if isinstance(bits, str):
        return [int(c) for c in bits]
    return [int(b) for b in bits]


# ===========================================================================
#  1. TRACKED EXTRACTORS  — one per method
#     Each mirrors the corresponding core extractor exactly so that the
#     coordinates returned are the real read positions.
# ===========================================================================

def tracked_extract_lsb_random(stego_bgr: np.ndarray,
                               message_length: int,
                               key,
                               ) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    LSB Random Spatial — 1 bit per visited flat position.

    Mirrors lsbRandomProcent.extract_random_lsb_spatial / generate_intervals_spatial.
    The flat index is unravelled back to (row, col) for the overlay.
    """
    import random

    flat = stego_bgr.flatten()
    N = len(flat)

    # --- reproduce generate_intervals_spatial(message_length, N, key) ---
    random.seed(key)
    intervals = []
    max_first = N // message_length
    j0 = random.randint(0, max_first)
    intervals.append(j0)
    for i in range(1, message_length):
        remaining_positions = N - j0 - 1   # note: uses the *initial* j like the original
        remaining_bits = message_length - i
        if remaining_bits <= 0:
            break
        max_step = remaining_positions // remaining_bits
        if max_step <= 0:
            raise ValueError("Not enough space to embed message")
        step = random.randint(1, max_step)
        intervals.append(step)
        j0 += step
    # ---------------------------------------------------------------------

    bits: List[int] = []
    coords: List[Tuple[int, int]] = []
    j = intervals[0]
    for i in range(message_length):
        bits.append(int(flat[j] & 1))
        r, c = np.unravel_index(j, stego_bgr.shape)[:2]
        coords.append((int(r), int(c)))
        if i < message_length - 1:
            j += intervals[i + 1]
    return bits, coords


_PVD_RANGES = [
    (0, 7, 3), (8, 15, 3), (16, 31, 4),
    (32, 63, 5), (64, 127, 6), (128, 255, 7),
]


def _pvd_get_range(diff, ranges):
    for (low, high, n) in ranges:
        if low <= diff <= high:
            return low, high, n
    return ranges[-1]


def tracked_extract_pvd(stego_gray: np.ndarray,
                       num_bits: int,
                       ranges=None,
                       ) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    PVD Sequential — n bits per pixel pair (n depends on the difference bucket).

    Mirrors pvdUpdated.pvd_extract. Every bit emitted by a pair is tagged with
    the (row, col) of that pair's LEFT pixel, so a bucket-boundary desync shows
    up as a band of wrong pixels from the divergence point onward.
    """
    if ranges is None:
        ranges = _PVD_RANGES
    if stego_gray.ndim == 3:
        stego_gray = cv2.cvtColor(stego_gray, cv2.COLOR_BGR2GRAY)

    bits: List[int] = []
    coords: List[Tuple[int, int]] = []
    h, w = stego_gray.shape

    for row in range(h):
        col = 0
        while col < w - 1 and len(bits) < num_bits:
            p1 = int(stego_gray[row, col])
            p2 = int(stego_gray[row, col + 1])
            diff = abs(p1 - p2)
            low, _, n_bits = _pvd_get_range(diff, ranges)
            secret_val = diff - low
            for b in format(secret_val, f"0{n_bits}b"):
                bits.append(int(b))
                coords.append((row, col))
            col += 2
        if len(bits) >= num_bits:
            break

    return bits[:num_bits], coords[:num_bits]


def _canny_sobel_masks(image):
    """Identical to lsb_Canny_Sobel.get_hybrid_xyz_masks (0xF8 stabilisation)."""
    stable = image.copy() & 0xF8
    gray = cv2.cvtColor(stable, cv2.COLOR_BGR2GRAY) if stable.ndim == 3 else stable.copy()
    canny_mask = (cv2.Canny(gray, 100, 200) > 0).astype(np.uint8)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mask = (np.sqrt(sx ** 2 + sy ** 2) > 50).astype(np.uint8)
    z = cv2.bitwise_and(canny_mask, sobel_mask)
    y = cv2.bitwise_xor(cv2.bitwise_or(canny_mask, sobel_mask), z)
    x = cv2.bitwise_not(cv2.bitwise_or(canny_mask, sobel_mask))
    return x, y, z


def tracked_extract_canny_sobel(stego_rgb: np.ndarray,
                               x_bits: int = 1, y_bits: int = 2, z_bits: int = 3,
                               ) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    LSB Canny–Sobel — reads the 32-bit length header, then that many payload
    bits, in Z→Y→X region order across the 3 channels.

    Returns ONLY the payload bits (header is consumed internally) together with
    the (row, col) each payload bit was read from. Multiple bits can share a
    pixel; a pixel appears once per bit read from it.
    """
    x_mask, y_mask, z_mask = _canny_sobel_masks(stego_rgb)

    bit_idx = 0
    total_to_read = 32                # header phase
    header_parsed = False
    header_bits = ""
    payload_bits: List[int] = []
    payload_coords: List[Tuple[int, int]] = []

    for mask, depth in [(z_mask, z_bits), (y_mask, y_bits), (x_mask, x_bits)]:
        for r, c in np.argwhere(mask > 0):
            for channel in range(3):
                pixel_val = int(stego_rgb[r, c, channel])
                for b in range(depth):
                    if bit_idx < total_to_read:
                        bit = (pixel_val >> b) & 1
                        if bit_idx < 32:
                            header_bits += str(bit)
                        else:
                            payload_bits.append(int(bit))
                            payload_coords.append((int(r), int(c)))
                        bit_idx += 1
                        if bit_idx == 32 and not header_parsed:
                            msg_len = int(header_bits, 2)
                            total_to_read = 32 + msg_len
                            header_parsed = True
                if bit_idx >= total_to_read:
                    return payload_bits, payload_coords
    return payload_bits, payload_coords


def _lsbmr_f_relation(x_i, x_i1):
    return (int(x_i) // 2 + int(x_i1)) & 1


def tracked_extract_lsbmr(stego_bgr: np.ndarray,
                         total_bits_to_extract: int,
                         shared_key_coordinates: list,
                         channel_idx: int = 0,
                         ) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    LSBMR — 2 bits per shared-key coordinate pair.

    Mirrors lsbmr.extract. The first bit of a pair is tagged to the left pixel
    (y, x); the second bit to the right pixel (y, x+1), so a per-pixel overlay
    can distinguish which half of a pair failed.
    """
    channel = stego_bgr[:, :, channel_idx]
    bits: List[int] = []
    coords: List[Tuple[int, int]] = []

    for y, x in shared_key_coordinates:
        if len(bits) >= total_bits_to_extract:
            break
        xi = int(channel[y, x])
        xi1 = int(channel[y, x + 1])
        bits.append(int(xi) & 1)
        coords.append((int(y), int(x)))
        if len(bits) < total_bits_to_extract:
            bits.append(_lsbmr_f_relation(xi, xi1))
            coords.append((int(y), int(x + 1)))

    return bits, coords


# ===========================================================================
#  3. ONE-CALL DISPATCH  — the GUI (and anything else) calls only this.
# ===========================================================================

def build_extraction_report(strategy_name: str,
                           stego_array: np.ndarray,
                           message: str,
                           coordinate_key=None,
                           channel_idx: int = 0,
                           lsb_key: str = "BlueAvatarlife123",
                           ) -> "ExtractionReport":
    """
    Run the right tracked extractor for `strategy_name` on an in-memory stego
    array and verify it against `message`. Returns an ExtractionReport.

    strategy_name is the display name used in the app:
        "LSB Random Spatial", "PVD Sequential", "LSB Canny-Sobel", "LSBMR"

    - LSB Random needs the same key used at embedding (registry default below).
    - LSBMR needs the coordinate_key produced at embedding (+ channel_idx).
    - PVD / Canny-Sobel need nothing extra (Canny reads its own length header).

    Array expectations match what the facade stores in extra["stego_array"]:
    PVD is 2-D grayscale; the others are 3-channel BGR.
    """
    gt_bits = "".join(f"{ord(c):08b}" for c in message)
    msg_len = len(gt_bits)

    if strategy_name == "PVD Sequential":
        bits, coords = tracked_extract_pvd(stego_array, msg_len)
    elif strategy_name == "LSB Canny-Sobel":
        bits, coords = tracked_extract_canny_sobel(stego_array)
    elif strategy_name == "LSBMR":
        if coordinate_key is None:
            raise ValueError("LSBMR needs the coordinate_key from embedding.")
        bits, coords = tracked_extract_lsbmr(stego_array, msg_len, coordinate_key, channel_idx)
    elif strategy_name == "LSB Random Spatial":
        bits, coords = tracked_extract_lsb_random(stego_array, msg_len, lsb_key)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return verify_against_message(bits, coords, gt_bits)


# ===========================================================================
#  4. VERIFICATION HELPERS / REPORT
# ===========================================================================


@dataclass
class ExtractionReport:
    correct_count: int = 0
    incorrect_count: int = 0
    total_message_bits: int = 0
    good_coords: List[Tuple[int, int]] = field(default_factory=list)
    bad_coords: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def percent_correct(self) -> float:
        if self.total_message_bits == 0:
            return 0.0
        return 100.0 * self.correct_count / self.total_message_bits

    def summary(self) -> str:
        return (f"{self.correct_count} / {self.total_message_bits} bits correct "
                f"({self.percent_correct:.1f}%) | wrong: {self.incorrect_count}")


def verify_against_message(extracted_bits: List[int],
                          per_bit_coords: List[Tuple[int, int]],
                          ground_truth_bits: Union[str, List[int]],
                          ) -> ExtractionReport:
    """
    Compare each extracted bit to the corresponding ground-truth message bit.

    - bit matches  → coordinate goes to good_coords, correct_count += 1
    - bit differs  → coordinate goes to bad_coords,  incorrect_count += 1
    - a message bit that was never reached (extraction too short) counts as
      incorrect but contributes no coordinate.

    total_message_bits is the LENGTH OF THE MESSAGE, so percent_correct is
    "how much of the message came back intact" exactly as requested.
    """
    truth = bitstring_to_list(ground_truth_bits)
    report = ExtractionReport(total_message_bits=len(truth))

    n_compare = min(len(extracted_bits), len(truth))
    for i in range(n_compare):
        coord = per_bit_coords[i] if i < len(per_bit_coords) else None
        if extracted_bits[i] == truth[i]:
            report.correct_count += 1
            if coord is not None:
                report.good_coords.append(coord)
        else:
            report.incorrect_count += 1
            if coord is not None:
                report.bad_coords.append(coord)

    # message bits that extraction never reached
    report.incorrect_count += max(0, len(truth) - n_compare)
    return report


# ===========================================================================
#  3. PLOT
# ===========================================================================

def render_extraction_map(ax,
                         image: Union[str, np.ndarray],
                         report: ExtractionReport,
                         title: str = "",
                         max_points: int = 90000,
                         dot_size: float = 4.0):
    """
    Draw the extraction map onto an existing matplotlib Axes.

    Grayscale image + green dots (correct bits) + red dots (wrong bits), titled
    "<title> | correct / total (percent%)". Shared by both the file saver
    (plot_extraction_map) and the GUI plot tab so there is ONE implementation.

    If a coordinate list is huge it is randomly down-sampled FOR DISPLAY only;
    the counts in the title stay exact and a note is appended.
    """
    if isinstance(image, str):
        gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Could not load image for plot: {image}")
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    def _sample(coords):
        if len(coords) <= max_points:
            return (np.array(coords) if coords else np.empty((0, 2))), False
        idx = np.random.choice(len(coords), max_points, replace=False)
        return np.array(coords)[idx], True

    good, good_sampled = _sample(report.good_coords)
    bad, bad_sampled = _sample(report.bad_coords)

    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    # coords are (row, col) → scatter wants (x=col, y=row)
    if len(good):
        ax.scatter(good[:, 1], good[:, 0], s=dot_size, c="#2ecc71",
                   marker=".", linewidths=0, label="correct", rasterized=True)
    if len(bad):
        ax.scatter(bad[:, 1], bad[:, 0], s=dot_size, c="#e74c3c",
                   marker=".", linewidths=0, label="wrong", rasterized=True)

    full = f"{title} | " if title else ""
    full += report.summary()
    if good_sampled or bad_sampled:
        full += f"\n(Max dots shown: {max_points:,}/ for display)"
    ax.set_title(full, fontsize=10)
    ax.axis("off")
    ax.legend(loc="upper right", markerscale=4, framealpha=0.85)
    return ax


def plot_extraction_map(image: Union[str, np.ndarray],
                       report: ExtractionReport,
                       title_prefix: str = "",
                       save_path: Optional[str] = None,
                       max_points: int = 90000,
                       dot_size: float = 4.0,
                       show: bool = False):
    """
    Standalone figure version (writes a PNG). Uses render_extraction_map.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 10))
    render_extraction_map(ax, image, report, title=title_prefix,
                          max_points=max_points, dot_size=dot_size)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path