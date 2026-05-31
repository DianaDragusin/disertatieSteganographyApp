"""Facade layer for steganography operations."""
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image

from core.models import MetricsResult, StegoResult, ComparisonResult
from core.strategy_registry import StrategyRegistry
from ssimAndPsnr import (
    calculate_psnr,
    calculate_ssim,
    calculate_mse,
    calculate_entropy,
    calculate_bpp,
)
from steganalysis.chi_square import calculate_chi2
from steganalysis.rs_analysis import rs_analysis


class StegoFacade:
    """High-level interface for steganography operations."""

    def __init__(self, stego_root: Optional[Path] = None):
        self.stego_root = stego_root or Path("./embeddings")
        self.stego_root.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    #  EMBED
    # ──────────────────────────────────────────────────────────────────────
    def embed_pair(
        self,
        natural_image_path: Path,
        ai_image_path: Path,
        message: str,
        strategy_name: str,
        **kwargs,
    ) -> Tuple[StegoResult, StegoResult]:
        """
        Embed the same message into both natural and AI images.

        Always loads covers as BGR color first, then converts to grayscale
        inside the facade if needed. This guarantees the cover and stego
        arrays go through the same grayscale conversion path.
        """
        message_bits = self._message_to_bits(message)
        total_bits = len(message_bits)
        print(f"\n📊 Embedding {total_bits} bits using '{strategy_name}' strategy...")

        strategy = StrategyRegistry.get_strategy(strategy_name)
        color_mode = StrategyRegistry.get_color_mode(strategy_name)

        # ── Always load covers as BGR color first ─────────────────────────────
        cover_natural_bgr = cv2.imread(str(natural_image_path), cv2.IMREAD_COLOR)
        cover_ai_bgr      = cv2.imread(str(ai_image_path),      cv2.IMREAD_COLOR)
        if cover_natural_bgr is None or cover_ai_bgr is None:
            raise FileNotFoundError("Could not read one of the cover images")

        # Convert to grayscale here if the strategy is grayscale-only.
        # This matches the conversion path used inside pvd_embed exactly.
        if color_mode == "grayscale":
            cover_natural = cv2.cvtColor(cover_natural_bgr, cv2.COLOR_BGR2GRAY)
            cover_ai      = cv2.cvtColor(cover_ai_bgr,      cv2.COLOR_BGR2GRAY)
        else:
            cover_natural = cover_natural_bgr
            cover_ai      = cover_ai_bgr

        # ── Run the strategy ──────────────────────────────────────────────────
        print(f"   ✓ Embedding into natural image...")
        stego_natural_raw = strategy.embed(str(natural_image_path), message_bits, **kwargs)
        # Capture LSBMR coordinate key if available
        key_natural = getattr(strategy, "_last_coordinate_key", None) if strategy_name == "LSBMR" else None
        
        print(f"   ✓ Embedding into AI image...")
        stego_ai_raw      = strategy.embed(str(ai_image_path),      message_bits, **kwargs)
        # Capture LSBMR coordinate key if available
        key_ai = getattr(strategy, "_last_coordinate_key", None) if strategy_name == "LSBMR" else None

        # Normalize whatever the strategy returned into a clean array.
        # _normalize_stego now handles RGB→BGR internally for PIL images,
        # so we do NOT do a second swap here.
        stego_natural = self._normalize_stego(stego_natural_raw, color_mode)
        stego_ai      = self._normalize_stego(stego_ai_raw,      color_mode)

        # ── Sanity checks ─────────────────────────────────────────────────────
        if stego_natural.shape != cover_natural.shape:
            raise ValueError(
                f"Stego/cover shape mismatch (natural): "
                f"cover={cover_natural.shape}, stego={stego_natural.shape}, "
                f"strategy={strategy_name}, color_mode={color_mode}"
            )
        if stego_ai.shape != cover_ai.shape:
            raise ValueError(
                f"Stego/cover shape mismatch (AI): "
                f"cover={cover_ai.shape}, stego={stego_ai.shape}"
            )

        print(f"   ✅ Successfully embedded {total_bits} bits!\n")

        return (
            StegoResult(stego_natural, total_bits, extra={"cover": cover_natural, "lsbmr_key": key_natural}),
            StegoResult(stego_ai,      total_bits, extra={"cover": cover_ai, "lsbmr_key": key_ai}),
        )

        # ──────────────────────────────────────────────────────────────────────
        #  EXTRACT
        # ──────────────────────────────────────────────────────────────────────
    def extract_pair(
        self,
        stego_natural_path: Path,
        stego_ai_path: Path,
        message_length: int,
        strategy_name: str,
        lsbmr_key_natural=None,
        lsbmr_key_ai=None,
        **kwargs,
    ) -> Tuple[str, str]:
        strategy = StrategyRegistry.get_strategy(strategy_name)

        # For LSBMR, pass coordinate keys if available
        print(f"\n[facade.extract_pair DEBUG]")
        print(f"  strategy_name = '{strategy_name}'")
        print(f"  lsbmr_key_natural is None: {lsbmr_key_natural is None}")
        print(f"  lsbmr_key_ai is None: {lsbmr_key_ai is None}")


        extract_kwargs_natural = dict(kwargs)
        extract_kwargs_ai = dict(kwargs)
        
        if strategy_name == "LSBMR":
            if lsbmr_key_natural is not None:
                extract_kwargs_natural["coordinate_key"] = lsbmr_key_natural
            if lsbmr_key_ai is not None:
                extract_kwargs_ai["coordinate_key"] = lsbmr_key_ai

        extracted_bits_natural = strategy.extract(
            str(stego_natural_path),
            message_length=message_length,
            **extract_kwargs_natural,
        )
        message_natural = self._bits_to_message(extracted_bits_natural)

        extracted_bits_ai = strategy.extract(
            str(stego_ai_path),
            message_length=message_length,
            **extract_kwargs_ai,
        )
        message_ai = self._bits_to_message(extracted_bits_ai)

        return (message_natural, message_ai)


    # ──────────────────────────────────────────────────────────────────────
    #  METRICS
    # ──────────────────────────────────────────────────────────────────────
    def compute_metrics(
        self,
        cover_array: np.ndarray,
        stego_array: np.ndarray,
        kb_payload: float,
        image_type: str,
        strategy_name: str,
    ) -> MetricsResult:
        """
        Compute quality metrics for stego image vs cover.

        - grayscale strategies (PVD) → compare on 2D uint8 arrays
        - color strategies → compare on 3-channel BGR uint8 arrays
        """
        color_mode = StrategyRegistry.get_color_mode(strategy_name)

        # ── DEBUG (remove later) ──────────────────────────────────────────
        print(f"\n=== METRIC DEBUG | strategy={strategy_name} | image_type={image_type} ===")
        print(f"  cover_array: shape={cover_array.shape}, dtype={cover_array.dtype}, "
              f"min={cover_array.min()}, max={cover_array.max()}, mean={cover_array.mean():.2f}")
        print(f"  stego_array: shape={stego_array.shape}, dtype={stego_array.dtype}, "
              f"min={stego_array.min()}, max={stego_array.max()}, mean={stego_array.mean():.2f}")
        if cover_array.shape == stego_array.shape:
            d = np.abs(cover_array.astype(np.int32) - stego_array.astype(np.int32))
            print(f"  pixels changed: {np.sum(d>0)}/{d.size} "
                  f"({100*np.sum(d>0)/d.size:.2f}%), max diff={d.max()}, "
                  f"mean diff (over changed): "
                  f"{d[d>0].mean() if np.any(d>0) else 0:.4f}")
        # ──────────────────────────────────────────────────────────────────

        if color_mode == "grayscale":
            cover_proc = cv2.cvtColor(cover_array, cv2.COLOR_BGR2GRAY) if cover_array.ndim == 3 else cover_array
            stego_proc = cv2.cvtColor(stego_array, cv2.COLOR_BGR2GRAY) if stego_array.ndim == 3 else stego_array
        else:
            cover_proc = cover_array
            stego_proc = stego_array

        if cover_proc.shape != stego_proc.shape:
            raise ValueError(
                f"Shape mismatch after normalization: "
                f"cover={cover_proc.shape}, stego={stego_proc.shape}, "
                f"strategy={strategy_name}, color_mode={color_mode}"
            )

        cover_u8 = np.clip(cover_proc, 0, 255).astype(np.uint8)
        stego_u8 = np.clip(stego_proc, 0, 255).astype(np.uint8)

        psnr = calculate_psnr(cover_u8, stego_u8)
        ssim = calculate_ssim(cover_u8, stego_u8)
        mse  = calculate_mse(cover_u8, stego_u8)

        height, width = cover_proc.shape[:2]
        bpp = calculate_bpp(kb_payload, width, height)
        entropy = calculate_entropy(stego_u8)

        # Compute steganalysis metrics
        extra = {}
        try:
            # Chi-square analysis
            chi2_dict = calculate_chi2(cover_u8, stego_u8)
            chi2_values = [v["chi2_statistic"] for v in chi2_dict.values()]
            chi2_mean = np.mean(chi2_values) if chi2_values else None
            extra["chi2_mean"] = chi2_mean
            extra["chi2_per_channel"] = chi2_dict
        except Exception as e:
            print(f"Chi-square analysis failed: {e}")
            extra["chi2_mean"] = None
            extra["chi2_per_channel"] = None

        try:
            # RS analysis
            if stego_u8.ndim == 2:
                # Grayscale
                p_est, _, _, _, _ = rs_analysis(stego_u8, group_size=4)
                extra["rs_p_est"] = p_est
                extra["rs_per_channel"] = {"Gray": p_est}
            else:
                # Color: analyze each channel and average
                p_estimates = {}
                channel_names = ["Blue", "Green", "Red"]
                for ch in range(3):
                    channel_array = stego_u8[:, :, ch]
                    p_est, _, _, _, _ = rs_analysis(channel_array, group_size=4)
                    p_estimates[channel_names[ch]] = p_est
                
                p_est_mean = np.mean(list(p_estimates.values()))
                extra["rs_p_est"] = p_est_mean
                extra["rs_per_channel"] = p_estimates
        except Exception as e:
            print(f"RS analysis failed: {e}")
            extra["rs_p_est"] = None
            extra["rs_per_channel"] = None

        return MetricsResult(
            psnr=psnr,
            ssim=ssim,
            mse=mse,
            bpp=bpp,
            entropy=entropy,
            image_type=image_type,
            strategy=strategy_name,
            extra=extra,
        )

    # ──────────────────────────────────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize_stego(stego_raw, color_mode: str) -> np.ndarray:
        """
        Normalize whatever a strategy returns into a clean uint8 array:
        - grayscale strategies → 2D uint8
        - color strategies     → 3-channel BGR uint8

        PIL.Image is always treated as RGB (PIL's native order), so we
        swap channels here when needed.
        """
        if isinstance(stego_raw, tuple):
            stego_raw = stego_raw[0]

        # Track whether the input was a PIL image (which means RGB order)
        if isinstance(stego_raw, Image.Image):
            arr = np.array(stego_raw)
            was_pil = True
        else:
            arr = np.asarray(stego_raw)
            was_pil = False

        if color_mode == "grayscale":
            # Need a 2D grayscale array
            if arr.ndim == 3:
                # If it came from PIL, channels are RGB; convert to BGR
                # first so the BGR2GRAY weights line up with cv2's loader.
                if was_pil:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            return arr.astype(np.uint8)
        else:
            # color_mode == "color" → return BGR (matches cv2.imread)
            if arr.ndim == 2:
                # Grayscale array under a color strategy — promote to 3-channel
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif was_pil:
                # PIL gave us RGB; swap to BGR so it matches the cover
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr.astype(np.uint8)
        

    @staticmethod
    def _message_to_bits(message: str) -> str:
        return "".join(f"{ord(c):08b}" for c in message)

    @staticmethod
    def _bits_to_message(bits: str) -> str:
        try:
            out = ""
            for i in range(0, len(bits), 8):
                byte = bits[i:i+8]
                if len(byte) < 8:
                    break
                out += chr(int(byte, 2))
            return out
        except (ValueError, OverflowError):
            return "[Extraction Error]"