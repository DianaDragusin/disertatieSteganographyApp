"""Worker threads for long-running operations."""
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import tempfile
import traceback
from PyQt6.QtCore import QThread, pyqtSignal

from core.facade import StegoFacade
from core.models import ComparisonResult
from batch_runner import run_full_batch


class EmbedWorker(QThread):
    """Worker thread for embedding operation."""
    
    progress = pyqtSignal(int)
    finished = pyqtSignal(ComparisonResult)
    error = pyqtSignal(str)
    
    def __init__(self, facade: StegoFacade, natural_path: Path, ai_path: Path,
                 message: str, strategy_name: str, kb_payload: float, **kwargs):
        """Initialize embed worker."""
        super().__init__()
        self.facade = facade
        self.natural_path = Path(natural_path)
        self.ai_path = Path(ai_path)
        self.message = message
        self.strategy_name = strategy_name
        self.kb_payload = kb_payload
        self.kwargs = kwargs
    
    def run(self):
        """Execute embedding operation."""
        try:
            self.progress.emit(10)

            # Embed — facade now loads the covers itself in the right color mode
            # and returns them inside StegoResult.extra["cover"]. Don't reload here.
            stego_natural, stego_ai = self.facade.embed_pair(
                self.natural_path, self.ai_path, self.message,
                self.strategy_name, **self.kwargs
            )

            self.progress.emit(50)

            # Compute metrics using the cover the facade actually used.
            metrics_natural = self.facade.compute_metrics(
                cover_array=stego_natural.extra["cover"],
                stego_array=stego_natural.stego_image,
                kb_payload=self.kb_payload,
                image_type="Natural",
                strategy_name=self.strategy_name,
            )

            self.progress.emit(75)

            metrics_ai = self.facade.compute_metrics(
                cover_array=stego_ai.extra["cover"],
                stego_array=stego_ai.stego_image,
                kb_payload=self.kb_payload,
                image_type="AI",
                strategy_name=self.strategy_name,
            )

            self.progress.emit(90)

            # Stash arrays for downstream use (diff viewer, extract, plots)
            result = ComparisonResult(ai=metrics_ai, natural=metrics_natural)
            result.ai.extra["stego_array"] = stego_ai.stego_image
            result.natural.extra["stego_array"] = stego_natural.stego_image
            result.ai.extra["cover_array"] = stego_ai.extra["cover"]
            result.natural.extra["cover_array"] = stego_natural.extra["cover"]
            # Stash LSBMR coordinate keys for extraction if needed
            result.ai.extra["lsbmr_key"] = stego_ai.extra.get("lsbmr_key")
            result.natural.extra["lsbmr_key"] = stego_natural.extra.get("lsbmr_key")

            self.progress.emit(100)
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(f"Embedding error: {str(e)}\n{traceback.format_exc()}")


class ExtractWorker(QThread):
    """Worker thread for extraction operation."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, facade: StegoFacade, stego_natural_array: np.ndarray,
                 stego_ai_array: np.ndarray, original_message: str,
                 strategy_name: str,
                 lsbmr_key_natural=None, lsbmr_key_ai=None,
                 **kwargs):
        """Initialize extract worker."""
        super().__init__()
        self.facade = facade
        self.stego_natural_array = stego_natural_array
        self.stego_ai_array = stego_ai_array
        self.original_message = original_message
        self.strategy_name = strategy_name
        self.lsbmr_key_natural = lsbmr_key_natural
        self.lsbmr_key_ai = lsbmr_key_ai
        self.kwargs = kwargs

    def run(self):
        """Execute extraction operation."""

        print(f"\n[ExtractWorker.run DEBUG]")
        print(f"  strategy_name = '{self.strategy_name}'")
        print(f"  self.lsbmr_key_natural is None: {self.lsbmr_key_natural is None}")
        print(f"  self.lsbmr_key_ai is None: {self.lsbmr_key_ai is None}")
        try:
            self.progress.emit(10)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                stego_natural_path = tmpdir / "stego_natural.png"
                stego_ai_path = tmpdir / "stego_ai.png"

                cv2.imwrite(str(stego_natural_path), self.stego_natural_array)
                cv2.imwrite(str(stego_ai_path),      self.stego_ai_array)

                self.progress.emit(20)

                message_bits = StegoFacade._message_to_bits(self.original_message)
                message_length = len(message_bits)

                self.progress.emit(40)

                # Forward LSBMR keys to the facade
                extracted_natural, extracted_ai = self.facade.extract_pair(
                    stego_natural_path,
                    stego_ai_path,
                    message_length,
                    self.strategy_name,
                    lsbmr_key_natural=self.lsbmr_key_natural,
                    lsbmr_key_ai=self.lsbmr_key_ai,
                    **self.kwargs,
                )

                self.progress.emit(80)

                result = {
                    'natural': extracted_natural,
                    'ai': extracted_ai,
                    'match_natural': extracted_natural == self.original_message,
                    'match_ai': extracted_ai == self.original_message,
                }

                self.progress.emit(100)
                self.finished.emit(result)

        except Exception as e:
            self.error.emit(f"Extraction error: {str(e)}\n{traceback.format_exc()}")


class BatchWorker(QThread):
    """Worker thread for batch analysis pipeline."""

    progress = pyqtSignal(str, int, int)  # stage_label, current, total
    verify_tally = pyqtSignal(str, str, str, dict)  # strategy, place, img_type, tally
    finished = pyqtSignal(dict)  # summary dict
    error = pyqtSignal(str)

    def __init__(self, strategies: list[str], places: list[str], steps: list[str]):
        """Initialize batch worker."""
        super().__init__()
        self.strategies = strategies
        self.places = places
        self.steps = steps
        self._stop = False

    def request_stop(self):
        """Request stop flag to be set."""
        self._stop = True

    def run(self):
        """Execute batch pipeline."""
        try:
            def progress_cb(stage_label: str, current: int, total: int) -> None:
                """Callback for progress updates."""
                self.progress.emit(stage_label, current, total)

            def verify_cb(strategy: str, place: str, img_type: str, tally: dict) -> None:
                """Callback for extraction tally updates."""
                self.verify_tally.emit(strategy, place, img_type, tally)

            summary = run_full_batch(
                strategies=self.strategies,
                places=self.places,
                steps=self.steps,
                should_stop=lambda: self._stop,
                progress_cb=progress_cb,
                verify_cb=verify_cb,
            )

            self.finished.emit(summary)

        except Exception as e:
            self.error.emit(f"Batch pipeline error: {str(e)}\n{traceback.format_exc()}")