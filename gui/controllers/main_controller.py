"""MVC Controller for main application."""
from pathlib import Path
from typing import Optional
from PyQt6.QtCore import QObject, pyqtSignal

from core.facade import StegoFacade
from core.models import ComparisonResult
from gui.workers import EmbedWorker, ExtractWorker


class MainController(QObject):
    """Controller mediating between View and Facade."""
    
    # Signals for embed operation
    embed_progress = pyqtSignal(int)
    embed_finished = pyqtSignal(ComparisonResult)
    embed_error = pyqtSignal(str)
    
    # Signals for extract operation
    extract_progress = pyqtSignal(int)
    extract_finished = pyqtSignal(dict)  # {'natural': str, 'ai': str, 'match_natural': bool, 'match_ai': bool}
    extract_error = pyqtSignal(str)
    
    def __init__(self, facade: StegoFacade):
        """
        Initialize controller.
        
        Args:
            facade: StegoFacade instance
        """
        super().__init__()
        self.facade = facade
        self.embed_worker: Optional[EmbedWorker] = None
        self.extract_worker: Optional[ExtractWorker] = None
    
    def embed_pair(
        self,
        natural_path: Path,
        ai_path: Path,
        message: str,
        strategy_name: str,
        kb_payload: float,
        **kwargs
    ):
        """
        Start embedding operation in background thread.
        
        Args:
            natural_path: Path to natural image
            ai_path: Path to AI image
            message: Secret message
            strategy_name: Embedding strategy name
            kb_payload: Payload size in KB
            **kwargs: Additional strategy parameters
        """
        # Stop any existing worker
        if self.embed_worker and self.embed_worker.isRunning():
            self.embed_worker.quit()
            self.embed_worker.wait()
        
        # Create and start worker
        self.embed_worker = EmbedWorker(
            self.facade,
            natural_path,
            ai_path,
            message,
            strategy_name,
            kb_payload,
            **kwargs
        )
        
        # Connect signals
        self.embed_worker.progress.connect(self.embed_progress.emit)
        self.embed_worker.finished.connect(self._on_embed_finished)
        self.embed_worker.error.connect(self._on_embed_error)
        
        self.embed_worker.start()
    
    def extract_pair(
        self,
        comparison_result: ComparisonResult,
        message: str,
        strategy_name: str,
        **kwargs
    ):
        """
        Start extraction operation in background thread.
        """
        # Stop any existing worker
        if self.extract_worker and self.extract_worker.isRunning():
            self.extract_worker.quit()
            self.extract_worker.wait()

        # Get stego arrays from result
        stego_natural = comparison_result.natural.extra.get("stego_array")
        stego_ai = comparison_result.ai.extra.get("stego_array")

        if stego_natural is None or stego_ai is None:
            self.extract_error.emit("Stego image data not available")
            return

        # Pull LSBMR coordinate keys (will be None for other strategies — that's fine)
        lsbmr_key_natural = comparison_result.natural.extra.get("lsbmr_key")
        lsbmr_key_ai      = comparison_result.ai.extra.get("lsbmr_key")

        # Create and start worker
        self.extract_worker = ExtractWorker(
            self.facade,
            stego_natural,
            stego_ai,
            message,
            strategy_name,
            lsbmr_key_natural=lsbmr_key_natural,
            lsbmr_key_ai=lsbmr_key_ai,
            **kwargs,
        )

        # Connect signals
        self.extract_worker.progress.connect(self.extract_progress.emit)
        self.extract_worker.finished.connect(self._on_extract_finished)
        self.extract_worker.error.connect(self._on_extract_error)

        self.extract_worker.start()
    
    def _on_embed_finished(self, result: ComparisonResult):
        """Handle embed worker completion."""
        self.embed_finished.emit(result)
    
    def _on_embed_error(self, error: str):
        """Handle embed worker error."""
        self.embed_error.emit(error)
    
    def _on_extract_finished(self, result: dict):
        """Handle extract worker completion."""
        self.extract_finished.emit(result)
    
    def _on_extract_error(self, error: str):
        """Handle extract worker error."""
        self.extract_error.emit(error)
