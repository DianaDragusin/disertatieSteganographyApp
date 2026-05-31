"""Data models for steganography comparison application."""
from dataclasses import dataclass, field
from typing import Literal, Any
import numpy as np


@dataclass
class MetricsResult:
    """Metrics for a single stego image."""
    psnr: float
    ssim: float
    mse: float
    bpp: float
    entropy: float
    image_type: Literal["AI", "Natural"]
    strategy: str
    extra: dict = field(default_factory=dict)
    
    def composite_score(self) -> float:
        """
        Composite score combining PSNR, SSIM, and MSE.
        Higher PSNR and SSIM are better; lower MSE is better.
        Returns: score (higher is better)
        """
        # Normalize PSNR to [0, 1] (assuming typical range 20-50 dB)
        psnr_norm = max(0, min(1, (self.psnr - 20) / 30))
        
        # SSIM is typically in [0, 1]
        ssim_norm = max(0, min(1, self.ssim))
        
        # Normalize MSE inverse (lower MSE is better, assuming max 5000)
        mse_norm = max(0, 1 - (self.mse / 5000))
        
        return (psnr_norm * 0.4 + ssim_norm * 0.4 + mse_norm * 0.2)


@dataclass
class StegoResult:
    """Result of embedding operation."""
    stego_image: np.ndarray
    total_bits: int
    extra: dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Paired results for AI vs Natural image embedding."""
    ai: MetricsResult
    natural: MetricsResult
    
    def winner(self) -> Literal["AI", "Natural"]:
        """
        Return which image has better composite score.
        This is the "Best at hiding" metric.
        """
        ai_score = self.ai.composite_score()
        natural_score = self.natural.composite_score()
        return "AI" if ai_score >= natural_score else "Natural"
    
    def winner_score(self) -> float:
        """Return the composite score of the winner."""
        ai_score = self.ai.composite_score()
        natural_score = self.natural.composite_score()
        return max(ai_score, natural_score)
