"""Data models for steganography comparison application."""
from dataclasses import dataclass, field
from typing import Literal, Any
import numpy as np


def _pair_share(a: float, b: float, higher_better: bool) -> tuple[float, float]:
    """Score a and b as their share of the pair, oriented so higher = better.
 
    Returns (score_a, score_b) in [0, 1] summing to 1. Lower-is-better metrics
    are clamped at 0 and inverted; a degenerate (zero-sum) pair returns 0.5/0.5.
    """
    if not higher_better:
        a, b = max(0.0, a), max(0.0, b)
    total = a + b
    if total <= 0:
        return 0.5, 0.5
    return (a / total, b / total) if higher_better else (b / total, a / total)
 
 
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
 
    def _chi2(self) -> float:
        for k in ("chi2_avg", "chi2_mean", "chi2"):
            if k in self.extra:
                return float(self.extra[k])
        return 0.0
 
    def _rs(self) -> float:
        for k in ("rs_p_est", "rs_p", "rs_estimate", "rs"):
            if k in self.extra:
                return float(self.extra[k])
        return 0.0
 
 
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
 
    def composite_scores(self, w_fidelity: float = 0.5) -> dict:
        """Pairwise AI-vs-natural composite for this method/payload.
 
        Combines a fidelity axis (PSNR + SSIM, higher better) and a stealth axis
        (Chi2 + RS estimate, lower better). Each metric is normalised as its
        share of the AI+natural pair, so the numbers are comparable only WITHIN
        this pair, not across pairs or methods. MSE and entropy are excluded:
        MSE is redundant with PSNR, and entropy is a cover property that, on this
        dataset, moves opposite to chi2 -- so folding it in inverts the verdict.
 
        w_fidelity: weight on fidelity vs stealth (default 0.5/0.5). Lower it to
        let the 'hiding' verdict lean on stealth (security) over visual quality.
        """
        w_stealth = 1.0 - w_fidelity
        ai, nat = self.ai, self.natural
 
        # fidelity axis (higher better): PSNR + SSIM, 50/50
        ai_p, nat_p = _pair_share(ai.psnr, nat.psnr, higher_better=True)
        ai_s, nat_s = _pair_share(ai.ssim, nat.ssim, higher_better=True)
        ai_fid, nat_fid = 0.5 * ai_p + 0.5 * ai_s, 0.5 * nat_p + 0.5 * nat_s
 
        # stealth axis (lower better): chi2 + RS estimate, 50/50
        ai_c, nat_c = _pair_share(ai._chi2(), nat._chi2(), higher_better=False)
        ai_r, nat_r = _pair_share(ai._rs(), nat._rs(), higher_better=False)
        ai_st, nat_st = 0.5 * ai_c + 0.5 * ai_r, 0.5 * nat_c + 0.5 * nat_r
 
        return {
            "ai_composite": w_fidelity * ai_fid + w_stealth * ai_st,
            "natural_composite": w_fidelity * nat_fid + w_stealth * nat_st,
            "ai_fidelity": ai_fid, "natural_fidelity": nat_fid,
            "ai_stealth": ai_st, "natural_stealth": nat_st,
            "axes_agree": (ai_fid >= nat_fid) == (ai_st >= nat_st),
        }
 
    def winner(self, w_fidelity: float = 0.5) -> Literal["AI", "Natural"]:
        """Which cover is 'Best at Hiding' on the pairwise composite."""
        s = self.composite_scores(w_fidelity)
        return "AI" if s["ai_composite"] >= s["natural_composite"] else "Natural"
 
    def winner_score(self, w_fidelity: float = 0.5) -> float:
        """Composite score of the winning cover."""
        s = self.composite_scores(w_fidelity)
        return max(s["ai_composite"], s["natural_composite"])
 
    def formula_caption(self, w_fidelity: float = 0.5) -> str:
        """Human-readable description of how the composite was built."""
        ws = 1.0 - w_fidelity
        cap = (
            f"Composite = {w_fidelity:.0%} imperceptibility + {ws:.0%} undetectability. "
            "imperceptibility = mean of pairwise-normalised PSNR and SSIM  "
            "; stealth = mean of pairwise-normalised, inverted Chi\u00b2 "
            "and RS estimate.  "
        )
        if not self.composite_scores(w_fidelity)["axes_agree"]:
            cap += " (Fidelity and stealth disagree here; verdict is their weighted mix.)"
        return cap