"""Strategy registry and factory for embedding methods."""
from typing import Dict, Type
from strategy.lsb_strategy import LSBStrategy
from strategy.lsb_Canny_Sobel2 import LSBCannySobelStrategy
from strategy.pvd_strategy import PVDStrategy
from strategy.lsbmr_startegy import LSBMRStrategy
from strategy.embedding_strategy import EmbeddingStrategy


class StrategyRegistry:
    """Registry for embedding strategies."""

    STRATEGIES: Dict[str, Type[EmbeddingStrategy]] = {
        "LSB Random Spatial": LSBStrategy,
        "LSB Canny-Sobel": LSBCannySobelStrategy,
        "PVD Sequential": PVDStrategy,
        "LSBMR": LSBMRStrategy,
    }

    @classmethod
    def get_strategy(cls, name: str) -> EmbeddingStrategy:
        """Get an instance of a strategy by display name."""
        if name not in cls.STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {name}. Available: {list(cls.STRATEGIES.keys())}"
            )

        strategy_class = cls.STRATEGIES[name]

        if name == "LSB Random Spatial":
            return strategy_class(key="BlueAvatarlife123")
        return strategy_class()

    @classmethod
    def get_all_names(cls) -> list[str]:
        """Return all available strategy display names."""
        return list(cls.STRATEGIES.keys())

    @classmethod
    def get_color_mode(cls, strategy_name: str) -> str:
        """
        Return color mode for a strategy.
        "grayscale" for PVD, "color" for all others.
        """
        if strategy_name == "PVD Sequential":
            return "grayscale"
        return "color"
