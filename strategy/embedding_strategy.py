from abc import ABC, abstractmethod
from PIL import Image
import numpy as np


class EmbeddingStrategy(ABC):
    """Abstract base class for embedding strategies."""
    
    @abstractmethod
    def embed(self, image_path, message_bits, **kwargs):
        """
        Embed message bits into an image.
        
        Args:
            image_path (str): Path to the cover image
            message_bits (str): Binary message as string of 0s and 1s
            **kwargs: Additional parameters for the strategy
            
        Returns:
            PIL.Image: Stego image with embedded message
        """
        pass
    
    @abstractmethod
    def extract(self, image_path, **kwargs):
        """
        Extract message bits from a stego image.
        
        Args:
            image_path (str): Path to the stego image
            **kwargs: Additional parameters for the strategy
            
        Returns:
            str: Extracted binary message
        """
        pass
    
    @abstractmethod
    def get_name(self):
        """Return the name of the strategy."""
        pass
