from strategy.embedding_strategy import EmbeddingStrategy
from PIL import Image
import numpy as np
import cv2
from pvdUpdated import pvd_embed, pvd_extract, message_to_bits, bits_to_message


class PVDStrategy(EmbeddingStrategy):
    """PVD (Pixel Value Differencing) embedding strategy."""
    
    def __init__(self, ranges=None):
        """
        Initialize PVD strategy.
        
        Args:
            ranges (list): Range table for PVD. Uses default if None.
        """
        if ranges is None:
            # Default PVD ranges
            self.ranges = [
                (0,   7,   3),
                (8,   15,  3),
                (16,  31,  4),
                (32,  63,  5),
                (64,  127, 6),
                (128, 255, 7)
            ]
        else:
            self.ranges = ranges
    
    def embed(self, image_path, message_bits, **kwargs):
        """
        Embed message bits using PVD method.
        
        Args:
            image_path (str): Path to the cover image
            message_bits (str): Binary message as string of 0s and 1s
            **kwargs: Additional parameters (ignored for PVD)
            
        Returns:
            PIL.Image: Stego image
        """
        # Convert message bits string to message text
        message = self._bits_to_text(message_bits)
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Embed using PVD method
        stego_array, bits_embedded = pvd_embed(img, message, ranges=self.ranges)
        
        # Convert back to PIL Image (convert BGR to RGB)
        stego_img = Image.fromarray(stego_array, mode='L')
        return stego_img
    
    def extract(self, image_path, message_length=None, **kwargs):
        """
        Extract message bits using PVD method.
        
        Args:
            image_path (str): Path to the stego image
            message_length (int): Expected length of the message in bits
            **kwargs: Additional parameters (ignored for PVD)
            
        Returns:
            str: Extracted binary message
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        if message_length is None:
            raise ValueError("message_length is required for PVD extraction")
        
        extracted_message = pvd_extract(img, num_bits=message_length, ranges=self.ranges)
        # Convert message back to bits
        bits = ''.join(format(ord(char), '08b') for char in extracted_message)
        return bits[:message_length]
    
    def get_name(self):
        return "pvdSequential"
    
    @staticmethod
    def _bits_to_text(bits_string):
        """Convert binary string to text."""
        message = []
        for i in range(0, len(bits_string), 8):
            byte = bits_string[i:i+8]
            if len(byte) == 8:
                message.append(chr(int(byte, 2)))
        return ''.join(message)
