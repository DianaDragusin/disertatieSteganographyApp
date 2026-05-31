from strategy.embedding_strategy import EmbeddingStrategy
from PIL import Image
import numpy as np
import cv2
from lsbRandomProcent import embed_random_lsb_spatial, extract_random_lsb_spatial


class LSBStrategy(EmbeddingStrategy):
    """LSB (Least Significant Bit) embedding strategy."""
    
    def __init__(self, key="default_key"):
        self.key = key
    
    def embed(self, image_path, message_bits, **kwargs):
        """
        Embed message bits using LSB method.
        
        Args:
            image_path (str): Path to the cover image
            message_bits (str): Binary message as string of 0s and 1s
            **kwargs: Can include 'key' for pseudo-random spacing
            
        Returns:
            PIL.Image: Stego image
        """
        # Convert message bits string to message text
        message = self._bits_to_text(message_bits)
        
        # Load image as numpy array
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Embed using the adaptive LSB method
        stego_array = embed_random_lsb_spatial(img, message, key=self.key)
        
        # Convert back to PIL Image
        stego_img = Image.fromarray(cv2.cvtColor(stego_array, cv2.COLOR_BGR2RGB))
        return stego_img
    
    def extract(self, image_path, message_length=None, **kwargs):
        """
        Extract message bits using LSB method.
        
        Args:
            image_path (str): Path to the stego image
            message_length (int): Expected length of the message in bits
            **kwargs: Can include 'key' for pseudo-random spacing
            
        Returns:
            str: Extracted binary message
        """
        if message_length is None:
            raise ValueError("message_length is required for LSB extraction")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Reshape image to 1D for extraction
        flat = img.flatten()
        extracted_message = extract_random_lsb_spatial(flat.reshape(img.shape), message_length=message_length, key=self.key)
        # Convert message back to bits
        bits = ''.join(format(ord(char), '08b') for char in extracted_message)
        return bits[:message_length]
    
    def get_name(self):
        return "lsbRandomSpatial"
    
    @staticmethod
    def _bits_to_text(bits_string):
        """Convert binary string to text."""
        message = []
        for i in range(0, len(bits_string), 8):
            byte = bits_string[i:i+8]
            if len(byte) == 8:
                message.append(chr(int(byte, 2)))
        return ''.join(message)
