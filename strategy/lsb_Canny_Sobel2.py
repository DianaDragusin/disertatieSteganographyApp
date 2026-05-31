import os
import cv2
import numpy as np
from PIL import Image
from strategy.embedding_strategy import EmbeddingStrategy
from lsbCannySobel2 import embed_xyz, extract_xyz


# Import the core backend processing functions from your local file
# Ensure 'lsb_xyz_core' matches your target core file name precisely



class LSBCannySobelStrategy(EmbeddingStrategy):
    """
    Hybrid Canny-Sobel XYZ Spatial Domain Steganography Strategy.
    Uses multi-depth dynamic pixel masking across stratified texture priority zones.
    """
    
    def __init__(self, x_bits=1, y_bits=2, z_bits=3):
        super().__init__()
        self.x_bits = x_bits
        self.y_bits = y_bits
        self.z_bits = z_bits

    def embed(self, image_path, message_bits_str, **kwargs):
        """
        Embeds binary bit stream payload natively into texture-stratified masks.
        
        Args:
            image_path (str): File system source string path.
            message_bits_str (str): Raw string of binary bits (e.g., "01001...").
            channel_idx (int, optional): Ignored here because this algorithm 
                                         processes all 3 channels simultaneously.
        """
        # 1. Load cover matrix natively using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load cover target asset: {image_path}")
            
        # 2. Convert incoming framework bit-string into raw string text 
        # because your embed_xyz engine handles parsing message lengths natively
        secret_text = self._bits_to_text(message_bits_str)
        
        # 3. Execute dynamic hybrid masking and bit insertion
        stego_array, _ = embed_xyz(
            img, 
            secret_text, 
            x_bits=self.x_bits, 
            y_bits=self.y_bits, 
            z_bits=self.z_bits
        )
        
        # 4. Convert array format from BGR (OpenCV) to RGB for PIL framework compatibility
        stego_img = Image.fromarray(cv2.cvtColor(stego_array, cv2.COLOR_BGR2RGB))
        return stego_img

    def extract(self, image_path, message_length=None, **kwargs):
        """
        Extracts the secret binary stream payload from the stratified pixel arrays.
        Uses the embedded 32-bit header to discover payload boundary thresholds.
        """
        # 1. Load stego asset matrix
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load target stego asset: {image_path}")
            
        # 2. Execute extraction parsing function directly
        extracted_bits_string = extract_xyz(
            img, 
            x_bits=self.x_bits, 
            y_bits=self.y_bits, 
            z_bits=self.z_bits
        )
        
        # 3. Clean fallback normalization check
        if message_length is not None:
            return extracted_bits_string[:message_length]
            
        return extracted_bits_string

    def get_name(self):
        """
        Must return the matching configuration string expected 
        by your automated LSBAnalyzer pipeline class branches.
        """
        return "lsbCannySobel"

    @staticmethod
    def _bits_to_text(bits_string):
        """Convert standard framework binary bit stream string safely to characters."""
        message = []
        for i in range(0, len(bits_string), 8):
            byte = bits_string[i:i+8]
            if len(byte) == 8:
                message.append(chr(int(byte, 2)))
        return ''.join(message)