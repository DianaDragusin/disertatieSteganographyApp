# UNUSED FILE — superseded by strategy/lsb_Canny_Sobel2.py which is registered in core/strategy_registry.py
# from strategy.embedding_strategy import EmbeddingStrategy
# from PIL import Image
# import numpy as np
# import cv2
# from lsb_Canny_Sobel import embed_xyz, extract_xyz

# class LSBCannySobelStrategy_(EmbeddingStrategy):
#     """LSB (Least Significant Bit) embedding strategy."""
#
#     def __init__(self, key="default_key"):
#         self.key = key
#
#     def embed(self, image_path, message_bits, **kwargs):
#         # Load image
#         img = cv2.imread(image_path)
#         if img is None:
#             raise ValueError(f"Could not load image: {image_path}")
#
#         # message_bits is already a string of '0' and '1' from your pipeline
#         stego_array, bits_count = embed_xyz(img, message_bits)
#
#         # Convert BGR (OpenCV) to RGB for PIL Image
#         stego_img = Image.fromarray(cv2.cvtColor(stego_array, cv2.COLOR_BGR2RGB))
#         return stego_img
#
#     def extract(self, image_path, message_length=None, **kwargs):
#         if message_length is None:
#             raise ValueError("message_length required")
#
#         # Load image - cv2.imread returns BGR
#         stego_img = cv2.imread(image_path)
#         if stego_img is None:
#             raise ValueError(f"Could not load: {image_path}")
#
#         # The image was saved as RGB by PIL, but cv2.imread treats it as BGR
#         # So we need to convert BGR back to RGB to get the original color order
#         stego_img = cv2.cvtColor(stego_img, cv2.COLOR_BGR2RGB)
#
#         # Extract the exact number of bits
#         bits = extract_xyz(stego_img, message_length)
#         return bits

#     def get_name(self):
#         return "lsbCannySobel"
