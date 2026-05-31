"""Configuration for steganography application."""
from pathlib import Path
from typing import Dict, Literal

# Base directory
BASE_DIR = Path(r"C:\Users\Diana\Desktop\disertatieSteganographyApp")

# Image folder structure: {scene_type: {image_type: path}}
IMAGE_FOLDERS: Dict[Literal["indoor", "outdoor"], Dict[Literal["ai", "natural"], Path]] = {
    "indoor": {
        "ai": BASE_DIR / "images" / "indoor" / "pozeTelefonAI1792x2400",
        "natural": BASE_DIR / "images" / "indoor" / "pozeTelefonPngRealResized1792x2400",
    },
    "outdoor": {
        "ai": BASE_DIR / "images" / "outdoor" / "OutdoorAIGenerated1792x2400",
        "natural": BASE_DIR / "images" / "outdoor" / "OutdoorNatural1792x2400",
    },
}

# Secret payload file
SECRET_FILE = BASE_DIR / "secret.txt"

# Stego output root
STEGO_ROOT = BASE_DIR / "embeddings"

# Secret key for LSB Random Spatial
SECRET_KEY = "BlueAvatarlife123"

# Number of images per scene type
NUM_IMAGES = 20

# Supported image extensions (order matters for discovery)
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

PAYLOAD_LIST = [1, 5, 10, 25, 50, 100, 200, 300, 500]
