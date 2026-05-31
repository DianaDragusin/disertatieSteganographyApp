from strategy.embedding_strategy import EmbeddingStrategy
from PIL import Image
import numpy as np
import json
import cv2
import os

# Import the precise paper functions from your local file (e.g., lsbmr.py)
from lsbmr import embed, extract, bits_to_message


class LSBMRStrategy(EmbeddingStrategy):
    """
    Adaptive LSBMR Steganography Strategy implementing Mungmode et al.'s
    Threshold-Value Region Selection method with tracking persistence.
    """

    def __init__(self):
        super().__init__()
        # In-memory database repository dictionary map config
        self.shared_keys_db = {}
        # Last coordinate key produced by embed() — used by the in-memory GUI flow.
        self._last_coordinate_key = None

    def embed(self, image_path, message_bits_str, channel_idx=0, **kwargs):
        """
        Embeds message bits and automatically serializes the tracking coordinates
        to a physical file alongside the image path.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        message_bits_list = [int(b) for b in message_bits_str]
        message_text = bits_to_message(message_bits_list)

        stego_array, total_bits, secret_coordinate_key = embed(img, message_text, channel_idx=channel_idx)

        # ── Stash the key on the instance so the facade can pick it up ──
        self._last_coordinate_key = secret_coordinate_key

        # ── Build a composite tracking key for memory ──
        filename = os.path.basename(image_path)
        category = "ai" if "ai" in image_path.lower() else "natural"
        composite_id = f"{category}_{filename}"
        payload_bits_length = len(message_bits_str)

        if composite_id not in self.shared_keys_db:
            self.shared_keys_db[composite_id] = {}

        self.shared_keys_db[composite_id][payload_bits_length] = secret_coordinate_key

        # ── FILE BACKED PERSISTENCE: Save the key to disk based on context ──
        # Guess the stego path location dynamically based on payload sizes or structure passed in kwargs
        stego_folder = kwargs.get('stego_root')
        if stego_folder:
            key_filename = f"{os.path.splitext(filename)[0]}_key.json"
            key_save_path = os.path.join(stego_folder, key_filename)
            os.makedirs(stego_folder, exist_ok=True)
            with open(key_save_path, "w", encoding="utf-8") as key_file:
                json.dump(secret_coordinate_key, key_file)

        stego_img = Image.fromarray(cv2.cvtColor(stego_array, cv2.COLOR_BGR2RGB))
        return stego_img

    def extract(self, image_path, message_length=None, channel_idx=0, coordinate_key=None, **kwargs):
        """
        Extracts message bits.

        Lookup priority:
        1. coordinate_key kwarg (used by the in-memory GUI flow)
        2. RAM cache keyed by composite_id (same-process batch use)
        3. JSON sidecar file next to the stego image (batch pipeline)
        """
        if message_length is None:
            raise ValueError("message_length parameter required for extraction.")

        print(f"\n[LSBMR.extract DEBUG]")
        print(f"  image_path = {image_path}")
        print(f"  message_length = {message_length}")
        print(f"  coordinate_key is None: {coordinate_key is None}")
        print(f"  kwargs received: {list(kwargs.keys())}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load stego image: {image_path}")

        # ── Priority 1: caller supplied the key directly ──
        if coordinate_key is not None:
            current_key = coordinate_key
        else:
            current_key = None

            filename = os.path.basename(image_path)
            category = "ai" if "ai" in image_path.lower() else "natural"
            composite_id = f"{category}_{filename}"

            # Priority 2: RAM cache (same process as embed)
            current_key = self.shared_keys_db.get(composite_id, {}).get(message_length)

            # Priority 3: JSON sidecar next to the stego file
            if current_key is None:
                stego_dir = os.path.dirname(image_path)
                key_filename = f"{os.path.splitext(filename)[0]}_key.json"
                potential_key_path = os.path.join(stego_dir, key_filename)

                if os.path.exists(potential_key_path):
                    with open(potential_key_path, "r", encoding="utf-8") as key_file:
                        current_key = json.load(key_file)

                        if composite_id not in self.shared_keys_db:
                            self.shared_keys_db[composite_id] = {}
                        self.shared_keys_db[composite_id][message_length] = current_key

            if current_key is None:
                raise ValueError(
                    f"Secret Coordinate Key missing for composite mapping lookup: "
                    f"{composite_id} at length {message_length} bits."
                )

        extracted_bits_list = extract(img, message_length, current_key, channel_idx=channel_idx)
        return "".join(map(str, extracted_bits_list))

    def get_name(self):
        return "lsbmr"