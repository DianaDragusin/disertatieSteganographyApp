import os
from PIL import Image
from readingFromFile import get_payload_bits_by_kb

def process_stego_dataset(image_folder, type_folder, place, payload_list, secret_file_path, strategy, stego_root, channel_idx=None):
    """
    Loops through images and payloads to create stego images using a specified embedding strategy.
    Natively supports variable keyword arguments to adapt to different architecture standards.
    """
    # ── FIXED: Define the missing variable by getting the name from the strategy object ──
    strategy_name = strategy.get_name()

    # ── Map the channel index to a readable folder name ──
    channel_map = {0: "channel_blue", 1: "channel_green", 2: "channel_red"}
    # Default to "channel_default" if no channel index is provided (e.g., for PVD or general LSB)
    channel_folder = channel_map.get(channel_idx, "channel_default")
    
    # ── FIXED: Now strategy_name works flawlessly on this line! ──
    base_output = os.path.join(stego_root, f"{strategy_name}_embeddings", place, channel_folder, type_folder)
    
    if not os.path.exists(image_folder):
        print(f"Source folder {image_folder} does not exist.")
        return

    for img_name in os.listdir(image_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, img_name)
            
            with Image.open(img_path) as img:
                width, height = img.size
                max_cap = width * height * 3
            
            for p in payload_list:
                bits_to_hide = get_payload_bits_by_kb(max_cap, p, secret_file_path)
                
                if bits_to_hide is None or len(bits_to_hide) == 0:
                    print(f"Warning: No bits to hide for {p}KB")
                    continue
                
                try:
                    subfolder_name = str(p)
                    final_dir = os.path.join(base_output, subfolder_name)
                    os.makedirs(final_dir, exist_ok=True)
                    
                    print(f"[{strategy_name.upper()}] Embedding {len(bits_to_hide)} bits into {img_name} at {p}KB...")
                    
                    # Package keyword arguments dynamically based on configuration requirements
                    kwargs = {}
                    if channel_idx is not None:
                        kwargs['stego_root'] = channel_idx
                    else:
                        kwargs['channel_idx'] = 0  # Default to Blue channel if None is explicitly passed
                    
                    kwargs['stego_root'] = stego_root  # Pass the root for potential key saving in LSBMR

                    # All strategy types are safely routed through the standard entry point
                    stego_output = strategy.embed(img_path, bits_to_hide, **kwargs)
                    
                    # Safeguard: If a strategy returns a tuple, unpack only the image object
                    if isinstance(stego_output, tuple):
                        stego_image = stego_output[0]
                    else:
                        stego_image = stego_output
                    
                    # Save the result losslessly
                    output_file_path = os.path.join(final_dir, img_name)
                    stego_image.save(output_file_path)
                    print(f"  ✓ Saved to {output_file_path}")
                    
                except Exception as e:
                    print(f"  ✗ Error embedding {img_name} at {p}KB: {str(e)}")
                    continue

    print(f"[{strategy_name.upper()}] Processing complete for {type_folder} images")