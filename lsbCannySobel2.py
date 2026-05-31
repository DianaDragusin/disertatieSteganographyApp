import numpy as np
import cv2

def message_to_bits(message: str) -> list:
    """Converts a text string into a list of binary bits (0s and 1s)."""
    bits = []
    for char in message:
        b = format(ord(char), '08b')
        bits.extend([int(bit) for bit in b])
    return bits

def bits_to_message(bits: str) -> str:
    """Converts a string of bits back into a readable text string."""
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

def get_hybrid_xyz_masks(image):
    """
    Generates identical stable edge masks for both embedding and extraction 
    by clearing the lower 3 bits before executing Canny and Sobel filters.
    """
    # Clear the lower 3 bits (0xF8 = 11111000) so LSB changes don't cause mask drift
    stable_image = image.copy() & 0xF8  
    
    # Safely handle grayscale conversion based on channel count
    if len(stable_image.shape) == 3:
        gray = cv2.cvtColor(stable_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = stable_image.copy()
    
    # 1. Edge detection processing
    canny_mask = (cv2.Canny(gray, 100, 200) > 0).astype(np.uint8)
    
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mask = (sobel_mag > 50).astype(np.uint8)
    
    # 2. Divide into X, Y, Z regions
    z_mask = cv2.bitwise_and(canny_mask, sobel_mask)                        # High Texture (3 bits)
    y_mask = cv2.bitwise_xor(cv2.bitwise_or(canny_mask, sobel_mask), z_mask) # Medium Texture (2 bits)
    x_mask = cv2.bitwise_not(cv2.bitwise_or(canny_mask, sobel_mask))        # Smooth Zones (1 bit)
    
    return x_mask, y_mask, z_mask

def embed_xyz(image, secret_message: str, x_bits=1, y_bits=2, z_bits=3):
    """
    Embeds a message along with a 32-bit length header across all three 
    color channels using dynamic hybrid pixel masking.
    """
    # 1. Prepare payload (32-bit header containing length + actual message bits)
    message_bits = message_to_bits(secret_message)
    total_msg_bits = len(message_bits)
    header_bits = [int(bit) for bit in format(total_msg_bits, '032b')]
    payload_bits = header_bits + message_bits
    
    x_mask, y_mask, z_mask = get_hybrid_xyz_masks(image)
    stego = image.copy().astype(np.uint8)
    
    bit_idx = 0
    n_payload = len(payload_bits)
    
    # 2. Iterate through regions by priority order: Z -> Y -> X
    for mask, depth in [(z_mask, z_bits), (y_mask, y_bits), (x_mask, x_bits)]:
        coords = np.argwhere(mask > 0)
        for r, c in coords:
            # Multi-channel tracking (Blue=0, Green=1, Red=2)
            for channel in range(3):
                pixel_val = int(stego[r, c, channel])
                
                for b in range(depth):
                    if bit_idx < n_payload:
                        bit = int(payload_bits[bit_idx])
                        # Clear target bit and update value
                        pixel_val = (pixel_val & ~(1 << b)) | (bit << b)
                        bit_idx += 1
                
                stego[r, c, channel] = np.uint8(pixel_val)
                
                if bit_idx >= n_payload:
                    return stego, total_msg_bits
                
    return stego, total_msg_bits

def extract_xyz(stego_image, x_bits=1, y_bits=2, z_bits=3):
    """
    Dynamically extracts data by parsing the embedded 32-bit header 
    first to discover the precise end boundary of the secret message.
    """
    x_mask, y_mask, z_mask = get_hybrid_xyz_masks(stego_image)
    
    extracted_bits = ""
    bit_idx = 0
    total_bits_to_extract = 32  # Default to header parsing phase
    header_parsed = False
    
    for mask, depth in [(z_mask, z_bits), (y_mask, y_bits), (x_mask, x_bits)]:
        coords = np.argwhere(mask > 0)
        for r, c in coords:
            for channel in range(3):
                pixel_val = int(stego_image[r, c, channel])
                
                for b in range(depth):
                    if bit_idx < total_bits_to_extract:
                        bit = (pixel_val >> b) & 1
                        extracted_bits += str(bit)
                        bit_idx += 1
                        
                        # Dynamically extend the extraction goal once header is read
                        if bit_idx == 32 and not header_parsed:
                            message_length = int(extracted_bits, 2)
                            total_bits_to_extract = 32 + message_length
                            header_parsed = True
                
                if bit_idx >= total_bits_to_extract:
                    # Return only payload bits, discarding the 32-bit header length data
                    return extracted_bits[32:]
                
    return extracted_bits[32:]


# --- TESTING ENVIRONMENT ---
if __name__ == "__main__":
    # 1. Custom string definition
    secret_message = "Hello! This is a custom secret message embedded inside the image. Mask stabilization ensures perfect decoding!"
    print(f"Original Message: {secret_message}")
    
    # 2. Mock base image generation (Or load via cv2.imread('path.png'))
    cover_image = np.zeros((600, 600, 3), dtype=np.uint8)
    cv2.putText(cover_image, "Steganography Baseline", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # 3. Data embedding phase
    stego_image, total_bits = embed_xyz(cover_image, secret_message)
    print(f"Successfully hidden payload context length: {total_bits} bits.")

    # 4. Save to filesystem (PNG is mandatory for lossless stego tracking)
    output_filename = "stego_secured.png"
    cv2.imwrite(output_filename, stego_image)

    # 5. Reload testing simulated transfer
    received_stego_image = cv2.imread(output_filename)

    # 6. Extract data automatically without passing an external bit key length
    extracted_bits_str = extract_xyz(received_stego_image)

    # 7. Convert string back to text output
    decoded_message = bits_to_message(extracted_bits_str)
    print(f"Decoded Message:  {decoded_message}")
    
    # 8. Check validation integrity
    if secret_message == decoded_message:
        print("\n🎉 Success! The extracted message matches perfectly. No mask shifting occurred.")
    else:
        print("\n❌ Error: Integrity validation failed.")