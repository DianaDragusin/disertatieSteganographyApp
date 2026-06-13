from PIL import Image

# UNUSED — not imported anywhere in the active codebase
# def text_to_binary_payload(file_path):
#     """
#     Reads a text file and converts its content into a binary bitstream.
#     Each character is converted to an 8-bit string (padded with zeros).
#     """
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             text_data = file.read()
#         binary_message = ''.join(format(ord(char), '08b') for char in text_data)
#         return binary_message
#     except FileNotFoundError:
#         return "Error: The file was not found."
#     except Exception as e:
#         return f"An error occurred: {e}"


# UNUSED — only get_payload_bits_by_kb() is used; this percent-based variant is not imported anywhere
# def get_payload_bits(max_capacity_bits, payload_percent, file_path):
#     """
#     Calculates bit budget and reads file until the budget is hit.
#     Returns a string of '0's and '1's representing the payload.
#     """
#     allowed_bits = int(max_capacity_bits * (payload_percent / 100))
#     secret_bits = ""
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
#         for char in content:
#             char_bits = format(ord(char), '08b')
#             if len(secret_bits) + 8 <= allowed_bits:
#                 secret_bits += char_bits
#             else:
#                 break
#         return secret_bits
#     except FileNotFoundError:
#         print("Secret file not found.")
#         return ""


def get_payload_bits_by_kb(max_capacity_bits, target_kb, file_path):
    """
    Reads a specific number of KB from a file and converts to bits.
    """
    # 1. Convert KB to bits (1 KB = 1024 bytes * 8 bits)
    requested_bits = int(target_kb * 1024 * 8)
    
    # 2. Safety Check: Don't try to embed more than the image allows
    allowed_bits = min(requested_bits, max_capacity_bits)
    
    if requested_bits > max_capacity_bits:
        print(f"Warning: Requested {target_kb}KB is too large. Clipping to {max_capacity_bits//8//1024}KB.")

    secret_bits = ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        for char in content:
            char_bits = format(ord(char), '08b')
            
            # Stop once we hit our KB-based bit budget
            if len(secret_bits) + 8 <= allowed_bits:
                secret_bits += char_bits
            else:
                break 
                
        return secret_bits
    except FileNotFoundError:
        print("Secret file not found.")
        return ""

# UNUSED — only referenced in notNeeded/main2.py (deprecated)
# def check_secret_size(secret_message_path, image_path):
#     """Reads the secret file and returns its size in bits."""
#     try:
#         with open(secret_message_path, 'r', encoding='utf-8') as file:
#             text_data = file.read()
#         with Image.open(image_path) as img:
#             width, height = img.size
#             channels = len(img.getbands())
#             image_capacity_bits = width * height * channels
#         binary_message = ''.join(format(ord(char), '08b') for char in text_data)
#         print(f"Secret file size: {len(binary_message)} bits")
#         print(f"Image capacity: {image_capacity_bits} bits")
#         if len(binary_message) <= image_capacity_bits:
#             print("The secret file can be embedded in the image. ...")
#         else:
#             print("The secret file is too large to be embedded in the image. ...")
#     except FileNotFoundError:
#         return "Error: The file was not found."
#     except Exception as e:
#         return f"An error occurred: {e}"


# UNUSED — not imported anywhere in the active codebase
# def binary_to_text(binary_string):
#     """Reverse process: Converts a binary bitstream back into readable text."""
#     words = []
#     for i in range(0, len(binary_string), 8):
#         byte = binary_string[i:i+8]
#         words.append(chr(int(byte, 2)))
#     return ''.join(words)

