import numpy as np
import cv2

# ==========================================
# 1. UTILITY AND BIT CONVERSION FUNCTIONS
# ==========================================

def message_to_bits(message: str) -> list:
    """Converts text string into a list of binary integers (0s and 1s)."""
    bits = []
    for char in message:
        b = format(ord(char), '08b')
        bits.extend([int(bit) for bit in b])
    return bits

def bits_to_message(bits: list) -> str:
    """Converts a list of bits back into a readable text string."""
    chars = []
    bits = [int(b) for b in bits]
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

def lsb(val: int) -> int:
    """Returns the Least Significant Bit of an integer value."""
    return int(val) & 1

def f_relation(x_i: int, x_i1: int) -> int:
    """The paper's core floor-relationship function (Equation 6)."""
    return lsb((int(x_i) // 2) + int(x_i1))


# ==========================================
# 2. FIXED MATHEMATICAL LSBMR ENGINE
# ==========================================

def lsbmr_embed_pair(x_i, x_i1, m_i, m_i1):
    """
    Executes the 4 structural adjustment cases defined in Section 6.

    FIX — Case 3 & 4: All ±1 operations are now clamped to [0, 255]
    so uint8 never wraps. The LSB parity after clamping is verified
    explicitly and the adjustment direction is flipped when needed.
    """
    x_i  = int(x_i)
    x_i1 = int(x_i1)

    # Case 1: Perfect natural match
    if lsb(x_i) == m_i and f_relation(x_i, x_i1) == m_i1:
        return x_i, x_i1

    # Case 2: LSB matches, function mismatches — adjust x_i1 only
    if lsb(x_i) == m_i and f_relation(x_i, x_i1) != m_i1:
        if x_i1 == 0:
            delta = 1
        elif x_i1 == 255:
            delta = -1
        else:
            delta = np.random.choice([1, -1])
        return x_i, x_i1 + delta

    # Case 3: LSB mismatches, function matches — flip x_i's LSB only.
    # Adding 1 to an even number and subtracting 1 from an odd number
    # both flip the LSB while keeping floor(x_i/2) identical, which
    # preserves f_relation.  Clamp to stay inside [0, 255].
    if lsb(x_i) != m_i and f_relation(x_i, x_i1) == m_i1:
        if lsb(x_i) == 0:           # even → need odd → add 1
            new_xi = min(x_i + 1, 255)
        else:                        # odd  → need even → subtract 1
            new_xi = max(x_i - 1, 0)
        # After clamping, verify the LSB is actually what we want.
        # If clamping reversed the parity, flip the other direction.
        if lsb(new_xi) != m_i:
            new_xi = x_i + 1 if lsb(x_i) == 0 else x_i - 1
        return new_xi, x_i1

    # Case 4: Complete mismatch — adjust x_i so BOTH lsb and f_relation flip.
    # Subtracting 1 from an even value changes the LSB (0→1) AND changes
    # floor(x_i/2) (e.g. 4//2=2 vs 3//2=1), which toggles f_relation.
    # Adding 1 to an odd value does the same.  Clamp and re-check as above.
    # lsb(x_i) != m_i and f_relation(x_i, x_i1) != m_i1
    if lsb(x_i) == 0:               # even → subtract 1 (→ odd, new floor)
        new_xi = x_i - 1 if x_i > 0 else x_i + 1
    else:                            # odd  → add 1 (→ even, new floor)
        new_xi = x_i + 1 if x_i < 255 else x_i - 1

    # Clamp and verify both conditions are satisfied; if not, try the
    # opposite direction (boundary corner cases such as x_i=0 or x_i=255).
    new_xi = max(0, min(255, new_xi))
    if lsb(new_xi) != m_i or f_relation(new_xi, x_i1) != m_i1:
        # Try the other direction
        alt = x_i + 1 if lsb(x_i) == 0 else x_i - 1
        alt = max(0, min(255, alt))
        if lsb(alt) == m_i and f_relation(alt, x_i1) == m_i1:
            new_xi = alt
        # If neither works (extremely rare boundary), fall back to adjusting
        # x_i1 to fix f_relation after setting the LSB.
        else:
            # First fix LSB of x_i
            if lsb(x_i) == 0:
                new_xi = min(x_i + 1, 255)
            else:
                new_xi = max(x_i - 1, 0)
            # Now fix f_relation by nudging x_i1
            if f_relation(new_xi, x_i1) != m_i1:
                delta = 1 if x_i1 < 255 else -1
                x_i1 = x_i1 + delta

    return new_xi, x_i1


# ==========================================
# 3. THE PAPER'S ADAPTIVE THRESHOLD CALCULATOR
# ==========================================

def calculate_paper_threshold(channel_data, required_pairs: int) -> int:
    """
    Finds the highest possible raw intensity difference threshold between
    adjacent pixel pairs that yields enough blocks to fit the payload.
    """
    h, w = channel_data.shape

    diff_list = []
    for y in range(h):
        for x in range(0, w - 1, 2):
            diff = abs(int(channel_data[y, x]) - int(channel_data[y, x+1]))
            diff_list.append(diff)
    diff_array = np.array(diff_list)

    for T in range(255, -1, -1):
        available_pairs = np.sum(diff_array >= T)
        if available_pairs >= required_pairs:
            return T

    raise ValueError(f"Message too large! Needs {required_pairs} pairs.")


# ==========================================
# 4. EMBEDDING AND EXTRACTION ENGINES
# ==========================================

def embed(image, secret_message: str, channel_idx: int):
    """
    Paper's Data Embedding Workflow (Section 7.1).

    Returns:
        stego       – modified image array
        total_bits  – exact number of payload bits embedded
        shared_key_coordinates – ordered list of (y, x) pairs used
    """
    stego = image.copy().astype(np.uint8)

    message_bits   = message_to_bits(secret_message)
    total_bits     = len(message_bits)
    required_pairs = int(np.ceil(total_bits / 2))

    target_channel = stego[:, :, channel_idx]

    T = calculate_paper_threshold(target_channel, required_pairs)
    print(f"[Sender Log] Calculated Intensity Threshold T = {T}")

    h, w = target_channel.shape
    bit_idx = 0
    shared_key_coordinates = []

    for y in range(h):
        for x in range(0, w - 1, 2):
            if bit_idx >= total_bits:
                break

            diff = abs(int(target_channel[y, x]) - int(target_channel[y, x+1]))

            if diff >= T:
                shared_key_coordinates.append((y, x))

                x_i  = int(target_channel[y, x])
                x_i1 = int(target_channel[y, x+1])

                m_i  = message_bits[bit_idx]
                # FIX: only pass a real payload bit for m_i1; use 0 as padding
                # BUT do NOT let that padding bit be written into extracted_bits
                # later — total_bits is the guard for that (see extract()).
                m_i1 = message_bits[bit_idx + 1] if (bit_idx + 1) < total_bits else 0

                stego_xi, stego_xi1 = lsbmr_embed_pair(x_i, x_i1, m_i, m_i1)

                target_channel[y, x]     = np.uint8(stego_xi)
                target_channel[y, x + 1] = np.uint8(stego_xi1)

                bit_idx += 2

        if bit_idx >= total_bits:
            break

    stego[:, :, channel_idx] = target_channel
    return stego, total_bits, shared_key_coordinates


def extract(stego_image, total_bits_to_extract: int, shared_key_coordinates: list, channel_idx: int) -> list:
    """
    Paper's Data Extraction Workflow (Section 7.2).

    FIX: the loop appends m_i1 only when doing so would not exceed
    total_bits_to_extract.  Previously a padding 0 appended for odd-length
    payloads would corrupt the final character.
    """
    target_channel = stego_image[:, :, channel_idx]
    extracted_bits = []

    for y, x in shared_key_coordinates:
        if len(extracted_bits) >= total_bits_to_extract:
            break

        stego_xi  = int(target_channel[y, x])
        stego_xi1 = int(target_channel[y, x + 1])

        m_i  = lsb(stego_xi)
        m_i1 = f_relation(stego_xi, stego_xi1)

        extracted_bits.append(m_i)

        # FIX: guard — only append m_i1 if we still need more bits
        if len(extracted_bits) < total_bits_to_extract:
            extracted_bits.append(m_i1)

    return extracted_bits


# ==========================================
# 5. EXECUTION ENVIRONMENT
# ==========================================

if __name__ == "__main__":
    secret_message = "This is the EXACT methodology proposed in the original paper!"
    print(f"Original Text: {secret_message}")

    cover_image = cv2.imread(r"C:\Users\Diana\Desktop\disertatieSteganography\images\outdoor\OutdoorAIGenerated1792x2400\1.png")
    if cover_image is None:
        raise FileNotFoundError("Cover image not found — check the path.")

    # 0 = Blue, 1 = Green, 2 = Red
    TARGET_CHANNEL = 0

    # FIX: call the function by its actual name 'embed', not 'extract_paper_method'
    stego_image, total_bits, secret_coordinate_key = embed(
        cover_image, secret_message, TARGET_CHANNEL
    )

    # FIX: call 'extract', not the nonexistent 'extract_paper_method'
    extracted_bits = extract(
        stego_image, total_bits, secret_coordinate_key, TARGET_CHANNEL
    )

    decoded_message = bits_to_message(extracted_bits)
    print(f"Decoded Text:  {decoded_message}")

    if secret_message == decoded_message:
        print("\n✅ Success! Embedding and extraction are perfectly consistent.")
    else:
        print("\n❌ Mismatch detected.")
        # Diagnostic: show first differing bit position
        orig_bits = message_to_bits(secret_message)
        for i, (a, b) in enumerate(zip(orig_bits, extracted_bits)):
            if a != b:
                print(f"   First bit mismatch at index {i} "
                      f"(char {i//8} '{secret_message[i//8]}'): "
                      f"expected {a}, got {b}")
                break