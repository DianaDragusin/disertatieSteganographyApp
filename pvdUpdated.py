import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_func
import os



# --- Range Table ---

RANGES = [

    (0,   7,   3),

    (8,   15,  3),

    (16,  31,  4),

    (32,  63,  5),

    (64,  127, 6),

    (128, 255, 7)

]



def get_range(diff, ranges=RANGES):

    """Find which range bucket the difference falls into"""

    for (low, high, bits) in ranges:

        if low <= diff <= high:

            return low, high, bits

    return ranges[-1]  # fallback



def message_to_bits(message: str):

    return [int(b) for ch in message

            for b in format(ord(ch), '08b')]



def bits_to_message(bits):

    chars = []

    for i in range(0, len(bits) - 7, 8):

        chars.append(chr(int(''.join(map(str, bits[i:i+8])), 2)))

    return ''.join(chars)



# --- EMBED ---

def pvd_embed(image, message, ranges=RANGES):

    """

    PVD embedding on grayscale image.

    Processes consecutive pixel pairs horizontally.

    """

    if len(image.shape) == 3:

        # Convert to grayscale for classic PVD

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    else:

        gray = image.copy()

   

    stego = gray.copy().astype(np.int32)

    bits = message_to_bits(message)

    bit_idx = 0

    h, w = stego.shape

   

    for row in range(h):

        col = 0

        while col < w - 1 and bit_idx < len(bits):

            p1 = int(stego[row, col])

            p2 = int(stego[row, col + 1])

           

            diff = abs(p1 - p2)

            low, high, n_bits = get_range(diff, ranges)

           

            # Check if we have enough bits left

            bits_to_embed = bits[bit_idx:bit_idx + n_bits]

            if len(bits_to_embed) < n_bits:

                # Pad with zeros if needed

                bits_to_embed += [0] * (n_bits - len(bits_to_embed))

           

            # Convert bits to integer

            secret_val = int(''.join(map(str, bits_to_embed)), 2)

           

            # Compute new difference

            new_diff = low + secret_val

           

            # Adjust pixel values

            delta = new_diff - diff

            if p1 >= p2:

                p1_new = p1 + int(np.ceil(delta / 2))

                p2_new = p2 - int(np.floor(delta / 2))

            else:

                p1_new = p1 - int(np.ceil(delta / 2))

                p2_new = p2 + int(np.floor(delta / 2))

           

            # Clamp to valid range

            p1_new = np.clip(p1_new, 0, 255)

            p2_new = np.clip(p2_new, 0, 255)

           

            stego[row, col] = p1_new

            stego[row, col + 1] = p2_new

           

            bit_idx += n_bits

            col += 2  # move to next pair

   

    return stego.astype(np.uint8), bit_idx



# --- EXTRACT ---

def pvd_extract(stego_image, num_bits, ranges=RANGES):

    """

    PVD extraction — must use same range table as embedding.

    """

    if len(stego_image.shape) == 3:

        gray = cv2.cvtColor(stego_image, cv2.COLOR_BGR2GRAY)

    else:

        gray = stego_image.copy()

   

    bits = []

    h, w = gray.shape

   

    for row in range(h):

        col = 0

        while col < w - 1 and len(bits) < num_bits:

            p1 = int(gray[row, col])

            p2 = int(gray[row, col + 1])

           

            diff = abs(p1 - p2)

            low, high, n_bits = get_range(diff, ranges)

           

            # Extract hidden value

            secret_val = diff - low

            extracted = list(format(secret_val, f'0{n_bits}b'))

            bits.extend([int(b) for b in extracted])

           

            col += 2

   

    return bits_to_message(bits[:num_bits])





def get_max_pvd_capacity(image, ranges=RANGES):

    """Calculates the total number of bits this specific image can hold."""

    total_bits = 0

    h, w = image.shape

    for row in range(h):

        for col in range(0, w - 1, 2):

            p1, p2 = int(image[row, col]), int(image[row, col + 1])

            diff = abs(p1 - p2)

            _, _, n_bits = get_range(diff, ranges)

            total_bits += n_bits

    return total_bits

def run_percentage_test(image_path, percentages=[0.05, 0.10, 0.20, 0.50, 0.80, 1.0]):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Image not found at {image_path}")
        return
        
    max_bits = get_max_pvd_capacity(img)
    print("\n" + "="*75)
    print(f"IMAGE ANALYSIS: {image_path.split('\\')[-1]}")
    print(f"Total Theoretical Capacity: {max_bits} bits")
    print("="*75)
    
    results_table = []
    
    for p in percentages:
        target_bits = int(max_bits * p)
        
        # Calculate how many characters we need to hit the target bits
        n_chars = target_bits // 8
        if n_chars == 0: n_chars = 1 # Ensure at least some data
        
        # Generate random characters for the test
        secret_data = ''.join(chr(np.random.randint(32, 126)) for _ in range(n_chars))
        actual_target_bits = n_chars * 8
        
        # Perform embedding
        stego, bits_actually_embedded = pvd_embed(img, secret_data)

        # Perform extraction to verify correctness
        extracted_message = pvd_extract(stego, actual_target_bits)
        if extracted_message != secret_data:
            print(f"Warning: Extracted message does not match original at {p*100:.0f}% payload!")
        
        # Calculate Quality (PSNR)
        psnr, ssim = calculate_metrics(img, stego)
        #mse = np.mean((img.astype(float) - stego.astype(float)) ** 2)
       # psnr = 100 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
        
        # Perform Edge Analysis (Canny)
        analysis = analyze_edges(img, stego)
        # 3. Create the full path for the image
        filename = f"diffNat1_{int(p*100)}_percent.png"
        save_path = os.path.join(OUTPUT_FOLDER, filename)

        # 4. Write the file
        cv2.imwrite(save_path, analysis['diff_map'])

        #cv2.imwrite(f"diffAi_{p*100}_percent.png", analysis['diff_map'])
        
        results_table.append({
            "pct": f"{int(p*100)}%",
            "embedded": bits_actually_embedded,
            "psnr": psnr,
            "ssim": ssim,
            "edge": round(analysis['density_change'], 6)
        })

    # --- PRINT THE UPDATED TABLE ---
    header = f"{'Payload %':<10} | {'Embedded':<10} | {'PSNR (dB)':<12} | {'SSIM':<8} | {'Edge Change %'}"
    print(header)
    print("-" * len(header))
    
    for res in results_table:
        print(f"{res['pct']:<10} | {res['embedded']:<10} | {res['psnr']:<12} | {res['ssim']:<8} | {res['edge']}")

# --- Test ---

if __name__ == "__main__":

    # FIX 1: Added 'r' for raw string to fix Unicode error

    path = r"C:\Users\Diana\Desktop\disertatie\lsbTRy\pozeTelefonInPngREsized1800x2400\1.png"
    OUTPUT_FOLDER = r"C:\Users\Diana\Desktop\disertatie\lsbTRy\tryfulllsb\resultsPvdPlotCanny"

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

   

    if img is not None:

        #message = "PVD_Test_2026"

        #num_bits = len(message) * 8

       

        # 1. Embed and Extract

        #stego, bits_embedded = pvd_embed(img, message)

        #decoded = pvd_extract(stego, num_bits)

       

        #print(f"Bits embedded: {bits_embedded}")

        #print(f"Success: {message == decoded}")

        #print(f"Decoded: {decoded}")



        # FIX 2: Pass 'img' (the array) instead of a path variable that doesn't exist

        #results = analyze_edges(img, stego)

       

        #print(f"Edge Density Change: {results['density_change']:.4f}%")

        #cv2.imwrite("canny_diff_evidence.png", results['diff_map'])



        run_percentage_test(path)

    else:

        print("Error: Could not load image. Check the path!")