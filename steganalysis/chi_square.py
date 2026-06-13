import numpy as np
from scipy.stats import chi2
# import numpy as np  # duplicate
import pandas as pd
import cv2
from scipy.stats import chi2_contingency, chisquare


def chi_square_attack(image_channel):
    """Per-channel chi-square attack (single grayscale channel)."""
    counts = np.bincount(image_channel.flatten(), minlength=256)
    chi2_stat = 0.0
    used_pairs = 0

    for i in range(128):
        f_even = counts[2 * i]
        f_odd  = counts[2 * i + 1]
        expected = (f_even + f_odd) / 2.0
        if expected == 0:
            continue
        chi2_stat += (f_even - expected) ** 2 / expected
        used_pairs += 1

    df = used_pairs - 1
    p_value = 1 - chi2.cdf(chi2_stat, df=df)
    return chi2_stat, p_value


def calculate_chi2(original_image, stego_image):
    """
    Per-channel chi-square analysis for color or grayscale images.
    
    Args:
        original_image: numpy array (H x W) or (H x W x 3) for grayscale or BGR
        stego_image: numpy array same shape as original_image
        
    Returns:
        dict: Per-channel chi2 statistics {'B': {...}, 'G': {...}, 'R': {...}}
              for color, or {'Grayscale': {...}} for single channel
    """
    if original_image is None or stego_image is None:
        return None
    
    results = {}
    
    # Handle grayscale (2D) images
    if len(original_image.shape) == 2:
        chi2_stat, p_val = chi_square_attack(stego_image)
        results['Grayscale'] = {
            'chi2_statistic': chi2_stat,
            'p_value': p_val,
            'detected': p_val < 0.05  # typically 0.05 significance level
        }
    # Handle color (3D BGR) images
    elif len(original_image.shape) == 3 and original_image.shape[2] == 3:
        channel_names = ['Blue', 'Green', 'Red']
        for ch_idx, ch_name in enumerate(channel_names):
            chi2_stat, p_val = chi_square_attack(stego_image[:, :, ch_idx])
            results[ch_name] = {
                'chi2_statistic': chi2_stat,
                'p_value': p_val,
                'detected': p_val < 0.05
            }
    
    return results


   
        

def analyze_stego_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not find image.")
        return

    pixels = img.flatten()
    counts = np.bincount(pixels, minlength=256)

    observed = []
    expected = []

    for i in range(0, 256, 2):
        n_even = counts[i]
        n_odd  = counts[i + 1]
        total  = n_even + n_odd
        if total == 0:
            continue
        observed.append(n_even)
        expected.append(total / 2)

    observed = np.array(observed, dtype=float)
    expected = np.array(expected, dtype=float)

    # Fix: rescale expected so sums match exactly
    expected = expected * (observed.sum() / expected.sum())

    chi, p = chisquare(f_obs=observed, f_exp=expected)

    print(f"--- Steganography Analysis Results ---")
    print(f"Chi-Square Statistic: {chi:.4f}")
    print(f"P-Value:             {p:.6f}")

    if p > 0.05:
        print("\nConclusion: HIGH probability of LSB embedding detected.")
    else:
        print("\nConclusion: No strong evidence of LSB steganography.")

