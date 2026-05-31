import numpy as np

import numpy as np
from scipy.ndimage import convolve

import cv2
import numpy as np

def calculate_ssim(f, g, window_size=11):
    # Determine if we need to process multiple channels
    if f.ndim == 3:
        # Calculate for each channel and average
        s1, s2, s3 = [calculate_ssim(f[:,:,i], g[:,:,i], window_size) for i in range(3)]
        return (s1 + s2 + s3) / 3.0

    # Ensure float64
    f = f.astype(np.float64)
    g = g.astype(np.float64)
    
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    # Using OpenCV's GaussianBlur is much faster than scipy.ndimage.convolve
    # It acts as the "sliding window" mean
    mu_f = cv2.GaussianBlur(f, (window_size, window_size), 1.5)
    mu_g = cv2.GaussianBlur(g, (window_size, window_size), 1.5)

    mu_f_sq = mu_f**2
    mu_g_sq = mu_g**2
    mu_fg = mu_f * mu_g

    # Calculate variances and covariance
    sigma_f_sq = cv2.GaussianBlur(f**2, (window_size, window_size), 1.5) - mu_f_sq
    sigma_g_sq = cv2.GaussianBlur(g**2, (window_size, window_size), 1.5) - mu_g_sq
    sigma_fg = cv2.GaussianBlur(f * g, (window_size, window_size), 1.5) - mu_fg

    # SSIM formula
    num = (2 * mu_fg + C1) * (2 * sigma_fg + C2)
    den = (mu_f_sq + mu_g_sq + C1) * (sigma_f_sq + sigma_g_sq + C2)

    ssim_map = num / den
    return np.mean(ssim_map)

def calculate_psnr(original, stego):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) between original and stego images.
    Matches the logic described in the dissertation.
    """
    mse = np.mean((original.astype(float) - stego.astype(float)) ** 2)
    
    if mse == 0:
        return 100.0  # Perfect match
    else:
        psnr = 10 * np.log10((255.0**2) / mse)
        return psnr


def calculate_mse(img1, img2):
    """Calculate Mean Squared Error between two image matrices."""
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)


def calculate_entropy(img):
    """Calculate the Shannon Entropy of an image matrix."""
    hist, _ = np.histogram(img, bins=256, range=(0, 256))
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]  # Filter out zero probabilities to avoid log2(0)
    return -np.sum(prob * np.log2(prob))


def calculate_bpp(kb_payload, width, height):
    """Calculate Bits Per Pixel (BPP) density representation."""
    total_bits = kb_payload * 1024 * 8
    return total_bits / (width * height)