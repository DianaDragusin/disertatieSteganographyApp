import cv2
import numpy as np

def discrimination_vectorized(groups):
    """groups shape: (num_groups, group_size)"""
    return np.sum(np.abs(np.diff(groups.astype(np.int32), axis=1)), axis=1)

def apply_mask(group, mask):
    result = group.copy().astype(np.int32)
    for i, m in enumerate(mask):
        if m == 1:
            result[i] = result[i] ^ 1
        elif m == -1:
            if result[i] == 0:
                pass
            elif result[i] % 2 == 1:
                result[i] += 1
            else:
                result[i] -= 1
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_mask_vectorized(groups, mask):
    result = groups.copy().astype(np.int32)
    for i, m in enumerate(mask):
        if m == 1:
            result[:, i] = result[:, i] ^ 1
        elif m == -1:
            col = result[:, i]
            # odd -> +1, even -> -1 (except 0 stays)
            new_col = np.where(col == 0, 0,
                      np.where(col % 2 == 1, col + 1, col - 1))
            result[:, i] = new_col
            # safety if a math operation pushed a pixel value to -1 or 256, clip forces it back into the valid $0-255$ range
    return np.clip(result, 0, 255).astype(np.uint8)


def rs_analysis(channel, group_size=4):
    flat = channel.flatten()
    N = len(flat)
    num_groups = N // group_size

    # Reshape into groups all at once — no Python loop
    groups = flat[:num_groups * group_size].reshape(num_groups, group_size)

    mask     = np.array([0, 1, 1, 0])
    neg_mask = -mask

    f_orig = discrimination_vectorized(groups)
    f_M    = discrimination_vectorized(apply_mask_vectorized(groups, mask))
    f_nM   = discrimination_vectorized(apply_mask_vectorized(groups, neg_mask))

    RM  = np.sum(f_M  > f_orig) / num_groups
    SM  = np.sum(f_M  < f_orig) / num_groups
    RnM = np.sum(f_nM > f_orig) / num_groups
    SnM = np.sum(f_nM < f_orig) / num_groups

    denom = 2 * (RnM + SnM)
    if abs(denom) < 1e-10:
        p = 0.0
    else:
        p = (abs(RnM - RM)+ abs(SnM - SM)) / denom

    return p, RM, SM, RnM, SnM

    
def rs_analysis_flattened(image, group_size):
    flat = image.flatten()
    N = len(flat)
    num_groups = N // group_size

    # Reshape into groups all at once — no Python loop
    groups = flat[:num_groups * group_size].reshape(num_groups, group_size)

    mask     = np.array([0, 1, 1, 0])
    neg_mask = -mask

    f_orig = discrimination_vectorized(groups)
    f_M    = discrimination_vectorized(apply_mask_vectorized(groups, mask))
    f_nM   = discrimination_vectorized(apply_mask_vectorized(groups, neg_mask))

    RM  = np.sum(f_M  > f_orig) / num_groups
    SM  = np.sum(f_M  < f_orig) / num_groups
    RnM = np.sum(f_nM > f_orig) / num_groups
    SnM = np.sum(f_nM < f_orig) / num_groups

    denom = 2 * (RnM + SnM)
    if abs(denom) < 1e-10:
        p = 0.0
    else:
        p = (abs(RnM - RM) + abs(SnM - SM)) / denom

    return p, RM, SM, RnM, SnM