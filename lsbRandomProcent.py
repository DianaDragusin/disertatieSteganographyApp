import numpy as np
import cv2
import random

def message_to_bits(message: str):
    bits = []
    for char in message:
        b = format(ord(char), '08b')
        bits.extend([int(bit) for bit in b])
    return bits

def bits_to_message(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

def generate_intervals_spatial(message_length, cover_size, seed):
    random.seed(seed)
    intervals = []
    max_first = cover_size // message_length
    j = random.randint(0, max_first)
    intervals.append(j)
    for i in range(1, message_length):
        remaining_positions = cover_size - j - 1
        remaining_bits = message_length - i
        if remaining_bits <= 0:
            break
        max_step = remaining_positions // remaining_bits
        if max_step <= 0:
            raise ValueError("Not enough space to embed message")
        step = random.randint(1, max_step)
        intervals.append(step)
        j += step
    return intervals

def embed_random_lsb_spatial(image, message, key):
    flat = image.flatten()
    N = len(flat)
    bits = message_to_bits(message)
    m_len = len(bits)
    intervals = generate_intervals_spatial(m_len, N, key)
    final_position = intervals[0] + sum(intervals[1:])
    if final_position >= N:
        raise ValueError("Message too long for cover image — aborting")
    stego = flat.copy()
    j = intervals[0]
    for i in range(m_len):
        stego[j] = (stego[j] & 0xFE) | bits[i]
        if i < m_len - 1:
            j += intervals[i + 1]
    return stego.reshape(image.shape)

def extract_random_lsb_spatial(stego_image, message_length, key):
    flat = stego_image.flatten()
    N = len(flat)
    intervals = generate_intervals_spatial(message_length, N, key)
    bits = []
    j = intervals[0]
    for i in range(message_length):
        bits.append(flat[j] & 1)
        if i < message_length - 1:
            j += intervals[i + 1]
    return bits_to_message(bits)

def generate_random_message(num_bits):
    num_chars = num_bits // 8
    chars = [chr(random.randint(32, 126)) for _ in range(num_chars)]
    return ''.join(chars)


def generate_random_bits(num_bits):
    num_bytes = num_bits // 8
    return np.frombuffer(os.urandom(num_bytes), dtype=np.uint8)