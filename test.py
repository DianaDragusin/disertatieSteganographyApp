import numpy as np
import cv2
import os

def get_edge_capacity_kb(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Canny + Sobel edge detection
    canny = cv2.Canny(img_gray, 100, 200)
    sx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = (np.sqrt(sx**2 + sy**2) > 50).astype(np.uint8) * 255
    union = cv2.bitwise_or(canny, sobel)
    
    edge_pixels = int(np.sum(union > 0))
    edge_kb = edge_pixels // 8 // 1024  # Convert bits to KB
    return edge_kb

if __name__ == "__main__":
    ai_folder = r"C:\Users\Diana\Desktop\disertatieSteganography\images\outdoor\OutdoorAIGenerated1792x2400"
    nat_folder = r"C:\Users\Diana\Desktop\disertatieSteganography\images\outdoor\OutdoorNatural1792x2400"

    ai_capacities = []
    nat_capacities = []

    header = f"{'ID':<4} | {'AI Edge (KB)':<15} | {'NAT Edge (KB)':<15}"
    print("\n" + "="*40)
    print(header)
    print("-" * 40)

    for i in range(1, len(os.listdir(ai_folder)) + 1):  # Assuming images are named 1.png, 2.png, ..., 100.png
        ai_path = os.path.join(ai_folder, f"{i}.png")
        nat_path = os.path.join(nat_folder, f"{i}.png")

        ai_cap = get_edge_capacity_kb(ai_path)
        nat_cap = get_edge_capacity_kb(nat_path)

        if ai_cap is not None: ai_capacities.append(ai_cap)
        if nat_cap is not None: nat_capacities.append(nat_cap)

        print(f"{i:<4} | {str(ai_cap) + ' KB':<15} | {str(nat_cap) + ' KB':<15}")

    # --- SUMMARY CALCULATIONS ---
    print("\n" + "="*45)
    print("FINAL STATISTICS: EDGE CAPACITY (KB)")
    print("="*45)
    
    stats_header = f"{'Metric':<10} | {'AI Images':<15} | {'Natural Images':<15}"
    print(stats_header)
    print("-" * 45)

    if ai_capacities and nat_capacities:
        metrics = [
            ("Mean", np.mean(ai_capacities), np.mean(nat_capacities)),
            ("Max", np.max(ai_capacities), np.max(nat_capacities)),
            ("Min", np.min(ai_capacities), np.min(nat_capacities)),
            ("Std Dev", np.std(ai_capacities), np.std(nat_capacities))
        ]

        for name, ai_val, nat_val in metrics:
            print(f"{name:<10} | {ai_val:<15.2f} | {nat_val:<15.2f}")
    
    print("="*45)
    print(f"Max Capacity (KB): {1792*2400*3//8//1024}")  # Max capacity if all pixels were used (for reference)
    print(f"Max Capacity per LSB (KB): {1792*2400//8//1024}")  # Max capacity if all pixels were used (for reference)