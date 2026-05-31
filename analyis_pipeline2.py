import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from ssimAndPsnr import calculate_psnr, calculate_ssim, calculate_mse, calculate_entropy, calculate_bpp
from steganalysis.chi_square import calculate_chi2
from steganalysis.rs_analysis import rs_analysis
from stego_processor import process_stego_dataset


# ─────────────────────────────────────────────────────────────────────────────
#  Paths — edit only here
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_ROOT = r"C:\Users\Diana\Desktop\disertatieSteganographyApp\results"
PATCH_SIZE = 32


def _results_path(place, *parts):
    """Build a results sub-path that always includes the place folder."""
    return os.path.join(RESULTS_ROOT, place, *parts)


# ─────────────────────────────────────────────────────────────────────────────
#  Base class
# ─────────────────────────────────────────────────────────────────────────────
class StegoAnalyzer(ABC):

    def __init__(self, strategy_name, stego_root="embeddings"):
        self.strategy_name = strategy_name
        self.stego_root    = stego_root

    def _stego_folder(self, place, type_folder, channel_idx=None):
        """Return path to embedded images for a given place, channel, and type."""
        channel_map = {0: "channel_blue", 1: "channel_green", 2: "channel_red"}
        channel_folder = channel_map.get(channel_idx, "channel_default")
        
        return os.path.join(self.stego_root, 
                            f"{self.strategy_name}_embeddings", 
                            place, 
                            channel_folder, 
                            type_folder)

    # ── template method ──────────────────────────────────────────────────────
    def run_pipeline(self, folder_ai, folder_nat, place, payload_list,
                     strategy_obj, secret_file, channel_idx=None):
        """
        Full pipeline for one place ('indoor' or 'outdoor').

        Args:
            folder_ai    : folder with AI cover images for this place
            folder_nat   : folder with natural cover images for this place
            place        : "indoor" or "outdoor"
            payload_list : list of KB values
            strategy_obj : embedding strategy
            secret_file  : path to secret file
        """
        #self._embed(folder_ai, folder_nat, place, payload_list,
        #        strategy_obj, secret_file, channel_idx)

        ai_files  = sorted(f for f in os.listdir(folder_ai)
                           if f.lower().endswith((".png", ".jpg")))
        nat_files = sorted(f for f in os.listdir(folder_nat)
                           if f.lower().endswith((".png", ".jpg")))

        #self._verify_extraction(place, payload_list, strategy_obj, secret_file, folder_ai, folder_nat, channel_idx)

        #metrics = self._compute_metrics(folder_ai, folder_nat,ai_files, nat_files,place, payload_list, channel_idx)
        #self._save_csv(metrics, ai_files, nat_files, place, payload_list, channel_idx)
        self._save_summary_table(place, payload_list, channel_idx)
        #self._save_tables(metrics, ai_files, nat_files, place, payload_list)

        #self._run_chi_square_analysis(folder_ai, folder_nat, ai_files, nat_files, place, payload_list, channel_idx)
        #self._run_rs_steganalysis(ai_files, nat_files, place, payload_list, channel_idx)
            
        #self._run_channel_correlation_analysis(folder_ai, folder_nat, ai_files, nat_files, place, payload_list, channel_idx)

        #self._plot_chi_square(place, payload_list, channel_idx)   # chi-square (already working)
        #self._plot_rs_analysis(place, payload_list, channel_idx)         # NEW — RS side-by-side

        
        #self._plot_pair_diff(folder_ai, folder_nat, place, payload_list, channel_idx=channel_idx)   

    # ── step 0: embed ─────────────────────────────────────────────────────────
    def _embed(self, folder_ai, folder_nat, place, payload_list,
               strategy_obj, secret_file, channel_idx=None):

        print(f"\n=== Embedding [{self.strategy_name.upper()}] | {place} === | Channel: {channel_idx} ===")
        process_stego_dataset(folder_ai,  "ai",      place, payload_list, secret_file, strategy_obj, self.stego_root, channel_idx=channel_idx)
        process_stego_dataset(folder_nat, "natural", place, payload_list, secret_file, strategy_obj, self.stego_root, channel_idx=channel_idx)

    # ── step 1: compute metrics ───────────────────────────────────────────────
    def _compute_metrics(self, folder_ai, folder_nat,
                         ai_files, nat_files, place, payload_list, channel_idx=None):
        """
        Returns:
            metrics[idx][kb] = {
                "ai_psnr", "ai_ssim", "nat_psnr", "nat_ssim"
            }
        """
        metrics = {}
        is_pvd  = self.strategy_name == "pvdSequential"

        stego_ai  = self._stego_folder(place, "ai", channel_idx = channel_idx)
        stego_nat = self._stego_folder(place, "natural", channel_idx = channel_idx)

        for idx, (f_ai, f_nat) in enumerate(zip(ai_files, nat_files)):
            metrics[idx] = {}
            print(f"  Metrics {idx+1}/{len(ai_files)}: {f_ai} & {f_nat}")

            orig_ai  = cv2.imread(os.path.join(folder_ai,  f_ai))
            orig_nat = cv2.imread(os.path.join(folder_nat, f_nat))

            if is_pvd:
                orig_ai  = cv2.cvtColor(orig_ai,  cv2.COLOR_BGR2GRAY)
                orig_nat = cv2.cvtColor(orig_nat, cv2.COLOR_BGR2GRAY)
            
            h_ai, w_ai   = orig_ai.shape[:2]
            h_nat, w_nat = orig_nat.shape[:2]

            for kb in sorted(payload_list):
                flag = cv2.IMREAD_GRAYSCALE if is_pvd else cv2.IMREAD_COLOR
                s_ai  = cv2.imread(os.path.join(stego_ai,  str(kb), f_ai),  flag)
                s_nat = cv2.imread(os.path.join(stego_nat, str(kb), f_nat), flag)

                if s_ai is None or s_nat is None:
                    print(f"    [SKIP] payload {kb} KB")
                    continue

                metrics[idx][kb] = {
                    "ai_psnr":  calculate_psnr(orig_ai,  s_ai),
                    "ai_ssim":  calculate_ssim(orig_ai,  s_ai),
                    "nat_psnr": calculate_psnr(orig_nat, s_nat),
                    "nat_ssim": calculate_ssim(orig_nat, s_nat),
                    "ai_mse":   calculate_mse(orig_ai,  s_ai),
                    "nat_mse":  calculate_mse(orig_nat, s_nat),
                    "ai_entropy": calculate_entropy(s_ai),
                    "nat_entropy": calculate_entropy(s_nat),
                    "ai_bpp": calculate_bpp(kb, w_ai, h_ai),
                    "nat_bpp": calculate_bpp(kb, w_nat, h_nat)
                }

        return metrics

    # ── step 2a: save CSV (Excel-ready) ───────────────────────────────────────
    def _save_csv(self, metrics, ai_files, nat_files, place, payload_list, channel_idx=None):
        """
        Writes one CSV file per place+strategy, with one row per
        (image_index, image_type, kb_payload).

        Columns:
            Place, Strategy, ImageIndex, ImageFile, ImageType,
            Payload_KB, PSNR, SSIM
        """
        out_dir = _results_path(place, "csv")
        os.makedirs(out_dir, exist_ok=True)
        
        # Build channel string label for unique filename tracking
        channel_map = {0: "blue", 1: "green", 2: "red"}
        c_label = channel_map.get(channel_idx, "default")
        
        out_path = os.path.join(out_dir, f"{self.strategy_name}_{place}_{c_label}_metrics.csv")

        rows = []
        for idx, (f_ai, f_nat) in enumerate(zip(ai_files, nat_files)):
            for kb in sorted(payload_list):
                if kb not in metrics.get(idx, {}):
                    continue
                m = metrics[idx][kb]

                # AI row
                rows.append({
                    "Place":       place,
                    "Strategy":    self.strategy_name,
                    "Channel":     c_label,  # Add extra column tracking here if your fieldnames match!
                    "ImageFile":   f_ai,
                    "ImageType":   "AI",
                    "Payload_KB":  kb,
                    "PSNR":        round(m["ai_psnr"],  4),
                    "SSIM":        round(m["ai_ssim"],  6),
                    "MSE":         round(m["ai_mse"],   4),
                    "Entropy":     round(m["ai_entropy"], 4),
                    "BPP":         round(m["ai_bpp"],   4)
                })
                # Natural row
                rows.append({
                    "Place":       place,
                    "Strategy":    self.strategy_name,
                    "Channel":     c_label,  # Add extra column tracking here if your fieldnames match!
                    "ImageFile":   f_nat,
                    "ImageType":   "Natural",
                    "Payload_KB":  kb,
                    "PSNR":        round(m["nat_psnr"], 4),
                    "SSIM":        round(m["nat_ssim"], 6),
                    "MSE":         round(m["nat_mse"],   4),
                    "Entropy":     round(m["nat_entropy"], 4),
                    "BPP":         round(m["nat_bpp"],   4)
                })

        if not rows:
            print(f"  [CSV] No data to write for {place}/{self.strategy_name}")
            return

        fieldnames = ["Place", "Strategy", "Channel", "ImageFile",
                      "ImageType", "Payload_KB", "PSNR", "SSIM", "MSE", "Entropy", "BPP"]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"  [CSV] Saved → {out_path}  ({len(rows)} rows)")

    # ── step 2b: save visual tables (PNG) ────────────────────────────────────
    def _save_tables(self, metrics, ai_files, nat_files, place, payload_list):
        out_dir = _results_path(place, "tables", self.strategy_name)
        os.makedirs(out_dir, exist_ok=True)
        payloads    = sorted(payload_list)
        col_labels  = ["Payload (KB)", "AI PSNR", "Nat PSNR", "AI SSIM", "Nat SSIM", "AI MSE", "Nat MSE", "AI Entropy", "Nat Entropy", "AI BPP", "Nat BPP"]

        for idx, (f_ai, f_nat) in enumerate(zip(ai_files, nat_files)):
            rows = []
            for kb in payloads:
                if kb not in metrics.get(idx, {}):
                    continue
                m = metrics[idx][kb]
                rows.append([
                    f"{kb} KB",
                    f"{m['ai_psnr']:.2f}",
                    f"{m['nat_psnr']:.2f}",
                    f"{m['ai_ssim']:.4f}",
                    f"{m['nat_ssim']:.4f}",
                    f"{m['ai_mse']:.4f}",
                    f"{m['nat_mse']:.4f}",
                    f"{m['ai_entropy']:.4f}",
                    f"{m['nat_entropy']:.4f}",
                    f"{m['ai_bpp']:.4f}",
                    f"{m['nat_bpp']:.4f}"
                ])

            if not rows:
                continue

            fig, ax = plt.subplots(figsize=(10, 0.6 + len(rows) * 0.55))
            ax.axis("off")
            tbl = ax.table(cellText=rows, colLabels=col_labels,
                           cellLoc="center", loc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(11)
            tbl.scale(1.2, 2.2)

            for (row, col), cell in tbl.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#2c3e50")
                    cell.set_text_props(weight="bold", color="white")
                elif row % 2 == 0:
                    cell.set_facecolor("#ecf0f1")

            img_id = os.path.splitext(f_ai)[0]
            plt.title(f"{self.strategy_name.upper()} | {place} | Image {img_id}",
                      pad=12, fontsize=11, fontweight="bold")
            out_file = os.path.join(out_dir, f"image{idx+1}_metrics.png")
            plt.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close()

        print(f"  [Tables] Saved PNG tables → {out_dir}")

    # ── step 3: verify extraction ─────────────────────────────────────────────
    def _verify_extraction(self, place, payload_list, strategy_obj, secret_file,
                           folder_ai, folder_nat, channel_idx=None):
        """Verify that extraction works correctly for embedded images."""
        print(f"\n=== Verification [{self.strategy_name.upper()}] | {place} | Channel: {channel_idx} ===")
        
        # Read the original text file content directly
        with open(secret_file, 'r', encoding='utf-8') as f:
            secret_text = f.read()
        
        # Convert the full raw text source into a clean binary string of bits ("01001...")
        full_secret_bits = "".join(format(ord(char), '08b') for char in secret_text)
        
        ai_files  = sorted(f for f in os.listdir(folder_ai) if f.lower().endswith((".png", ".jpg")))
        nat_files = sorted(f for f in os.listdir(folder_nat) if f.lower().endswith((".png", ".jpg")))
        
        for category, files in [("ai", ai_files), ("natural", nat_files)]:
            stego_folder = self._stego_folder(place, category, channel_idx = channel_idx)
            
            for kb in sorted(payload_list):
                expected_bits_length = kb * 1024 * 8
                
                # Slice exactly how many bits this payload loop step expected
                expected_bit_string = full_secret_bits[:expected_bits_length]
                
                for f_img in files:
                    stego_path = os.path.normpath(os.path.join(stego_folder, str(kb), str(f_img)))
                    
                    if not os.path.exists(stego_path):
                        print(f"  [SKIP] File missing at path: {stego_path}")
                        continue
                    
                    # Pack keyword extraction arguments dynamically 
                    extra_args = {}
                    if channel_idx is not None:
                        extra_args['channel_idx'] = channel_idx

                    # Execute Extraction Process
                    try:
                        
                        extracted_bit_string = strategy_obj.extract(stego_path, expected_bits_length, **extra_args)
                        
                        # Compare directly via raw bit strings instead of converting back to bytes!
                        if extracted_bit_string == expected_bit_string:
                            print(f"  [PASS] {category.upper()} | {f_img} | Payload: {kb} KB matches.")
                        else:
                            print(f"  [FAIL!!] {category.upper()} | {f_img} | Payload: {kb} KB - Data mismatch!")
                            
                            # Simple debug breakdown for first asset failures
                            if kb == payload_list[0] and f_img == files[0]:
                                print(f"    [DEBUG] Expected length: {len(expected_bit_string)} bits")
                                print(f"    [DEBUG] Extracted length: {len(extracted_bit_string)} bits")
                                print(f"    [DEBUG] Expected prefix:  {expected_bit_string[:32]}")
                                print(f"    [DEBUG] Extracted prefix: {extracted_bit_string[:32]}\n")
                                
                    except Exception as e:
                        print(f"  [ERROR] Exception during extraction for {f_img} ({kb} KB): {str(e)}")
    
    # ── Chi-Square Steganalysis ────────────────────
    def _run_chi_square_analysis(self, folder_ai, folder_nat, ai_files, nat_files, place, payload_list, channel_idx=None):
        print(f"\n--- Running Chi-Square Attack Pipeline | {place} ---")
        out_dir = _results_path(place, "csv")
        os.makedirs(out_dir, exist_ok=True)
        
        channel_map = {0: "blue", 1: "green", 2: "red"}
        c_label = channel_map.get(channel_idx, "default")
        out_path = os.path.join(out_dir, f"{self.strategy_name}_{place}_{c_label}_chisquare.csv")
        
        stego_ai_root  = self._stego_folder(place, "ai", channel_idx=channel_idx)
        stego_nat_root = self._stego_folder(place, "natural", channel_idx=channel_idx)
        
        is_pvd = self.strategy_name == "pvdSequential"
        rows = []
        
        categories = [
            ("AI", ai_files, folder_ai, stego_ai_root), 
            ("Natural", nat_files, folder_nat, stego_nat_root)
        ]
        
        for img_type, files, cover_folder, stego_folder in categories:
            for f_img in files:
                orig_path = os.path.join(cover_folder, f_img)
                orig_img = cv2.imread(orig_path) # Read regular color first
                if orig_img is None:
                    continue

                # Align image structure exactly with your strategy type (just like _compute_metrics)
                if is_pvd:
                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

                for kb in sorted(payload_list):
                    stego_path = os.path.join(stego_folder, str(kb), f_img)
                    if not os.path.exists(stego_path):
                        continue
                    
                    flag = cv2.IMREAD_GRAYSCALE if is_pvd else cv2.IMREAD_COLOR
                    stego_img = cv2.imread(stego_path, flag)
                    if stego_img is None:
                        continue
                    
                    # Call your custom calculate_chi2 method
                    chi2_results = calculate_chi2(orig_img, stego_img)
                    
                    # DEBUG PRINTS: If it still skips, this tells you exactly why!
                    if chi2_results is None:
                        print(f"  [DEBUG CHI2] calculate_chi2 returned None for {f_img} ({kb} KB). Shapes -> Orig: {orig_img.shape if orig_img is not None else 'None'}, Stego: {stego_img.shape if stego_img is not None else 'None'}")
                        continue
                    
                    # Unpack the returned per-channel dictionary structure
                    for channel_name, data in chi2_results.items():
                        rows.append({
                            "Place": place,
                            "Strategy": self.strategy_name,
                            "ExecutionChannel": c_label,
                            "ImageFile": f_img,
                            "ImageType": img_type,
                            "Payload_KB": kb,
                            "AnalyzedChannel": channel_name,
                            "Chi2_Statistic": round(data['chi2_statistic'], 4),
                            "P_Value": round(data['p_value'], 6),
                            "Detected": "YES" if data['detected'] else "NO"
                        })
                        
        if rows:
            fieldnames = ["Place", "Strategy", "ExecutionChannel", "ImageFile", "ImageType", "Payload_KB", "AnalyzedChannel", "Chi2_Statistic", "P_Value", "Detected"]
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  [Chi-Square CSV] Saved → {out_path} ({len(rows)} rows)")
        else:
            print(f"  [WARNING] Chi-Square Pipeline finished, but 0 rows were generated. No CSV written.")


    # ── Separate Method 2: RS Analysis via rs_analysis ────────────────────────
    def _run_rs_steganalysis(self, ai_files, nat_files, place, payload_list, channel_idx=None):
        print(f"\n--- Running RS Steganalysis Pipeline | {place} ---")
        out_dir = _results_path(place, "csv")
        os.makedirs(out_dir, exist_ok=True)
        
        channel_map = {0: "blue", 1: "green", 2: "red"}
        c_label = channel_map.get(channel_idx, "default")
        out_path = os.path.join(out_dir, f"{self.strategy_name}_{place}_{c_label}_rsanalysis.csv")
        
        stego_ai  = self._stego_folder(place, "ai", channel_idx=channel_idx)
        stego_nat = self._stego_folder(place, "natural", channel_idx=channel_idx)
        
        rows = []
        categories = [("AI", ai_files, stego_ai), ("Natural", nat_files, stego_nat)]
        
        for img_type, files, folder in categories:
            for f_img in files:
                for kb in sorted(payload_list):
                    stego_path = os.path.join(folder, str(kb), f_img)
                    if not os.path.exists(stego_path):
                        continue
                    
                    stego_img = cv2.imread(stego_path, cv2.IMREAD_UNCHANGED)
                    if stego_img is None:
                        continue
                    
                    # Isolate target channel array for RS analysis input
                    if len(stego_img.shape) == 3:
                        # Fallback to grayscale if no single channel is declared explicitly
                        target_ch = stego_img[:, :, channel_idx] if channel_idx is not None else cv2.cvtColor(stego_img, cv2.COLOR_BGR2GRAY)
                    else:
                        target_ch = stego_img
                        
                    # Call your custom rs_analysis method directly
                    p_est, RM, SM, RnM, SnM = rs_analysis(target_ch, group_size=4)
                    
                    rows.append({
                        "Place": place,
                        "Strategy": self.strategy_name,
                        "Channel": c_label,
                        "ImageFile": f_img,
                        "ImageType": img_type,
                        "Payload_KB": kb,
                        "Estimated_Length_P": round(p_est, 6),
                        "RM": round(RM, 4),
                        "SM": round(SM, 4),
                        "RnM": round(RnM, 4),
                        "SnM": round(SnM, 4),
                    })
                    
        if rows:
            fieldnames = ["Place", "Strategy", "Channel", "ImageFile", "ImageType", "Payload_KB", "Estimated_Length_P", "RM", "SM", "RnM", "SnM"]
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  [RS Analysis CSV] Saved → {out_path} ({len(rows)} rows)")
    
    # ── Separate Method 3: Inter-Channel Pearson Correlation Analysis (Fixed) ──
    def _run_channel_correlation_analysis(self, folder_ai, folder_nat, ai_files, nat_files, place, payload_list, channel_idx=None):
        # 1. Strategy check: Skip grayscale-only pipelines like PVD
        if self.strategy_name == "pvdSequential":
            print(f"  [Channel Correlation] Skipping PVD (Grayscale only).")
            return

        print(f"\n--- Running Inter-Channel Correlation Pipeline | {place} ---")
        
        # FIXED: Added missing local imports for statistics and dataframe handling
        from scipy.stats import pearsonr
        import pandas as pd  

        out_dir = _results_path(place, "csv")
        plot_dir = _results_path(place, "plots")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        
        channel_map = {0: "blue", 1: "green", 2: "red"}
        c_label = channel_map.get(channel_idx, "default")
        out_path = os.path.join(out_dir, f"{self.strategy_name}_{place}_{c_label}_correlation.csv")
        
        stego_ai_root  = self._stego_folder(place, "ai", channel_idx=channel_idx)
        stego_nat_root = self._stego_folder(place, "natural", channel_idx=channel_idx)
        
        rows = []
        categories = [
            ("AI", ai_files, folder_ai, stego_ai_root), 
            ("Natural", nat_files, folder_nat, stego_nat_root)
        ]
        
        # Pairs to calculate: (Channel A Index, Channel B Index, Label Name)
        # Note: OpenCV reads as BGR -> B=0, G=1, R=2
        channel_pairs = [
            (2, 1, "R-G"),
            (2, 0, "R-B"),
            (1, 0, "G-B")
        ]

        for img_type, files, cover_folder, stego_folder in categories:
            for f_img in files:
                # Load Original Cover Image
                print(f"  Processing {img_type.upper()} | {f_img} | Channel: {c_label} | Payloads: {payload_list} KB")
                orig_img = cv2.imread(os.path.join(cover_folder, f_img), cv2.IMREAD_COLOR)
                if orig_img is None or len(orig_img.shape) != 3:
                    continue

                for kb in sorted(payload_list):
                    stego_path = os.path.join(stego_folder, str(kb), f_img)
                    if not os.path.exists(stego_path):
                        continue
                    
                    stego_img = cv2.imread(stego_path, cv2.IMREAD_COLOR)
                    if stego_img is None or len(stego_img.shape) != 3:
                        continue
                    
                    # Calculate Pearson Correlation Coefficient for each pair
                    for ch_a, ch_b, pair_name in channel_pairs:
                        # Cover vectors
                        orig_a = orig_img[:, :, ch_a].flatten()
                        orig_b = orig_img[:, :, ch_b].flatten()
                        r_cover, _ = pearsonr(orig_a, orig_b)

                        # Stego vectors
                        stego_a = stego_img[:, :, ch_a].flatten()
                        stego_b = stego_img[:, :, ch_b].flatten()
                        r_stego, _ = pearsonr(stego_a, stego_b)
                        
                        # Delta metric shows the structural damage caused by steganography
                        delta_r = r_stego - r_cover

                        rows.append({
                            "Place": place,
                            "Strategy": self.strategy_name,
                            "ExecutionChannel": c_label,
                            "ImageFile": f_img,
                            "ImageType": img_type,
                            "Payload_KB": kb,
                            "ChannelPair": pair_name,
                            "R_Cover": round(r_cover, 6),
                            "R_Stego": round(r_stego, 6),
                            "Delta_R": round(delta_r, 6)
                        })
                        
        if not rows:
            return

        # Save Granular CSV 
        fieldnames = ["Place", "Strategy", "ExecutionChannel", "ImageFile", "ImageType", "Payload_KB", "ChannelPair", "R_Cover", "R_Stego", "Delta_R"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  [Correlation CSV] Saved → {out_path}")

        # 2. Aggregated Plot Generation (One per channel pair)
        df_current = pd.DataFrame(rows)
        
        # Look for the opposite place's CSV to combine them dynamically
        opposite_place = "outdoor" if place == "indoor" else "indoor"
        opposite_csv = os.path.join(_results_path(opposite_place, "csv"), f"{self.strategy_name}_{opposite_place}_{c_label}_correlation.csv")
        
        if os.path.exists(opposite_csv):
            try:
                df_opposite = pd.read_csv(opposite_csv)
                df = pd.concat([df_current, df_opposite], ignore_index=True)
                print(f"  [Plotting] Successfully combined {place} and {opposite_place} data for plotting.")
            except Exception:
                df = df_current
        else:
            df = df_current

        payloads = sorted(payload_list)

        for pair_name in ["R-G", "R-B", "G-B"]:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
            
            for ax_idx, target_place in enumerate(["indoor", "outdoor"]):
                ax = axes[ax_idx]
                ax.set_title(f"{target_place.upper()} ENVIRONMENT", fontsize=12, fontweight='bold', pad=10)
                
                # Filter using the target_place loop index directly!
                df_place = df[(df["ChannelPair"] == pair_name) & (df["Place"] == target_place)]
                
                if df_place.empty:
                    ax.text(0.5, 0.5, f"Waiting for {target_place}\npipeline execution...", 
                            ha='center', va='center', transform=ax.transAxes, color='gray', fontsize=11)
                    continue

                for img_type, color, marker in [("AI", "#3498db", "o"), ("Natural", "#e67e22", "s")]:
                    df_type = df_place[df_place["ImageType"] == img_type]
                    
                    means = []
                    stds = []
                    
                    for kb in payloads:
                        sub = df_type[df_type["Payload_KB"] == kb]
                        if not sub.empty:
                            means.append(sub["Delta_R"].mean())
                            stds.append(sub["Delta_R"].std())
                        else:
                            means.append(0)
                            stds.append(0)
                    
                    means = np.array(means)
                    stds = np.array(stds)
                    
                    ax.plot(payloads, means, label=f"{img_type} (Mean)", color=color, marker=marker, linewidth=2)
                    ax.fill_between(payloads, means - stds, means + stds, color=color, alpha=0.15)
                
                ax.set_xlabel("Payload Size (KB)", fontsize=10)
                if ax_idx == 0:
                    ax.set_ylabel(r"Shift in Pearson Correlation ($\Delta$ R)", fontsize=10)
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend(loc="upper right" if target_place == "indoor" else "lower left") # customized to prevent clipping lines
            
            fig.suptitle(f"{self.strategy_name.upper()} | Channel Relationship Analysis ({pair_name})", 
                         fontsize=14, fontweight="bold", y=0.98)
            plt.tight_layout()
            
            # Save the plot inside the active folder path
            plot_path = os.path.join(plot_dir, f"{self.strategy_name}_{place}_{c_label}_{pair_name}_correlation_plot.png")
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close()

        # ── Separate Method 5: Visualizing PoV Equalization Trends ──────────────
        # ── Separate Method 6: RS Analysis side-by-side indoor/outdoor plot ───────────
    def _plot_rs_analysis(self, place, payload_list, channel_idx=None):
        print(f"\n--- Generating RS Analysis Plots | {place} ---")
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        channel_map = {0: "blue", 1: "green", 2: "red"}
        c_label = channel_map.get(channel_idx, "default")
        opposite_place = "outdoor" if place == "indoor" else "indoor"

        csv_dir_current  = _results_path(place, "csv")
        csv_dir_opposite = _results_path(opposite_place, "csv")
        plot_dir         = _results_path(place, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        dfs = []
        for target_env, target_dir in [(place, csv_dir_current), (opposite_place, csv_dir_opposite)]:
            if not os.path.exists(target_dir):
                continue
            for f in os.listdir(target_dir):
                if f.startswith(self.strategy_name) and f.endswith("_rsanalysis.csv"):
                    try:
                        df_part = pd.read_csv(os.path.join(target_dir, f))
                        dfs.append(df_part)
                        print(f"  [RS Loader] {target_env} → {f}")
                    except Exception:
                        pass

        if not dfs:
            print("  [Warning] No RS analysis CSV files found.")
            return

        df = pd.concat(dfs, ignore_index=True)
        payloads = sorted(payload_list)

        # Plot one figure per metric
        metrics_to_plot = [
            ("Estimated_Length_P", "Estimated embedding length (P̂)", "P̂")
        ]

        for col, full_name, y_label in metrics_to_plot:
            if col not in df.columns:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

            for ax_idx, target_place in enumerate(["indoor", "outdoor"]):
                ax = axes[ax_idx]
                ax.set_title(f"{target_place.upper()} ENVIRONMENT", fontsize=12,
                            fontweight="bold", pad=10)

                df_place = df[df["Place"] == target_place]

                if df_place.empty:
                    ax.text(0.5, 0.5, f"Waiting for {target_place}\ndata...",
                            ha="center", va="center", transform=ax.transAxes,
                            color="gray", fontsize=11)
                    continue

                for img_type, color, marker, ls in [
                    ("AI",      "#378ADD", "o", "-"),
                    ("Natural", "#BA7517", "s", "--"),
                ]:
                    df_type = df_place[df_place["ImageType"] == img_type]
                    means, stds = [], []
                    for kb in payloads:
                        sub = df_type[df_type["Payload_KB"] == kb]
                        means.append(sub[col].mean() if not sub.empty else np.nan)
                        stds.append(sub[col].std()  if not sub.empty else np.nan)

                    means = np.array(means)
                    stds  = np.array(stds)

                    if not np.all(np.isnan(means)):
                        ax.plot(payloads, means, label=f"{img_type} (mean)",
                                color=color, marker=marker, linestyle=ls, linewidth=2)
                        ax.fill_between(payloads,
                                        np.where(np.isnan(means - stds), np.nan, means - stds),
                                        np.where(np.isnan(means + stds), np.nan, means + stds),
                                        color=color, alpha=0.12)

                ax.set_xlabel("Payload size (KB)", fontsize=10)
                ax.set_ylabel(y_label, fontsize=10)
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(loc="upper left")

            fig.suptitle(
                f"{self.strategy_name.upper()} | RS Steganalysis — {full_name}",
                fontsize=14, fontweight="bold", y=0.98
            )
            plt.tight_layout()
            plot_path = os.path.join(
                plot_dir,
                f"{self.strategy_name}_{place}_{c_label}_rs_{col.lower()}_plot.png"
            )
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  [RS Plot] Saved → {plot_path}")

    # ── Separate Method 5: Dynamic & Adaptive PoV Equalization Trends ───────
    def _plot_chi_square(self, place, payload_list, channel_idx=None):
        print(f"\n--- Generating Dynamic PoV Equalization Plots | {place} ---")
        import os
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        channel_map = {0: "blue", 1: "green", 2: "red"}
        c_label = channel_map.get(channel_idx, "default")
        
        opposite_place = "outdoor" if place == "indoor" else "indoor"
        csv_dir_current = _results_path(place, "csv")
        csv_dir_opposite = _results_path(opposite_place, "csv")
        plot_dir = _results_path(place, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        dfs = []
        
        # Build absolute paths to check for the single combined default CSV structure
        default_current = os.path.join(csv_dir_current, f"{self.strategy_name}_{place}_default_chisquare.csv")
        default_opposite = os.path.join(csv_dir_opposite, f"{self.strategy_name}_{opposite_place}_default_chisquare.csv")
        
        # Condition check: Does a unified default file exist on disk?
        has_default_on_disk = os.path.exists(default_current) or os.path.exists(default_opposite)
        
        if c_label == "default" and has_default_on_disk:
            # Scenario A: Combined default exists -> Read strictly from the unified sheets
            if os.path.exists(default_current):
                dfs.append(pd.read_csv(default_current))
                print(f"  [Adaptive Loader] Ingesting combined default matrix for: {place}")
            if os.path.exists(default_opposite):
                try:
                    dfs.append(pd.read_csv(default_opposite))
                    print(f"  [Adaptive Loader] Ingesting combined default matrix for: {opposite_place}")
                except Exception:
                    pass
        else:
            # Scenario B: Individual channel run OR default file missing on disk -> Scan for separate RGB files
            print("  [Adaptive Loader] Default file not found or separate channels requested. Scanning individual RGB files...")
            for target_env, target_dir in [(place, csv_dir_current), (opposite_place, csv_dir_opposite)]:
                if not os.path.exists(target_dir):
                    continue
                for f in os.listdir(target_dir):
                    # Catch channel-specific chi-square files belonging to this strategy, excluding default duplicate logs
                    if f.startswith(self.strategy_name) and f.endswith("_chisquare.csv") and "_default_" not in f:
                        file_path = os.path.join(target_dir, f)
                        try:
                            dfs.append(pd.read_csv(file_path))
                            print(f"  [Adaptive Loader] Ingesting independent channel segment: {target_env} -> {f}")
                        except Exception:
                            pass

        if not dfs:
            print("  [Warning] No matching Chi-Square CSV files found to generate plots.")
            return

        # Combine all found sheets seamlessly into a single workspace dataframe
        df = pd.concat(dfs, ignore_index=True)
        payloads = sorted(payload_list)

        # ── DYNAMIC LAYOUT LOGIC ──
        # Check which environments actually populated into our workspace data
        available_places = df["Place"].unique()
        # Sort so 'indoor' is always positioned on the left column if both are present
        available_places = sorted(list(available_places), key=lambda x: 0 if x == "indoor" else 1)
        num_cols = len(available_places)

        # Extract unique channels detected across the combined data (Blue, Green, Red, or Grayscale)
        channels_to_plot = df["AnalyzedChannel"].unique()

        for ch_name in channels_to_plot:
            # Setup Figure dynamically: Width scales automatically (7 inches per active environment)
            # squeeze=False prevents matplotlib from crushing the 2D array grid mapping on 1x1 steps
            fig, axes = plt.subplots(1, num_cols, figsize=(7 * num_cols, 6), squeeze=False)

            for ax_idx, target_place in enumerate(available_places):
                ax = axes[0, ax_idx]
                ax.set_title(f"{target_place.upper()} ENVIRONMENT", fontsize=12, fontweight='bold', pad=10)

                # Filter data down to the specific environment column and color channel layer
                df_slice = df[(df["Place"] == target_place) & (df["AnalyzedChannel"] == ch_name)]

                for img_type, color, marker in [("AI", "#2ecc71", "o"), ("Natural", "#9b59b6", "s")]:
                    df_type = df_slice[df_slice["ImageType"] == img_type]
                    
                    means = []
                    stds = []

                    for kb in payloads:
                        sub = df_type[df_type["Payload_KB"] == kb]
                        if not sub.empty:
                            means.append(sub["Chi2_Statistic"].mean())
                            stds.append(sub["Chi2_Statistic"].std())
                        else:
                            means.append(np.nan)
                            stds.append(np.nan)

                    means = np.array(means)
                    stds = np.array(stds)

                    # Only plot curves if the calculated numerical array contains valid elements
                    if not np.all(np.isnan(means)):
                        ax.plot(payloads, means, label=f"{img_type} (Mean $\chi^2$)", color=color, marker=marker, linewidth=2)
                        ax.fill_between(payloads, means - stds, means + stds, color=color, alpha=0.15)

                ax.set_xlabel("Payload Size (KB)", fontsize=10)
                ax.set_ylabel("Chi-Square Asymmetry Statistic ($\chi^2$)", fontsize=10)
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend(loc="upper right")

            fig.suptitle(f"{self.strategy_name.upper()} | Pairs of Values Equalization ({ch_name} Channel)", 
                         fontsize=14, fontweight="bold", y=0.98)
            plt.tight_layout()

            # Save the plot using the clear contextual execution labels
            plot_path = os.path.join(plot_dir, f"{self.strategy_name}_{place}_{c_label}_{ch_name}_equalization_plot.png")
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  [Equalization Plot] Saved → {plot_path}")

    # ── Separate Method 6: Reconstructed Pair-Difference Analysis ───────────
    def _plot_pair_diff(self, folder_ai, folder_nat, place, payload_list, channel_idx=None):
        print(f"\n--- Generating Adaptive Pair-Difference Histogram | {place} ---")
        import os
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt

        channel_map = {0: "blue", 1: "green", 2: "red"}
        c_label = channel_map.get(channel_idx, "default")

        plot_dir = _results_path(place, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        def get_stego_folders(category):
            """
            Returns a list of (stego_folder, force_channel_idx) tuples to aggregate from.
            Tries channel_default first. If empty/missing, falls back to all RGB channels.
            """
            default_folder = self._stego_folder(place, category, channel_idx=None)
            if os.path.exists(default_folder):
                for kb in payload_list:
                    kb_path = os.path.join(default_folder, str(kb))
                    if os.path.exists(kb_path) and any(
                        f.lower().endswith((".png", ".jpg")) for f in os.listdir(kb_path)
                    ):
                        return [(default_folder, None)]

            print(f"  [Pair-Diff] channel_default empty for {category}, aggregating separate RGB channels.")
            result = []
            for ch_idx, ch_name in channel_map.items():
                ch_folder = self._stego_folder(place, category, channel_idx=ch_idx)
                if os.path.exists(ch_folder):
                    result.append((ch_folder, ch_idx))
            return result

        def compute_histograms_all_payloads(folder_cover, category, files, bins):
            """
            Returns {kb: (cover_hist, stego_hist)} by reconstructively blending channels
            when separate RGB folders are present, or processing default grayscale containers.
            """
            stego_sources = get_stego_folders(category)
            if not stego_sources:
                print(f"  [Pair-Diff] No stego folders found for {category}.")
                return {}

            # Cover calculation: Standardized luminance conversion used as clean baseline
            cover_diffs = []
            for f in files:
                orig = cv2.imread(os.path.join(folder_cover, f), cv2.IMREAD_COLOR)
                if orig is not None:
                    gray_cover = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.int16)
                    diffs = np.abs(gray_cover[:, 1:] - gray_cover[:, :-1]).flatten()
                    cover_diffs.append(diffs)
            if not cover_diffs:
                return {}
            cover_hist, _ = np.histogram(np.concatenate(cover_diffs), bins=bins, density=True)

            results = {}
            for kb in sorted(payload_list):
                stego_diffs = []
                is_multi_channel = len(stego_sources) > 1

                if is_multi_channel:
                    # Scenario B: Map separate folders to dynamically reconstruct the combined stego matrix
                    folder_map = {ch_idx: path for path, ch_idx in stego_sources}
                    
                    for f in files:
                        stg_b = cv2.imread(os.path.join(folder_map[0], str(kb), f), cv2.IMREAD_COLOR) if 0 in folder_map else None
                        stg_g = cv2.imread(os.path.join(folder_map[1], str(kb), f), cv2.IMREAD_COLOR) if 1 in folder_map else None
                        stg_r = cv2.imread(os.path.join(folder_map[2], str(kb), f), cv2.IMREAD_COLOR) if 2 in folder_map else None
                        
                        sample_img = stg_b if stg_b is not None else (stg_g if stg_g is not None else stg_r)
                        if sample_img is None:
                            continue
                        
                        # Isolate the exact embedded layer channel vectors
                        b_ch = stg_b[:, :, 0] if stg_b is not None else sample_img[:, :, 0]
                        g_ch = stg_g[:, :, 1] if stg_g is not None else sample_img[:, :, 1]
                        r_ch = stg_r[:, :, 2] if stg_r is not None else sample_img[:, :, 2]
                        
                        # Apply standard weighted luminance average matrix matching the cover
                        gray_stego = (0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch).astype(np.int16)
                        diffs = np.abs(gray_stego[:, 1:] - gray_stego[:, :-1]).flatten()
                        stego_diffs.append(diffs)
                else:
                    # Scenario A: Standard running routine using a single folder container
                    stego_folder, fch = stego_sources[0]
                    for f in files:
                        stg = cv2.imread(os.path.join(stego_folder, str(kb), f), cv2.IMREAD_UNCHANGED)
                        if stg is not None:
                            if self.strategy_name == "pvdSequential" or len(stg.shape) == 2:
                                gray_stego = stg.astype(np.int16)
                            elif fch is not None:
                                gray_stego = stg[:, :, fch].astype(np.int16)
                            elif channel_idx is not None:
                                gray_stego = stg[:, :, channel_idx].astype(np.int16)
                            else:
                                gray_stego = cv2.cvtColor(stg, cv2.COLOR_BGR2GRAY).astype(np.int16)
                                
                            diffs = np.abs(gray_stego[:, 1:] - gray_stego[:, :-1]).flatten()
                            stego_stiffs = stego_diffs.append(diffs)

                if not stego_diffs:
                    continue
                stego_hist, _ = np.histogram(np.concatenate(stego_diffs), bins=bins, density=True)
                results[kb] = (cover_hist, stego_hist)
            return results

        # ── Load image file lists ─────────────────────────────────────────────
        ai_files  = sorted(f for f in os.listdir(folder_ai) if f.lower().endswith((".png", ".jpg")))
        nat_files = sorted(f for f in os.listdir(folder_nat) if f.lower().endswith((".png", ".jpg")))

        bins = np.arange(0, 64, 2)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        ai_data  = compute_histograms_all_payloads(folder_ai,  "ai",      ai_files,  bins)
        nat_data = compute_histograms_all_payloads(folder_nat, "natural", nat_files, bins)

        if not ai_data and not nat_data:
            print("  [Pair-Diff] No data found, skipping.")
            return

        payloads = sorted(payload_list)
        base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
        payload_colors = {kb: base_colors[i % len(base_colors)] for i, kb in enumerate(payloads)}

        # Setup custom title label
        if self.strategy_name == "pvdSequential":
            signal_label = "grayscale"
        elif len(get_stego_folders("ai")) > 1 or len(get_stego_folders("natural")) > 1:
            signal_label = "reconstructed luminance (combined RGB streams)"
        elif channel_idx is not None:
            signal_label = f"{c_label} channel"
        else:
            signal_label = "luminance / all channels"

        row_labels = ["AI images", "Natural images"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{self.strategy_name.upper()} | Pair-Difference Distribution | {place.upper()} | {signal_label}",
                     fontsize=14, fontweight="bold", y=1.02)

        for col_idx, (label, data) in enumerate([("AI", ai_data), ("Natural", nat_data)]):
            ax = axes[col_idx]
            ax.set_title(f"{row_labels[col_idx]} — Stego − Cover", fontsize=11, fontweight="bold")
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

            for kb, (cover_hist, stego_hist) in data.items():
                diff = stego_hist - cover_hist
                ax.plot(bin_centers, diff, color=payload_colors[kb], linestyle="-", linewidth=1.8, alpha=0.95, label=f"{kb} KB")

            ax.set_xlabel("Pixel pair difference", fontsize=10)
            ax.set_ylabel("Stego − Cover density", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_xlim(0, 64)
            ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.7)

        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f"{self.strategy_name}_{place}_{c_label}_pairdiff_plot.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [Pair-Diff Plot] Saved → {plot_path}")



    def _save_summary_table(self, place, payload_list, channel_idx=None):
        """
        Generates a LaTeX summary table showing mean metrics per payload,
        split by scene (indoor/outdoor) and image type (AI/Natural).
        Reads from the already-saved CSV files.
        """
        import pandas as pd
        import numpy as np

        channel_map = {0: "blue", 1: "green", 2: "red"}
        c_label = channel_map.get(channel_idx, "default")

        # Load both indoor and outdoor CSVs for this strategy
        dfs = []
        for target_place in ["indoor", "outdoor"]:
            csv_path = os.path.join(
                RESULTS_ROOT, target_place, "csv",
                f"{self.strategy_name}_{target_place}_{c_label}_metrics.csv"
            )
            if os.path.exists(csv_path):
                df_part = pd.read_csv(csv_path)
                dfs.append(df_part)
            else:
                print(f"  [Summary Table] Missing CSV: {csv_path}")

        if not dfs:
            print("  [Summary Table] No CSVs found, skipping.")
            return

        df = pd.concat(dfs, ignore_index=True)
        payloads = sorted(payload_list)

        # Build output path
        out_dir = os.path.join(RESULTS_ROOT, place, "tables")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"{self.strategy_name}_{c_label}_summary_table.tex"
        )

        lines = []
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(
            r"\caption{" + f"{self.strategy_name} --- mean image quality metrics "
            r"(20 images per group)" + r"}"
        )
        lines.append(r"\label{tab:" + f"{self.strategy_name}_{c_label}_metrics" + r"}")
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(r"\begin{tabular}{llrrrrrrr}")
        lines.append(r"\hline")
        lines.append(
            r"\textbf{Scene} & \textbf{Type} & \textbf{Payload (KB)} & "
            r"\textbf{PSNR (dB)} & \textbf{SSIM} & "
            r"\textbf{MSE} & \textbf{Entropy} & \textbf{BPP} \\"
        )
        lines.append(r"\hline")

        for scene in ["indoor", "outdoor"]:
            df_scene = df[df["Place"] == scene]
            first_scene = True

            for img_type in ["AI", "Natural"]:
                df_type = df_scene[df_scene["ImageType"] == img_type]
                first_type = True

                for kb in payloads:
                    sub = df_type[df_type["Payload_KB"] == kb]
                    if sub.empty:
                        continue

                    psnr    = f"{sub['PSNR'].mean():.2f}"
                    ssim    = f"{sub['SSIM'].mean():.6f}"
                    mse     = f"{sub['MSE'].mean():.4f}"
                    entropy = f"{sub['Entropy'].mean():.4f}"
                    bpp     = f"{sub['BPP'].mean():.4f}"

                    # Only print scene label on first row of each scene
                    scene_cell = r"\multirow{X}{*}{" + scene.capitalize() + r"}" \
                        if first_scene and first_type else ""
                    type_cell  = r"\multirow{X}{*}{" + img_type + r"}" \
                        if first_type else ""

                    lines.append(
                        f"  {scene_cell} & {type_cell} & {kb} & "
                        f"{psnr} & {ssim} & {mse} & {entropy} & {bpp} \\\\"
                    )
                    first_scene = False
                    first_type  = False

                lines.append(r"  \cline{2-8}")

            lines.append(r"  \hline")

        lines.append(r"\end{tabular}}")
        lines.append(r"\end{table}")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"  [Summary Table] Saved → {out_path}")
        # ── helpers ───────────────────────────────────────────────────────────────
        @staticmethod
        def pair_diff(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
            return np.abs(gray[:, 1:] - gray[:, :-1]).flatten()

        @staticmethod
        def lsb_counts(img):
            bits = (img & 1).flatten()
            return [np.sum(bits == 0), np.sum(bits == 1)]


# ─────────────────────────────────────────────────────────────────────────────
#  Concrete analyzers — add any strategy-specific overrides here
# ─────────────────────────────────────────────────────────────────────────────
class LSBAnalyzer(StegoAnalyzer):
    def __init__(self, stego_root="embeddings"):
        super().__init__("lsbRandomSpatial", stego_root)


class LSBCannySobelAnalyzer(StegoAnalyzer):
    def __init__(self, stego_root="embeddings"):
        super().__init__("lsbCannySobel", stego_root)


class PVDAnalyzer(StegoAnalyzer):
    def __init__(self, stego_root="embeddings"):
        super().__init__("pvdSequential", stego_root)

class LSBMRAnalyzer(StegoAnalyzer):
    def __init__(self, stego_root="embeddings"):
        super().__init__("lsbmr", stego_root)