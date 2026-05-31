from readingFromFile import check_secret_size
from strategy.lsb_strategy import LSBStrategy
from strategy.pvd_strategy import PVDStrategy
from strategy.lsbmr_startegy import LSBMRStrategy
from strategy.lsb_Canny_Sobel2 import LSBCannySobelStrategy
from analysis_pipeline3 import LSBAnalyzer, PVDAnalyzer, LSBCannySobelAnalyzer, LSBMRAnalyzer



# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — edit these paths and payloads, nothing else needs to change
# ─────────────────────────────────────────────────────────────────────────────

BASE = r"C:\Users\Diana\Desktop\disertatieSteganographyApp"  # Base folder for all paths below
listL = [1, 5, 10, 25, 50, 100, 200, 300, 500]  # in KB

# Image folders — one entry per (place, type)
FOLDERS = {
    "indoor": {
        "ai":      r"C:\Users\Diana\Desktop\disertatieSteganographyApp\images\indoor\pozeTelefonAI1792x2400",
        "natural": r"C:\Users\Diana\Desktop\disertatieSteganography\images\indoor\pozeTelefonPngRealResized1792x2400",
    },
    "outdoor": {
        "ai":      r"C:\Users\Diana\Desktop\disertatieSteganographyApp\images\outdoor\OutdoorAIGenerated1792x2400",
        "natural": r"C:\Users\Diana\Desktop\disertatieSteganographyApp\images\outdoor\OutdoorNatural1792x2400",
    },
}

# Fixed KB payloads (same for every image and every method)
# Indoor minimum edge capacity was 6 KB → use 5 KB as safe floor
# Outdoor minimum edge capacity was 11 KB → use 10 KB as safe floor

PAYLOADS = {
    "indoor":  listL,  # [1, 5, 10, 25, 50, 100, 200, 300, 500] KB
    "outdoor": listL,
}

SECRET_FILE = r"C:\Users\Diana\Desktop\disertatieSteganographyApp\secret.txt"
STEGO_ROOT  = r"C:\Users\Diana\Desktop\disertatieSteganographyApp\embeddings"
SECRET_KEY  = "BlueAvatarlife123"
NUM_IMAGES  = 20


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ─────────────────────────────────────────────────────────────────────────────
    # COMMENTED: OLD SINGLE ANALYSIS APPROACH
    # ─────────────────────────────────────────────────────────────────────────────
    # To run only ONE place and strategy combination:
    #
    target_place = "indoor"  # or "indoor"
    target_strat = "lsbRandomSpatial"  # or "lsbCannySobel" or "pvd"
    #
    folders = FOLDERS[target_place]
    payload_list = PAYLOADS[target_place]
    # 
    if target_strat == "lsbCannySobel":
        strategy_obj = LSBCannySobelStrategy()
        analyzer = LSBCannySobelAnalyzer(STEGO_ROOT)
    elif target_strat == "lsbRandomSpatial":
        strategy_obj = LSBStrategy(key=SECRET_KEY)
        analyzer = LSBAnalyzer(STEGO_ROOT)
    elif target_strat == "lsbmr":
        strategy_obj = LSBMRStrategy()
        analyzer = LSBMRAnalyzer(STEGO_ROOT)
    else:
        strategy_obj = PVDStrategy()
        analyzer = PVDAnalyzer(STEGO_ROOT)
    #
    print(f"\n{'='*60}")
    print(f"RUNNING SINGLE ANALYSIS")
    print(f"PLACE: {target_place.upper()}")
    print(f"STRATEGY: {target_strat.upper()}")
    print(f"PAYLOADS: {payload_list} KB")
    print(f"{'='*60}")
    #
    analyzer.run_pipeline(
        folder_ai    = folders["ai"],
        folder_nat   = folders["natural"],
        place        = target_place,
        payload_list = payload_list,
        strategy_obj = strategy_obj,
        secret_file  = SECRET_FILE,
        #channel_idx  = 0  # Only needed for LSBMR, but can be passed universally if your base strategy class handles it via **kwargs
    )
    # ─────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────
    # NEW: LOOP THROUGH ALL PLACE AND STRATEGY COMBINATIONS
    # ─────────────────────────────────────────────────────────────────────────────
    
    # Define all strategies to test
    '''
    strategies = {
        "lsbRandomSpatial": (LSBStrategy(key=SECRET_KEY), LSBAnalyzer(STEGO_ROOT)),
        "lsbCannySobel":    (LSBCannySobelStrategy(),     LSBCannySobelAnalyzer(STEGO_ROOT)),
        "pvdSequential":              (PVDStrategy(),               PVDAnalyzer(STEGO_ROOT)),
    }
    
    # Get all places (indoor/outdoor)
    all_places = list(FOLDERS.keys())
    
    # Loop through each place
    for target_place in all_places:
        # Loop through each strategy
        for target_strat, (strategy_obj, analyzer) in strategies.items():
            
            # Get the configuration for this place
            folders = FOLDERS[target_place]
            payload_list = PAYLOADS[target_place]
            
            print(f"\n{'='*60}")
            print(f"RUNNING ANALYSIS")
            print(f"PLACE: {target_place.upper()}")
            print(f"STRATEGY: {target_strat.upper()}")
            print(f"PAYLOADS: {payload_list} KB")
            print(f"{'='*60}")
            
            # Run the pipeline for this combination
            analyzer.run_pipeline(
                folder_ai    = folders["ai"],
                folder_nat   = folders["natural"],
                place        = target_place,
                payload_list = payload_list,
                strategy_obj = strategy_obj,
                secret_file  = SECRET_FILE
            )
            
            print(f"\n✓ {target_place.upper()} + {target_strat.upper()} Complete\n")
    
    print("\n" + "="*60)
    print("ALL ANALYSES COMPLETE. Check your CSV files for results.")
    print("="*60)

    '''