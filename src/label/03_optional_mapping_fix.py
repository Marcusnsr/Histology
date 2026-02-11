import pandas as pd
import shutil
from pathlib import Path

# ================= CONFIGURATION =================
# Path to your SORTED folder (Destination from script 3)
SORTED_ROOT = Path(r"/home/dlf903/erda_mount/GAIN/Histology/Histology_Anonymized_Sorted")
# Path to your FINAL Excel
EXCEL_FILE = Path(r"/home/dlf903/Histology_v2/original_cleaned_final.xlsx")

# Mapping: "Bad Folder Name" -> "Correct Folder Name"
FIX_MAPPING = {
    "01-HE": "HE",
    "02-APASU": "APASU",
    "03-WAS3": "WAS3"
}

def main():
    print("--- Starting Quick Fix for Stains ---")
    
    if not EXCEL_FILE.exists():
        print("Error: Excel file not found.")
        return

    # 1. Load Excel
    print("Loading Excel...")
    df = pd.read_excel(EXCEL_FILE)
    
    # Track changes
    files_moved = 0
    rows_updated = 0

    # 2. Process each fix
    for bad_name, correct_name in FIX_MAPPING.items():
        bad_folder = SORTED_ROOT / bad_name
        correct_folder = SORTED_ROOT / correct_name
        
        # A. Move Files on Disk
        if bad_folder.exists():
            print(f"Processing folder: {bad_name} -> {correct_name}")
            correct_folder.mkdir(parents=True, exist_ok=True)
            
            for file_path in bad_folder.iterdir():
                if file_path.is_file():
                    dest_path = correct_folder / file_path.name
                    
                    # Move file
                    shutil.move(str(file_path), str(dest_path))
                    files_moved += 1
            
            # Remove the now empty bad folder
            try:
                bad_folder.rmdir()
                print(f"  Removed empty folder: {bad_name}")
            except OSError:
                print(f"  Warning: Could not remove {bad_name} (might not be empty).")
        else:
            print(f"Folder {bad_name} not found on disk, checking Excel only...")

        # B. Update Excel 'stain_type' column
        # Find rows where stain_type is the bad name
        mask = df['stain_type'] == bad_name
        count = mask.sum()
        
        if count > 0:
            df.loc[mask, 'stain_type'] = correct_name
            rows_updated += count
            print(f"  Updated {count} rows in Excel.")

    # 3. Save Excel
    if rows_updated > 0 or files_moved > 0:
        print(f"Saving updated Excel to: {EXCEL_FILE}")
        df.to_excel(EXCEL_FILE, index=False)
        print("--- Fix Complete ---")
        print(f"Total Files Moved: {files_moved}")
        print(f"Total Rows Updated: {rows_updated}")
    else:
        print("No changes needed.")

if __name__ == "__main__":
    main()