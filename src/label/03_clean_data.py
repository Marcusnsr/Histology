import os
import shutil
import pandas as pd
import re
from pathlib import Path

# ================= CONFIGURATION =================
SOURCE_ROOT = Path(r"/home/dlf903/erda_mount/GAIN/Histology/Histology_Anonymized")
DEST_ROOT = Path(r"/home/dlf903/erda_mount/GAIN/Histology/Histology_Anonymized_Sorted")
INPUT_EXCEL = Path("/home/dlf903/Histology_v2/original_cleaned_synced.xlsx")
OUTPUT_EXCEL = Path("/home/dlf903/Histology_v2/original_cleaned_final.xlsx")

# ================= LOGIC =================
STAIN_REGEX = re.compile(r"^\d{4}_(.*?)(?:_\d+)?$")

CSV_BLOCK_REGEX = re.compile(r"\[(\d+)\]")

def get_largest_file(directory):
    """Finds the largest file (the main image) in a subfolder."""
    try:
        files = [f for f in directory.iterdir() if f.is_file() and not f.name.startswith('.')]
        if not files: return None
        return max(files, key=lambda f: f.stat().st_size)
    except: return None

def main():
    print("--- Starting Final Reorganization ---")
    
    if not INPUT_EXCEL.exists():
        print(f"Error: Could not find {INPUT_EXCEL}")
        return

    # 1. Load Excel
    print("Loading Excel metadata...")
    df = pd.read_excel(INPUT_EXCEL)
    df.columns = [c.strip() for c in df.columns]
    
    rekvn_col = next((c for c in df.columns if 'rekvn' in c.lower()), None)
    lok_col = next((c for c in df.columns if 'lokalisation' in c.lower()), None)
    
    if not rekvn_col or not lok_col:
        print("Error: Missing 'Rekvn' or 'Lokalisation' columns in Excel.")
        return

    valid_rekvns = set(df[rekvn_col].astype(str).str.strip().unique())
    print(f"Loaded {len(valid_rekvns)} valid patients.")

    final_rows = []
    
    # 2. Prepare Destination
    if not SOURCE_ROOT.exists():
        print(f"Error: Source folder not found: {SOURCE_ROOT}")
        return
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    total_files_copied = 0

    # 3. Iterate Patients
    for patient_folder in SOURCE_ROOT.iterdir():
        if not patient_folder.is_dir(): continue

        rekvn = patient_folder.name.strip()

        # Only process patients in our clean Excel
        if rekvn not in valid_rekvns:
            continue

        # 4. Iterate Subfolders
        for subfolder in patient_folder.iterdir():
            if not subfolder.is_dir(): continue
            
            # A. Determine Stain Type & Clean It
            match = STAIN_REGEX.match(subfolder.name)
            if match:
                raw_stain = match.group(1).strip()
                
                # 1. Remove spaces (WAS 3 -> WAS3)
                stain_name = raw_stain.replace(" ", "")
                
                # 2. Remove leading number prefixes (e.g. "01-HE" -> "HE")
                # This regex removes "Digits" followed by "Hyphen" at the start
                stain_name = re.sub(r"^\d+-", "", stain_name)
            else:
                stain_name = "Uncategorized"

            # B. Get Largest File
            largest_file = get_largest_file(subfolder)
            if not largest_file: continue

            # C. Define New Filename (Using the clean stain name implicitly via subfolder cleaning?)            
            clean_subfolder_name = subfolder.name.replace(" ", "")
            new_filename = f"{rekvn}_{clean_subfolder_name}_{largest_file.name}"
            
            # Destination: .../Processed/WAS3/filename
            dest_stain_folder = DEST_ROOT / stain_name
            dest_stain_folder.mkdir(parents=True, exist_ok=True)
            
            dest_file_path = dest_stain_folder / new_filename
            
            # D. Copy File
            if not dest_file_path.exists():
                shutil.copy2(largest_file, dest_file_path)
            
            total_files_copied += 1

            # E. Create Excel Row
            folder_block_id = subfolder.name[:2] # e.g. "01"
            
            def match_block(val):
                m = CSV_BLOCK_REGEX.search(str(val))
                if m:
                    return int(m.group(1)) == int(folder_block_id)
                return False

            matching_rows = df[
                (df[rekvn_col].astype(str).str.strip() == rekvn) & 
                (df[lok_col].apply(match_block))
            ]

            if not matching_rows.empty:
                row_data = matching_rows.iloc[0].to_dict()
                row_data['filename'] = new_filename
                row_data['stain_type'] = stain_name
                final_rows.append(row_data)
            else:
                # Fallback for orphaned images
                row_data = {col: None for col in df.columns}
                row_data[rekvn_col] = rekvn
                row_data['filename'] = new_filename
                row_data['stain_type'] = stain_name
                final_rows.append(row_data)

        processed_count += 1
        if processed_count % 50 == 0:
            print(f"Processed {processed_count} patients...")

    # 5. Save Final Excel
    if final_rows:
        new_df = pd.DataFrame(final_rows)
        
        # Reorder columns
        cols = ['filename', 'stain_type'] + [c for c in new_df.columns if c not in ['filename', 'stain_type']]
        new_df = new_df[cols]
        
        print(f"Saving final dataset to: {OUTPUT_EXCEL}")
        new_df.to_excel(OUTPUT_EXCEL, index=False)
        
        print("\n" + "="*40)
        print("  PROCESSING COMPLETE")
        print("="*40)
        print(f"Patients Processed: {processed_count}")
        print(f"Total Files Copied: {total_files_copied}")
        print(f"Excel Saved To:     {OUTPUT_EXCEL}")
        print(f"Images Sorted Into: {DEST_ROOT}")
    else:
        print("No files were processed.")

if __name__ == "__main__":
    main()