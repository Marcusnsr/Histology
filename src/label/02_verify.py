import os
import pandas as pd
import re
from pathlib import Path

# ================= CONFIGURATION =================
SOURCE_ROOT = Path(r"/home/dlf903/erda_mount/GAIN/Histology/Histology_Anonymized")       
INPUT_EXCEL = Path("/home/dlf903/Histology_v2/original_cleaned.xlsx") 
OUTPUT_EXCEL = Path("/home/dlf903/Histology_v2/original_cleaned_synced.xlsx")

# ================= LOGIC =================
CSV_BLOCK_REGEX = re.compile(r"\[(\d+)\]")
FOLDER_REGEX = re.compile(r".*?(\d{2})\d{2}_HE.*")

def main():
    print("--- Starting Alignment & Sync ---")
    
    if not INPUT_EXCEL.exists():
        print(f"Error: Could not find {INPUT_EXCEL}")
        return

    # 1. Load Excel
    print("Loading Excel...")
    df = pd.read_excel(INPUT_EXCEL)
    df.columns = [c.strip() for c in df.columns]
    
    rekvn_col = next((c for c in df.columns if 'rekvn' in c.lower()), None)
    
    # Get clean set of patients from Excel
    excel_patients = set(df[rekvn_col].astype(str).str.strip().unique())
    
    # 2. Scan Disk
    if not SOURCE_ROOT.exists():
        print(f"Error: Source folder not found: {SOURCE_ROOT}")
        return

    print(f"Scanning folders in {SOURCE_ROOT}...")
    disk_patients = set()
    for p in SOURCE_ROOT.iterdir():
        if p.is_dir():
            disk_patients.add(p.name.strip())

    # 3. Compare
    matched_patients = excel_patients.intersection(disk_patients)
    missing_on_disk = excel_patients - disk_patients  # In Excel, but NO folder
    
    print("\n" + "="*40)
    print(f"  RESULTS")
    print("="*40)
    print(f"MATCHED Patients: {len(matched_patients)}")
    print(f"MISSING Patients: {len(missing_on_disk)}")
    
    # --- PRINT THE MISSING LIST ---
    if missing_on_disk:
        print("\n" + "-"*40)
        print("  LIST OF MISSING PATIENTS (In Excel, No Folder)")
        print("-"*40)
        
        # Sort them so they are easy to read
        for i, pat in enumerate(sorted(missing_on_disk), 1):
            print(f"{i}. {pat}")
            
        print("-"*40)
        print(f"Removing these {len(missing_on_disk)} missing patients from the dataset...")

        # 4. REMOVE MISSING ROWS AND SAVE
        # Filter: Keep rows where the stripped Patient ID IS in the 'disk_patients' set
        df_synced = df[df[rekvn_col].astype(str).str.strip().isin(disk_patients)].copy()
        
        print(f"Saving synced data to: {OUTPUT_EXCEL}")
        df_synced.to_excel(OUTPUT_EXCEL, index=False)
        
        print("\n" + "="*40)
        print("  SYNC COMPLETE")
        print("="*40)
        print(f"Original Rows: {len(df)}")
        print(f"Rows Removed:  {len(df) - len(df_synced)}")
        print(f"Final Rows:    {len(df_synced)}")
        print("-" * 40)
        print(f"Created: {OUTPUT_EXCEL}")

    else:
        print("\nZero missing patients.")
        df.to_excel(OUTPUT_EXCEL, index=False)

if __name__ == "__main__":
    main()