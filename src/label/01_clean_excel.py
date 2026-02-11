import pandas as pd
import re
from pathlib import Path

# ================= CONFIGURATION =================
INPUT_FILE = Path("/home/dlf903/Histology_v2/original.xlsx")
OUTPUT_FILE = "/home/dlf903/Histology_v2/original_cleaned.xlsx"

# Regex to find existing "[Number]" pattern
REGEX_EXISTING_NUM = re.compile(r"\[(\d+)\]")

def clean_text_content(val):
    """Standardizes the location text (removes old [XX], capitalizes)."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    s = re.sub(r"\[\d+\]", "", s).strip()
    s = s.lstrip(".-: ")
    return s.capitalize()

def main():
    print(f"--- Processing {INPUT_FILE} ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_FILE)
    except:
        df = pd.read_excel(INPUT_FILE)

    # Standardize Column Names
    df.columns = [c.strip() for c in df.columns]
    # Trim trailing whitespace from all string cells in the dataframe
    df = df.applymap(lambda x: x.rstrip() if isinstance(x, str) else x)
    
    rekvn_col = next((c for c in df.columns if 'rekvn' in c.lower()), None)
    lok_col = next((c for c in df.columns if 'lokalisation' in c.lower()), None)
    besk_col = next((c for c in df.columns if 'beskrivelse' in c.lower()), None)

    if not rekvn_col or not lok_col or not besk_col:
        raise ValueError("Could not find 'Rekvn_nr', 'Lokalisation', or 'Beskrivelse' columns.")

    initial_len = len(df)

    # 2. Remove Empty Rows (Rekvn_nr is NaN)
    df = df.dropna(subset=[rekvn_col])
    
    # 3. Remove Empty Descriptions (Beskrivelse is NaN or whitespace)
    # Convert to string, strip whitespace, and check if length > 0
    df = df[df[besk_col].astype(str).str.strip().str.len() > 0]
    
    # Remove strictly NaN values again just to be safe
    df = df.dropna(subset=[besk_col])

    print(f"Removed {initial_len - len(df)} rows (Empty Rekvn or Empty Description).")

    # 4. Clean Descriptions (Remove line breaks)
    print("Cleaning 'Beskrivelse' column...")
    df[besk_col] = df[besk_col].astype(str).str.replace(r'[\r\n]+', ' ', regex=True)
    df[besk_col] = df[besk_col].str.replace(r'\s+', ' ', regex=True).str.strip()

    # 5. Fix Block Numbers (Sequential Logic)
    print("Fixing Block Numbers...")
    final_lokalisation_values = []
    
    current_rekvn = None
    block_counter = 1

    for _, row in df.iterrows():
        this_rekvn = row[rekvn_col]
        original_lok = str(row[lok_col])
        
        # New Patient? Reset counter.
        if this_rekvn != current_rekvn:
            current_rekvn = this_rekvn
            block_counter = 1 
        
        text_part = clean_text_content(original_lok)
        match = REGEX_EXISTING_NUM.search(original_lok)
        
        if match:
            # If number exists, use it
            existing_num = int(match.group(1))
            final_num = existing_num
            block_counter = existing_num + 1
        else:
            # If no number, use counter
            final_num = block_counter
            block_counter += 1
        
        new_val = f"[{final_num:02d}] {text_part}"
        final_lokalisation_values.append(new_val)

    df[lok_col] = final_lokalisation_values

    # 6. Save Result
    print(f"Saving cleaned file to: {OUTPUT_FILE}")
    df.to_excel(OUTPUT_FILE, index=False)
    print("--- Done ---")

if __name__ == "__main__":
    main()