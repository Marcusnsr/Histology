import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION (Must match your processing script) ---
IN_DIR    = "/home/dlf903/erda_mount/GAIN/Histology/Histology_Anonymized_Sorted/HE"
OUT_ROOT  = "/home/dlf903/erda_mount/GAIN/Histology/processed_gigapath/tiles_and_features"

def verify_dataset():
    input_path = Path(IN_DIR)
    output_path = Path(OUT_ROOT)
    
    # 1. Get list of all slides that SHOULD exist
    print(f"Scanning input directory: {IN_DIR}")
    dicoms = sorted(list(input_path.glob("*.dcm")))
    if not dicoms:
        print("No DICOMs found! Check your IN_DIR path.")
        return

    print(f"Found {len(dicoms)} source slides.")
    
    results = []
    
    # 2. Iterate and Check
    for dcm in tqdm(dicoms, desc="Verifying"):
        stem = dcm.stem
        work_dir = output_path / stem
        meta_json = work_dir / "metadata.json"
        
        status = "MISSING"
        details = "Folder not found"
        
        if work_dir.exists():
            if meta_json.exists():
                try:
                    # Try to read the JSON to ensure it's not corrupt (0 bytes or half-written)
                    with open(meta_json, 'r') as f:
                        data = json.load(f)
                    
                    # Optional: Check if embeddings list in metadata is not empty
                    if data.get("embeddings") and len(data["embeddings"]) > 0:
                        status = "COMPLETED"
                        details = f"Valid Metadata ({data['n_tiles']} tiles)"
                    else:
                        status = "CORRUPT"
                        details = "Metadata exists but embedding list is empty"
                        
                except json.JSONDecodeError:
                    status = "CORRUPT"
                    details = "metadata.json is not valid JSON"
                except Exception as e:
                    status = "CORRUPT"
                    details = f"Error reading metadata: {str(e)}"
            else:
                # Folder exists, but no metadata -> Crashed halfway
                status = "INCOMPLETE"
                
                # Check what IS there
                has_tiles = (work_dir / "tiles").exists()
                has_csv = (work_dir / "features.csv").exists()
                details = f"Partial run (Tiles: {has_tiles}, CSV: {has_csv})"
        
        results.append({
            "slide_id": stem,
            "status": status,
            "details": details,
            "path": str(dcm)
        })

    # 3. Analyze Results
    df = pd.DataFrame(results)
    
    print("\n" + "="*30)
    print("VERIFICATION SUMMARY")
    print("="*30)
    print(df["status"].value_counts().to_string())
    print("="*30)
    
    # 4. Save Reports
    # List of everything
    df.to_csv("verification_full_report.csv", index=False)
    
    # List of files that need processing (Missing or Corrupt or Incomplete)
    todo = df[df["status"] != "COMPLETED"]
    todo.to_csv("verification_remaining_files.csv", index=False)
    
    print(f"\nFull report saved to: verification_full_report.csv")
    print(f"Files to re-run:      {len(todo)}")
    print(f"Re-run list saved to: verification_remaining_files.csv")

    if len(todo) > 0:
        print("\nExample of failed/incomplete slides:")
        print(todo[["slide_id", "status", "details"]].head())

if __name__ == "__main__":
    verify_dataset()