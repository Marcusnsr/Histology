import json
import glob
import math
import random
import traceback
from pathlib import Path
from huggingface_hub import login
import pydicom
import pydicom.uid
from wsidicom import WsiDicom
from wsidicom.instance.dataset import WsiDataset
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch
import timm
from torchvision import transforms
from gigapath import slide_encoder as gp_slide
from tqdm.auto import tqdm

login(token="")

# Fixes Metadata in Memory
_original_is_supported = WsiDataset.is_supported_wsi_dicom

def _patched_is_supported(dataset):
    # Inject missing tags if needed
    if 'SOPClassUID' not in dataset:
        dataset.SOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.77.1.6')
    if 'SOPInstanceUID' not in dataset:
        dataset.SOPInstanceUID = pydicom.uid.generate_uid()
    if 'ImageType' not in dataset:
        dataset.ImageType = ['ORIGINAL', 'PRIMARY', 'VOLUME', 'NONE']
    return _original_is_supported(dataset)

# Apply patch
WsiDataset.is_supported_wsi_dicom = _patched_is_supported
print("[System] WSI Metadata Monkey-Patch applied.")

# Config
# Input Folder containing the 'HE' subfolder with DICOMs
IN_DIR    = "/home/dlf903/erda_mount/GAIN/Histology/Histology_Anonymized_Sorted/HE"

# Output Folders
OUT_ROOT  = "/home/dlf903/erda_mount/GAIN/Histology/processed_gigapath/tiles_and_features"
ALT_OUT_ROOT = "/home/dlf903/erda_mount/GAIN/Histology/processed_gigapath/slide_embeddings"

# Model Paths
SLIDE_PTH = "/home/dlf903/Histology_v2/prov-gigapath/checkpoints/slide_encoder.pth"
HF_MODEL  = "hf_hub:prov-gigapath/prov-gigapath"
ARCH      = "gigapath_slide_enc12l768d"

# Tiling
TILE_SIZE = 256  
WHITE_THRESHOLD = 0.90
WHITE_VAL = 200

# Inference
BATCH_SIZE = 128
USE_MIXED_PRECISION = True
FORCE = False             
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Stochastic Embeddings
TILE_DROPOUT = 0.3
N_SLIDE_SAMPLES = 10
BASE_SEED = 12345
SAVE_MODE = "separate" 

def is_mostly_white(tile_img, thr=WHITE_THRESHOLD, white_thr_val=WHITE_VAL) -> bool:
    # Convert to numpy array
    arr = np.asarray(tile_img, dtype=np.uint8)
    if arr.size == 0: return True
    
    # Check if pixels are brighter than threshold
    white = np.all(arr >= white_thr_val, axis=2)
    return float(white.mean()) >= thr

def tile_dicom(dicom_path: Path, out_tiles_dir: Path, target_size: int):
    """
    Reads a WSI DICOM, iterates over non-empty regions, and saves patches.
    """
    out_tiles_dir.mkdir(parents=True, exist_ok=True)
    kept_coords = []
    
    # Open WSI
    try:
        slide = WsiDicom.open(dicom_path)
    except Exception as e:
        print(f"[Err] Could not open {dicom_path.name}: {e}")
        return [], (0, 0)

    # Get dimensions
    W, H = slide.size.width, slide.size.height
    
    stride = target_size # Non-overlapping
    
    n_x = W // stride
    n_y = H // stride
    total = n_x * n_y
    
    pbar = tqdm(total=total, desc="Tiling DICOM", leave=False)
    
    for gy in range(n_y):
        for gx in range(n_x):
            x = gx * stride
            y = gy * stride
            try:
                # Returns a PIL Image in RGB
                tile = slide.read_region((x, y), 0, (stride, stride))
                
                # Check for white background
                if not is_mostly_white(tile):
                    # Save
                    name = f"{gx},{gy}.png"
                    
                    # If the read tile isn't exactly target_size (e.g. edge cases), resize
                    if tile.size != (target_size, target_size):
                        tile = tile.resize((target_size, target_size), Image.BICUBIC)
                        
                    tile.save(out_tiles_dir / name)
                    kept_coords.append((gx, gy))
                    
            except Exception as e:
                pass
            
            pbar.update(1)
            
    pbar.close()
    return kept_coords, (W, H)

def _load_tile_tensor(path: Path, tfm):
    p = Image.open(path).convert("RGB")
    return tfm(p)

def _coords_to_norm(coords, W, H, stride, tile_size):
    out = []
    for (gx, gy) in coords:
        # Centroid of the tile
        cx = gx * stride + tile_size / 2.0
        cy = gy * stride + tile_size / 2.0
        out.append((cx / W, cy / H))
    return np.asarray(out, dtype=np.float32)

def run_one_slide(dicom_path: Path, out_root: Path, tile_encoder, slide_enc, tfm):
    stem = dicom_path.stem 
    # Setup Paths
    slide_work_dir = Path(out_root) / stem
    tiles_dir = slide_work_dir / "tiles"
    features_csv = slide_work_dir / "features.csv"
    meta_json = slide_work_dir / "metadata.json"
    
    save_root = (Path(out_root) / stem) if SAVE_MODE == "inplace" else (Path(ALT_OUT_ROOT) / stem)
    save_root.mkdir(parents=True, exist_ok=True)

    # Check if done
    if meta_json.exists() and not FORCE:
            try:
                with open(meta_json, 'r') as f:
                    json.load(f)
                print(f"[skip] {stem} (Done)")
                return {"slide_id": stem, "status": "skipped"}
            except json.JSONDecodeError:
                # If the file exists but is corrupt (e.g. crash during write), we re-run it.
                print(f"[warn] {stem} (Corrupt metadata found - Reprocessing)")

    print(f"[proc] {stem}")
    slide_work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tiling
    # Check if we already tiled this slide in a previous run
    if tiles_dir.exists() and list(tiles_dir.glob("*.png")):
        print(f"       Using existing tiles in {tiles_dir}")
        tile_paths = sorted(tiles_dir.glob("*.png"))
        kept_coords = []
        for p in tile_paths:
            try:
                gx, gy = map(int, p.stem.split(","))
                kept_coords.append((gx, gy))
            except: continue
        
        # Load W/H from cached metadata if possible
        if meta_json.exists():
            try:
                m = json.loads(meta_json.read_text())
                W, H = m["original_width"], m["original_height"]
            except:
                 # Fallback Estimate
                max_x = max(c[0] for c in kept_coords) if kept_coords else 0
                max_y = max(c[1] for c in kept_coords) if kept_coords else 0
                W, H = (max_x+1)*TILE_SIZE, (max_y+1)*TILE_SIZE
        else:
            # Estimate
            max_x = max(c[0] for c in kept_coords) if kept_coords else 0
            max_y = max(c[1] for c in kept_coords) if kept_coords else 0
            W, H = (max_x+1)*TILE_SIZE, (max_y+1)*TILE_SIZE
            
    else:
        # Perform Tiling from DICOM
        kept_coords, (W, H) = tile_dicom(dicom_path, tiles_dir, TILE_SIZE)

    if not kept_coords:
        print(f"[warn] {stem}: No tissue found.")
        return {"slide_id": stem, "status": "empty"}

    # 2. Embed Tiles
    if features_csv.exists() and not FORCE:
        # Load cached embeddings
        df = pd.read_csv(features_csv)
        # Filter to ensure files exist
        valid_mask = df["patch_name"].apply(lambda x: (tiles_dir / x).exists())
        df = df[valid_mask]
        
        tile_embs = df[[c for c in df.columns if c.isdigit()]].to_numpy(dtype=np.float32)
        
        # Re-align coords
        name_map = {f"{x},{y}.png": (x,y) for x,y in kept_coords}
        loaded_coords = []
        loaded_embs = []
        
        for idx, row in df.iterrows():
            nm = row["patch_name"]
            if nm in name_map:
                loaded_coords.append(name_map[nm])
                # get embedding columns
                vec = row[[str(i) for i in range(1536)]].values.astype('float32')
                loaded_embs.append(vec)
        
        kept_coords = loaded_coords
        tile_embs = np.array(loaded_embs)
        print(f"       Loaded {len(tile_embs)} cached tile embeddings.")
        
    else:
        # Compute Embeddings
        embs = []
        n_batches = (len(kept_coords) + BATCH_SIZE - 1) // BATCH_SIZE
        
        with torch.no_grad():
            for i in tqdm(range(n_batches), desc="Encoding Tiles", leave=False):
                batch_coords = kept_coords[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                
                # Load images
                batch_tensors = []
                for (gx, gy) in batch_coords:
                    p = tiles_dir / f"{gx},{gy}.png"
                    batch_tensors.append(_load_tile_tensor(p, tfm))
                
                xb = torch.stack(batch_tensors).to(DEVICE)
                
                # Inference
                if USE_MIXED_PRECISION:
                    with torch.cuda.amp.autocast():
                        eb = tile_encoder(xb)
                else:
                    eb = tile_encoder(xb)
                
                embs.append(eb.detach().cpu().numpy())
                
        tile_embs = np.concatenate(embs, axis=0).astype("float32")
        
        # Save Features CSV
        # Columns: 0..1535, patch_name
        df_data = pd.DataFrame(tile_embs, columns=[str(i) for i in range(tile_embs.shape[1])])
        df_data["patch_name"] = [f"{x},{y}.png" for (x,y) in kept_coords]
        df_data.to_csv(features_csv, index=False)

    # 3. Slide Level Encoding
    coords_norm = _coords_to_norm(kept_coords, W, H, TILE_SIZE, TILE_SIZE)
    produced_files = []
    
    # Save ONE "Full" Embedding (100% tiles)
    with torch.no_grad():
        # Use ALL tiles (No slicing/dropping)
        E = torch.tensor(tile_embs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        C = torch.tensor(coords_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        
        out = slide_enc(E, C)
        # Unpack result
        if isinstance(out, dict): res = out.get("embed", list(out.values())[0])
        elif isinstance(out, (list, tuple)): res = out[0]
        else: res = out
        
        slide_vec = res.squeeze(0).detach().cpu().numpy()
        
        # Save as "full"
        fname_full = save_root / "slide_embedding_full.npy"
        np.save(fname_full, slide_vec)
        produced_files.append(str(fname_full))

    # Generate Stochastic Samples (Partial embeddings)
    n_total = len(tile_embs)
    n_drop = int(math.floor(TILE_DROPOUT * n_total))
    
    with torch.no_grad():
        for s in range(1, N_SLIDE_SAMPLES + 1):
            rng = random.Random(BASE_SEED + s)
            indices = list(range(n_total))
            if n_drop > 0:
                indices = rng.sample(indices, k=n_total - n_drop)
            
            E = torch.tensor(tile_embs[indices], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            C = torch.tensor(coords_norm[indices], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            
            out = slide_enc(E, C)
            if isinstance(out, dict): res = out.get("embed", list(out.values())[0])
            elif isinstance(out, (list, tuple)): res = out[0]
            else: res = out
                
            slide_vec = res.squeeze(0).detach().cpu().numpy()
            fname = save_root / f"slide_embedding_{s:03d}.npy"
            np.save(fname, slide_vec)
            produced_files.append(str(fname))

    # Save Metadata
    meta = {
        "original_width": W, "original_height": H,
        "n_tiles": n_total,
        "source": str(dicom_path),
        "embeddings": produced_files
    }
    (slide_work_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    
    return {"slide_id": stem, "status": "success", "n_tiles": n_total}

def main():
    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Gigapath DICOM Pipeline ---")
    print(f"Input:  {IN_DIR}")
    print(f"Output: {OUT_ROOT}")
    print(f"Embeddings: {ALT_OUT_ROOT}")
    print(f"Device: {DEVICE}")

    # Load Models
    print("Loading models...")
    try:
        tile_encoder = timm.create_model(HF_MODEL, pretrained=True).to(DEVICE).eval()
        slide_enc    = gp_slide.create_model(SLIDE_PTH, ARCH, 1536).to(DEVICE).eval()
    except Exception as e:
        print(f"Error loading models. Check paths or token.\n{e}")
        return
    
    # Transform
    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Find DICOMs
    input_path = Path(IN_DIR)
    dicoms = sorted(list(input_path.glob("*.dcm")))
    
    if not dicoms:
        print(f"[Error] No .dcm files found in {IN_DIR}")
        return

    print(f"Found {len(dicoms)} slides.")

    # Run
    summary = []
    for dcm in tqdm(dicoms, desc="Slides"):
        try:
            res = run_one_slide(dcm, out_root, tile_encoder, slide_enc, tfm)
            summary.append(res)
        except Exception as e:
            print(f"[Fail] {dcm.name}: {e}")
            traceback.print_exc()
            summary.append({"slide_id": dcm.stem, "status": "error"})
            
    pd.DataFrame(summary).to_csv(out_root / "index.csv", index=False)
    print("Done.")

if __name__ == "__main__":
    main()
