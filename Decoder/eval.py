import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from gigapath_decoder import load_and_group_data, GroupedEmbeddingDataset, Decoder
from torch.utils.data import DataLoader

# CONFIG - Match your training script
EMBED_PATH = Path("/home/dlf903/erda_mount/GAIN/Histology/processed_gigapath/slide_embeddings")
LABEL_PATH = Path("/home/dlf903/Histology_v2/labels_binary.xlsx")
MODEL_ROOT = Path("/home/dlf903/Histology_v2/Decoder/models")
TARGET_LABEL = "Approx_Nancy_Score"
MAX_SLIDES = 8
INPUT_DIM = 768 * MAX_SLIDES

def evaluate():
    # 1. Load data
    grouped = load_and_group_data(LABEL_PATH, EMBED_PATH, TARGET_LABEL)
    
    # Evaluate Fold 0 (matches your most recent run)
    fold = 0
    checkpoint_path = MODEL_ROOT / f"{TARGET_LABEL}_fold{fold}" / "best-checkpoint.ckpt"
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    # 2. Load Model
    model = Decoder.load_from_checkpoint(checkpoint_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # 3. Stratified Split (Seed 42)
    from sklearn.model_selection import StratifiedKFold
    labels = [d['label'] for d in grouped]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(grouped, labels))
    train_idx, val_idx = splits[fold]
    
    val_data = [grouped[i] for i in val_idx]
    
    # --- Class Distribution Check ---
    val_labels = [d['label'] for d in val_data]
    counts = Counter(val_labels)
    class_names = ["N0", "N1", "N2", "N3", "N4", "N?"] # N? handles unexpected indices
    
    print("\n" + "="*30)
    print(f" VALIDATION SET STATS (FOLD {fold})")
    print("="*30)
    for i in range(len(class_names)):
        print(f"{class_names[i]}: {counts.get(i, 0)} samples")
    print("-" * 30)

    # 4. Inference
    dataset = GroupedEmbeddingDataset(val_data, MAX_SLIDES, training=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    all_preds = []
    all_targets = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.numpy())

    # 5. Final Metrics
    print("\n--- CLASSIFICATION REPORT ---")
    # Mapping existing labels to names
    present_classes = sorted(list(set(all_targets) | set(all_preds)))
    target_names = [class_names[i] for i in present_classes]
    
    print(classification_report(all_targets, all_preds, target_names=target_names))
    
    print("\n--- CONFUSION MATRIX ---")
    print(confusion_matrix(all_targets, all_preds))

if __name__ == "__main__":
    evaluate()
