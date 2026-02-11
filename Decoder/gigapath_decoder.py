import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from functools import reduce

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Paths
EMBED_PATH = Path("/home/dlf903/erda_mount/GAIN/Histology/processed_gigapath/slide_embeddings")
LABEL_PATH = Path("/home/dlf903/Histology_v2/labels_binary.xlsx")
OUT_PATH   = Path("/home/dlf903/Histology_v2/Decoder/models")

# Label Configuration
# Options: 
# 1. "Approx_Nancy_Score" (Multiclass - single string)
# 2. List of strings (Multi-Label Classification)
'''
TARGET_LABEL = [
    'Neutrofile_LP', 
    'Kryptitis', 
    'Kryptabscesser', 
    'Ulcerationer_Erosioner', 
    'Basal_Lymfoplasmacytose', 
    'Forstyrret_Arkitektur', 
    'Kronisk_Inflammation', 
    'Akut_Inflammation', 
    'Eosinofili', 
    'Epitelioide_Granulomer', 
    'Intraepitelial_Lymfocytose', 
    'Subepitelial_Kollagenfortykkelse', 
    'Panethcellemetaplasi', 
    'Mucindepleteret_Epitel', 
    'Let_Til_Moderat_Aktivitet'
]
'''

TARGET_LABEL = "Approx_Nancy_Score"

# Grouping Columns (to prevent leakage and combine slides)
# We group by these to identify a unique "Case"
GROUP_COLS = ["Rekvn_nr", "Dato"] 

# Data Handling
MAX_SLIDES_PER_CASE = 8  # Pad/Cut input to this number of slides
INPUT_EMBED_DIM     = 768 # Dimension of Gigapath embedding

# Model Hyperparameters
SEED          = 42
BATCH_SIZE    = 8
LR            = 1e-4
EPOCHS        = 100
DROPOUT       = 0.5

# Augmentation (Stochastic Embeddings)
AUGMENTATION      = True
AUGMENTATION_PROB = 0.5
NUM_STOCHASTIC_SAMPLES = 10 # slide_embedding_001.npy to ...010.npy

# Experiment
PROJECT_NAME = "gigapath_decoder_v2"
N_SPLITS     = 5  # For CV

# -----------------------------------------------------------------------------
# DATASET & DATAMODULE
# -----------------------------------------------------------------------------

def load_and_group_data(label_path, embed_root, target_col, verbose=True):
    """
    Reads labels, filters for HE stains only, groups by case, 
    and verifies embeddings exist in the folder structure.
    """
    if verbose: 
        print(f"[System] Loading labels from: {label_path}")
    
    try:
        # Load data (handling CSV or Excel)
        if str(label_path).endswith('.csv'):
            df = pd.read_csv(label_path)
        else:
            df = pd.read_excel(label_path)
    except Exception as e:
        print(f"[Error] Failed to read label file: {e}")
        return []

    # --- NEW: Filter for HE Stain Only ---
    initial_count = len(df)
    df = df[df['stain_type'] == "HE"].copy()
    if verbose:
        print(f"[Filter] Kept {len(df)} HE rows out of {initial_count} total rows.")

    # 1. Handle Missing Targets
    subset_cols = target_col if isinstance(target_col, list) else [target_col]
    df = df.dropna(subset=subset_cols)
    
    # 2. Create Unique Case IDs (Grouping by Rekvn_nr and Dato)
    safe_cols = [df[c].astype(str) for c in GROUP_COLS]
    df['case_id'] = reduce(lambda x, y: x + '_' + y, safe_cols)
    
    grouped_data = []
    missing_slides = 0
    total_slides_found = 0
    
    # 3. Iterate over Clinical Cases
    for case_id, group in df.groupby('case_id'):
        slide_paths = []
        # filenames contain the .dcm extension
        filenames = group['filename'].unique() 
        
        # Prepare labels
        if isinstance(target_col, list):
            # Multi-label float vector for binary features
            label_val = group[target_col].iloc[0].values.astype(np.float32).tolist()
        else:
            # Single-label int for Nancy Score
            label_val = int(group[target_col].iloc[0])
        
        for fname in filenames:
            # Match folder name (Stem) as per your tiling script
            folder_name = Path(fname).stem
            full_emb_path = Path(embed_root) / folder_name / "slide_embedding_full.npy"
            
            if full_emb_path.exists():
                slide_paths.append(Path(embed_root) / folder_name)
            else:
                missing_slides += 1
                
        if len(slide_paths) > 0:
            grouped_data.append({
                'case_id': case_id,
                'slide_dirs': slide_paths,
                'label': label_val,
                'num_slides': len(slide_paths)
            })
            total_slides_found += len(slide_paths)
            
    if verbose:
        print(f"[Summary] Found {len(grouped_data)} HE-only cases.")
        print(f"[Summary] Total HE embeddings found: {total_slides_found}")
        if missing_slides > 0:
            print(f"[Note] {missing_slides} HE slides were missing embeddings on disk.")
        
    return grouped_data

class GroupedEmbeddingDataset(Dataset):
    def __init__(self, data_list, max_slides, training=False):
        self.data_list = data_list
        self.max_slides = max_slides
        self.training = training
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        slide_dirs = item['slide_dirs']
        label_val = item['label']
        
        current_slides = slide_dirs
        
        # Random Sampling or Truncation
        if len(current_slides) > self.max_slides:
            if self.training:
                current_slides = random.sample(current_slides, self.max_slides)
            else:
                current_slides = current_slides[:self.max_slides]
        
        embeddings = []
        for s_dir in current_slides:
            # Augmentation logic
            use_stochastic = False
            if self.training and AUGMENTATION and (np.random.rand() < AUGMENTATION_PROB):
                use_stochastic = True
                
            emb_vec = None
            if use_stochastic:
                # Pick 1..NUM_STOCHASTIC_SAMPLES
                k = np.random.randint(1, NUM_STOCHASTIC_SAMPLES + 1)
                p = s_dir / f"slide_embedding_{k:03d}.npy"
                if p.exists():
                    try:
                        emb_vec = np.load(p)
                    except: pass
            
            if emb_vec is None:
                p = s_dir / "slide_embedding_full.npy"
                # Fallback
                if not p.exists():
                     emb_vec = np.zeros(INPUT_EMBED_DIM, dtype=np.float32)
                else:
                    emb_vec = np.load(p)
                
            embeddings.append(emb_vec)
            
        # Stack
        embeddings = np.stack(embeddings) # (N_actual, 768)
        
        # Pad if necessary
        n_curr = embeddings.shape[0]
        n_pad = self.max_slides - n_curr
        
        if n_pad > 0:
            pad = np.zeros((n_pad, INPUT_EMBED_DIM), dtype=embeddings.dtype)
            embeddings = np.concatenate([embeddings, pad], axis=0)
            
        # Convert to tensor and flatten
        x = torch.from_numpy(embeddings.astype(np.float32)).view(-1)
        
        # Handle label type
        if isinstance(label_val, (np.ndarray, list)):
            y = torch.tensor(label_val, dtype=torch.float32)
        elif isinstance(label_val, (int, np.integer)):
            y = torch.tensor(label_val, dtype=torch.long)
        else:
            y = torch.tensor(label_val, dtype=torch.float32)

        return x, y

class GroupedDataModule(pl.LightningDataModule):
    def __init__(self, grouped_data, batch_size=8, seed=42, fold_idx=0, n_splits=5):
        super().__init__()
        self.grouped_data = grouped_data
        self.batch_size = batch_size
        self.seed = seed
        self.fold_idx = fold_idx
        self.n_splits = n_splits
        self.task_type = "multiclass"
        self.num_classes = 1
        self.class_weights = None
        
        # Determine number of classes / Label stats
        self._analyze_labels()

    def _analyze_labels(self):
        labels = [d['label'] for d in self.grouped_data]
        sample_label = labels[0]
        
        # Case 1: Multi-Label (Array/List of binaries)
        if isinstance(sample_label, (np.ndarray, list)):
            self.task_type = "multilabel"
            self.num_classes = len(sample_label)
            
            # Calculate pos weights for each class
            stacked = np.stack(labels)
            
            # For each class, calc pos_weight = (neg / pos)
            pos_counts = stacked.sum(axis=0)
            total = len(labels)
            neg_counts = total - pos_counts
            
            # Avoid division by zero
            pos_counts = np.clip(pos_counts, 1, None) 
            
            weights = neg_counts / pos_counts
            self.class_weights = torch.from_numpy(weights).float()
            
            print(f"Task: Multi-Label Classification ({self.num_classes} labels).")
            print(f"Pos Weights per class: {self.class_weights}")
            return

        self.unique_labels = sorted(list(set(labels)))
        
        # Check if binary (0, 1) or multiclass
        is_binary_values = set(labels).issubset({0, 1, 0.0, 1.0})
        
        if is_binary_values and len(self.unique_labels) <= 2:
            self.task_type = "binary"
            self.num_classes = 1
            pos = sum(labels)
            neg = len(labels) - pos
            self.class_weights = torch.tensor([neg/pos]) if pos > 0 else None
            print(f"Task: Binary Classification. Pos: {pos}, Neg: {neg}")
        else:
            self.task_type = "multiclass"
            self.num_classes = len(self.unique_labels)
            
            if max(labels) >= self.num_classes:
                 print("Warning: Max label index >= num_classes. Ensure labels are 0-indexed integers.")
            
            counts = pd.Series(labels).value_counts().sort_index()
            weights = 1.0 / counts
            weights = weights / weights.sum()
            self.class_weights = torch.from_numpy(weights.values).float()
            print(f"Task: Multiclass ({self.num_classes}). Weights: {self.class_weights}")

    def setup(self, stage=None):
        lbls = [d['label'] for d in self.grouped_data]
        
        # Stratification Strategy
        if self.task_type == "multilabel":
            # Convert arrays to strings for stratification buckets
            strat_labels = [str(l) for l in lbls]
        else:
            strat_labels = lbls
            
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        
        try:
            splits = list(skf.split(self.grouped_data, strat_labels))
        except ValueError:
            print("Warning: StratifiedKFold failed (likely rare label combinations). Falling back to standard KFold.")
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            splits = list(kf.split(self.grouped_data))
        
        train_idx, val_idx = splits[self.fold_idx]
        
        self.train_data = [self.grouped_data[i] for i in train_idx]
        self.val_data   = [self.grouped_data[i] for i in val_idx]
        
        self.train_ds = GroupedEmbeddingDataset(self.train_data, MAX_SLIDES_PER_CASE, training=True)
        self.val_ds   = GroupedEmbeddingDataset(self.val_data, MAX_SLIDES_PER_CASE, training=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, class_weights=None, reduction='mean', task='multiclass'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.task = task
        self.class_weights = class_weights
        self.alpha = alpha

    def forward(self, inputs, targets):
        if self.task == 'binary':
            # inputs are logits.
            # BCE with logits
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.unsqueeze(1).float(), reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = (1 - pt) ** self.gamma * bce_loss
            
            return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()
            
        elif self.task == 'multilabel':
            # Same as binary but over multiple classes
            # Inputs: (B, C), Targets: (B, C)
            
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = (1 - pt) ** self.gamma * bce_loss
            
            if self.class_weights is not None:
                # class_weights shape: (C,)
                # Broadcast across batch
                w = self.class_weights.to(inputs.device).unsqueeze(0) # (1, C)
                focal_loss = focal_loss * w
                
            return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

        else:
            # Multiclass
            log_probs = F.log_softmax(inputs, dim=1)
            probs = torch.exp(log_probs)
            
            # Gather prob of correct class
            # Ensure targets are long for gather
            targets = targets.long()
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            focal_term = (1 - pt) ** self.gamma
            loss = -focal_term * log_pt
            
            if self.class_weights is not None:
                # Apply class weights
                weights = self.class_weights.to(inputs.device)
                at = weights.gather(0, targets)
                loss = loss * at
                
            return loss.mean() if self.reduction == 'mean' else loss.sum()

class Decoder(pl.LightningModule):
    def __init__(self, input_dim, num_classes, task_type, lr=1e-3, dropout=0.5, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.task_type = task_type
        
        # Simple MLP Decoder
        # Input is flattened (MAX_SLIDES * 768)
        
        widths = [input_dim, 1024, 512, 256]
        layers = []
        for i in range(len(widths)-1):
            layers.append(nn.Linear(widths[i], widths[i+1]))
            layers.append(nn.BatchNorm1d(widths[i+1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            
        self.backbone = nn.Sequential(*layers)
        
        if task_type == 'binary':
            final_out = 1
        elif task_type == 'multilabel':
            final_out = num_classes
        else:
            final_out = num_classes
            
        self.head = nn.Linear(widths[-1], final_out)
        
        self.loss_fn = FocalLoss(gamma=2.0, class_weights=class_weights, task=task_type)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        if self.task_type == 'binary':
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            acc = (preds.squeeze() == y).float().mean()
        
        elif self.task_type == 'multilabel':
            probs = torch.sigmoid(logits)
            # Per-label accuracy
            preds = (probs > 0.5).float()
            # Mean accuracy over all labels and samples
            acc = (preds == y).float().mean()
            
        else:
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
            
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        return [optimizer], [scheduler]

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    pl.seed_everything(SEED)
    
    # 1. Load Data
    grouped = load_and_group_data(LABEL_PATH, EMBED_PATH, TARGET_LABEL)
    if len(grouped) == 0:
        print("No valid data found or embeddings missing.")
        return

    # Check max slides found vs configured
    max_found = max([d['num_slides'] for d in grouped])
    print(f"Max slides found in a single group: {max_found}")
    print(f"Configured MAX_SLIDES: {MAX_SLIDES_PER_CASE}. (Inputs will be padded/truncated to this)")
    
    # Sanitize run name if list
    if isinstance(TARGET_LABEL, list):
        run_name = "MultiLabel"
    else:
        run_name = str(TARGET_LABEL)

    # 2. Iterate Folds
    for fold in range(N_SPLITS):
        print(f"\n=== Starting Fold {fold+1}/{N_SPLITS} ===")
        
        dm = GroupedDataModule(grouped, batch_size=BATCH_SIZE, seed=SEED, fold_idx=fold, n_splits=N_SPLITS)
        dm.setup()
        
        # Calculate final input dim
        full_input_dim = MAX_SLIDES_PER_CASE * INPUT_EMBED_DIM
        
        model = Decoder(
            input_dim=full_input_dim,
            num_classes=dm.num_classes,
            task_type=dm.task_type,
            class_weights=dm.class_weights,
            lr=LR,
            dropout=DROPOUT
        )
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_accuracy',
            dirpath=OUT_PATH / f"{run_name}_fold{fold}",
            filename='best-checkpoint',
            save_top_k=1,
            mode='max'
        )
        
        logger = WandbLogger(project=PROJECT_NAME, name=f"{run_name}_fold{fold}", offline=True)
        
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator='auto',
            devices=1,
            logger=logger,
            callbacks=[checkpoint_callback],
            log_every_n_steps=10
        )
        
        trainer.fit(model, dm)
        print(f"Fold {fold} complete. Best model: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()
