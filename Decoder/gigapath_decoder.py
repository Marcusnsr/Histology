import warnings
# Suppress Pydantic warnings from libraries immediately
warnings.filterwarnings("ignore", message=".*The 'repr' attribute with value False.*")
warnings.filterwarnings("ignore", message=".*The 'frozen' attribute with value True.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from functools import reduce
import wandb

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Paths
EMBED_PATH = Path("/Users/marcusnsr/Desktop/Histology/slide_embeddings")
LABEL_PATH = Path("/Users/marcusnsr/Desktop/Histology/labels_binary.xlsx")
OUT_PATH   = Path("/Users/marcusnsr/Desktop/Histology/Decoder/models")

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

GROUP_COLS = ["Rekvn_nr", "Dato"] 

INPUT_EMBED_DIM     = 768
SEED          = 42
BATCH_SIZE    = 8
LR            = 1e-4
EPOCHS        = 100
DROPOUT       = 0.2

AUGMENTATION      = True
AUGMENTATION_PROB = 0.1
NUM_STOCHASTIC_SAMPLES = 10

if isinstance(TARGET_LABEL, list):
 PROJECT_NAME = "gigapath_decoder_binary_features"
else:
 PROJECT_NAME = "gigapath_decoder_nancy_score"

N_SPLITS     = 3

# -----------------------------------------------------------------------------
# DATASET & DATAMODULE
# -----------------------------------------------------------------------------

def load_and_group_data(label_path, embed_root, target_col):
    df = pd.read_excel(label_path)
    initial_count = len(df)
    df = df[df['stain_type'] == "HE"].copy()
    print(f"Kept {len(df)} HE rows out of {initial_count} total rows.")

    # Handle Missing Targets
    subset_cols = target_col if isinstance(target_col, list) else [target_col]
    df = df.dropna(subset=subset_cols)
    
    # Create Unique Case IDs
    safe_cols = [df[c].astype(str) for c in GROUP_COLS]
    df['case_id'] = reduce(lambda x, y: x + '_' + y, safe_cols)
    
    grouped_data = []
    missing_slides = 0
    total_slides_found = 0
    
    # Iterate
    for case_id, group in df.groupby('case_id'):
        slide_paths = []
        filenames = group['filename'].unique() 
        
        # Prepare labels based on type
        if isinstance(target_col, list):
            # Multi-label float vector for binary features (max pooling over group)
            label_val = group[target_col].max().values.astype(np.float32).tolist()
        else:
            # Single-label int for Nancy Score (max pooling over group)
            label_val = int(group[target_col].max())
        
        for fname in filenames:
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
            
    print(f"Found {len(grouped_data)} HE-only cases.")
    print(f"Total HE embeddings found: {total_slides_found}")
    if missing_slides > 0:
        print(f"{missing_slides} HE slides were missing embeddings on disk.")
        
    return grouped_data

class GroupedEmbeddingDataset(Dataset):
    def __init__(self, data_list, training=False):
        self.data_list = data_list
        self.training = training
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        slide_dirs = item['slide_dirs']
        label_val = item['label']
        
        current_slides = slide_dirs
        
        embeddings = []
        for s_dir in current_slides:
            # Augmentation logic
            use_stochastic = False
            if self.training and AUGMENTATION and (np.random.rand() < AUGMENTATION_PROB):
                use_stochastic = True
                
            emb_vec = None
            if use_stochastic:
                k = np.random.randint(1, NUM_STOCHASTIC_SAMPLES + 1)
                p = s_dir / f"slide_embedding_{k:03d}.npy"
                if p.exists():
                    try: emb_vec = np.load(p)
                    except: pass
            
            if emb_vec is None:
                p = s_dir / "slide_embedding_full.npy"
                try: emb_vec = np.load(p)
                except: emb_vec = np.zeros(INPUT_EMBED_DIM, dtype=np.float32)
                
            embeddings.append(emb_vec)
            
        # Stack & Mean Pooling
        if len(embeddings) > 0:
            embeddings = np.stack(embeddings) 
            embeddings = np.mean(embeddings, axis=0) # (1, 768)
        else:
            embeddings = np.zeros(INPUT_EMBED_DIM, dtype=np.float32)

        x = torch.from_numpy(embeddings.astype(np.float32))
        
        # Handle label type
        if isinstance(label_val, (list, np.ndarray)):
            y = torch.tensor(label_val, dtype=torch.float32)
        else:
            y = torch.tensor(label_val, dtype=torch.long)

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
        self.num_classes = None
        self.class_weights = None
        
        self._analyze_labels()

    def _analyze_labels(self):
        labels = [d['label'] for d in self.grouped_data]
        sample_label = labels[0]
        
        if isinstance(sample_label, (list, np.ndarray)):
            #Multilabel (List of binaries)
            self.task_type = "multilabel"
            self.num_classes = len(sample_label)
            
            # Weighted Loss Calculation for Imbalance
            stacked = np.array(labels)
            pos_counts = stacked.sum(0)
            neg_counts = len(labels) - pos_counts
            pos_counts = np.clip(pos_counts, 1, None)
            
            weights = neg_counts / pos_counts
            self.class_weights = torch.from_numpy(weights).float()
            
            print(f"Task: Multi-Label ({self.num_classes} features).")
            print(f"Pos Weights: {self.class_weights}")
            
        else:
            # Multiclass (Nancy Score)
            self.task_type = "multiclass"
            max_label = int(max(labels))
            self.num_classes = 5
            
            # Calculate Class Weights
            counts = pd.Series(labels).value_counts().reindex(range(self.num_classes), fill_value=0).sort_index()
            counts_safe = counts.replace(0, 1) 
            
            weights = 1.0 / counts_safe
            weights = weights / weights.sum()
            self.class_weights = torch.from_numpy(weights.values).float()
            
            print(f"Task: Multiclass ({self.num_classes} classes).")
            print(f"Distribution:\n{counts}")
            print(f"Weights: {self.class_weights}")

    def setup(self, stage=None):
        lbls = [d['label'] for d in self.grouped_data]
        
        if self.task_type == "multilabel":
            strat_labels = [str(l) for l in lbls]
        else:
            strat_labels = lbls
            
        if self.n_splits > 1:
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            try:
                splits = list(skf.split(self.grouped_data, strat_labels))
            except ValueError:
                print("Warning: StratifiedKFold failed. Falling back to standard KFold.")
                kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
                splits = list(kf.split(self.grouped_data))
            
            train_idx, val_idx = splits[self.fold_idx]
        else:
            # Single split (approx 80/20) if n_splits=1
            indices = np.arange(len(self.grouped_data))
            try:
                train_idx, val_idx = train_test_split(
                    indices, test_size=0.2, shuffle=True, random_state=self.seed, stratify=strat_labels
                )
            except ValueError:
                 print("Warning: train_test_split stratify failed. Falling back to random split.")
                 train_idx, val_idx = train_test_split(
                    indices, test_size=0.2, shuffle=True, random_state=self.seed
                 )
        
        self.train_data = [self.grouped_data[i] for i in train_idx]
        self.val_data   = [self.grouped_data[i] for i in val_idx]
        
        self.train_ds = GroupedEmbeddingDataset(self.train_data, training=True)
        self.val_ds   = GroupedEmbeddingDataset(self.val_data, training=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0, persistent_workers=False)

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
        if self.task == 'multilabel':
             # Inputs: (B, C), Targets: (B, C)
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = (1 - pt) ** self.gamma * bce_loss
            
            if self.class_weights is not None:
                w = self.class_weights.to(inputs.device).unsqueeze(0) # (1, C)
                focal_loss = focal_loss * w
                
            return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()
        
        else: # Multiclass
            log_probs = F.log_softmax(inputs, dim=1)
            probs = torch.exp(log_probs)
            
            targets = targets.long()
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            focal_term = (1 - pt) ** self.gamma
            loss = -focal_term * log_pt
            
            if self.class_weights is not None:
                weights = self.class_weights.to(inputs.device)
                at = weights.gather(0, targets)
                loss = loss * at
                
            return loss.mean() if self.reduction == 'mean' else loss.sum()

class Decoder(pl.LightningModule):
    def __init__(self, input_dim, num_classes, task_type='multiclass', lr=1e-3, dropout=0.5, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.task_type = task_type
        
        widths = [input_dim, 1024, 512, 256]
        layers = []
        for i in range(len(widths)-1):
            layers.append(nn.Linear(widths[i], widths[i+1]))
            layers.append(nn.BatchNorm1d(widths[i+1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(widths[-1], num_classes)
        self.loss_fn = FocalLoss(gamma=2.0, class_weights=class_weights, task=task_type)
        
        self.validation_step_outputs = [] # To gather preds/targets for per-epoch metrics

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
        
        if self.task_type == 'multilabel':
             probs = torch.sigmoid(logits)
             preds = (probs > 0.5).float()
             acc = (preds == y).float().mean()
        else:
             # Multiclass
             probs = torch.softmax(logits, dim=1)
             preds = torch.argmax(logits, dim=1)
             acc = (preds == y).float().mean()
            
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', acc, prog_bar=True)
        
        self.validation_step_outputs.append({
            "preds": preds, 
            "probs": probs,
            "targets": y
        })
        
        return loss

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if len(outputs) == 0: return
        
        # Concatenate all batches from the epoch
        all_preds   = torch.cat([x['preds'] for x in outputs])
        all_probs   = torch.cat([x['probs'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])
        
        # MULTICLASS (Nancy Score)
        if self.task_type == 'multiclass':
            # 1. Standard Confusion Matrix
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_targets.cpu().numpy(),
                    preds=all_preds.cpu().numpy(),
                    class_names=[str(i) for i in range(self.hparams.num_classes)]
                )
            })
            
            # 2. Collapsed Metrics (Optimization: Vectorized Lookup)
            # Map: 0,1 -> 0 (Low) | 2,3 -> 1 (Mod) | 4 -> 2 (High)
            lookup = torch.tensor([0, 0, 1, 1, 2], device=self.device)
            
            # Direct lookup (safe since num_classes=5)
            collapsed_preds = lookup[all_preds.long()]
            collapsed_targets = lookup[all_targets.long()]
            
            col_acc = (collapsed_preds == collapsed_targets).float().mean()
            self.log('val_acc_collapsed', col_acc, prog_bar=True)
            
            wandb.log({
                "confusion_matrix_collapsed": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=collapsed_targets.cpu().numpy(),
                    preds=collapsed_preds.cpu().numpy(),
                    class_names=["Low (0-1)", "Mod (2-3)", "High (4)"]
                )
            })
            
        # MULTILABEL (Binary Features)
        elif self.task_type == 'multilabel':
            # Determine feature names
            if isinstance(TARGET_LABEL, list):
                feature_names = TARGET_LABEL
            else:
                feature_names = [f"Feat_{i}" for i in range(self.hparams.num_classes)]
            
            # Convert to CPU numpy
            np_probs = all_probs.cpu().numpy()
            np_preds = all_preds.cpu().numpy()
            np_targets = all_targets.cpu().numpy()
            
            f1_scores = []
            
            for i, feat_name in enumerate(feature_names):
                if i >= np_targets.shape[1]: break
                
                y_true = np_targets[:, i]
                y_pred = np_preds[:, i]
                y_score = np_probs[:, i]
                
                # AUC (Only calculable if both positive and negative samples exist)
                if len(np.unique(y_true)) > 1:
                    try:
                        auc = roc_auc_score(y_true, y_score)
                        self.log(f"val_auc_{feat_name}", auc)
                    except ValueError: pass
                    
                # F1 Score
                f1 = f1_score(y_true, y_pred, zero_division=0)
                self.log(f"val_f1_{feat_name}", f1)
                f1_scores.append(f1)
            
            # Log Bar Chart of F1 Scores per Feature
            data = [[name, f1] for name, f1 in zip(feature_names, f1_scores)]
            table = wandb.Table(data=data, columns=["Feature", "F1 Score"])
            
            wandb.log({
                "per_feature_performance": wandb.plot.bar(table, "Feature", "F1 Score", title="Per-Feature F1 Scores")
            })
            
        # Free memory
        self.validation_step_outputs.clear()

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
        
        model = Decoder(
            input_dim=INPUT_EMBED_DIM,
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
        
        logger = WandbLogger(project=PROJECT_NAME, name=f"{run_name}_fold{fold}", mode="offline")
        
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
        wandb.finish()

if __name__ == "__main__":
    main()