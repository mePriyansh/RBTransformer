import os
import torch
import wandb
import random
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from dotenv import load_dotenv
from collections import Counter
from huggingface_hub import login
from model.model import RBTransformer
from torch.utils.data import DataLoader
from preprocessing import DatasetReshape
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


################################################################################
# PREPROCESSED-DATASETS
################################################################################
PREPROCESSED_DATASETS = {
    "deap": {
        "valence": {
            "binary": "preprocessed_datasets/deap_binary_valence_dataset.pkl",
            "multiclass": "preprocessed_datasets/deap_multi_valence_dataset.pkl",
        },
        "arousal": {
            "binary": "preprocessed_datasets/deap_binary_arousal_dataset.pkl",
            "multiclass": "preprocessed_datasets/deap_multi_arousal_dataset.pkl",
        },
        "dominance": {
            "binary": "preprocessed_datasets/deap_binary_dominance_dataset.pkl",
            "multiclass": "preprocessed_datasets/deap_multi_dominance_dataset.pkl",
        },
    },
    "dreamer": {
        "valence": {
            "binary": "preprocessed_datasets/dreamer_binary_valence_dataset.pkl",
            "multiclass": "preprocessed_datasets/dreamer_multi_valence_dataset.pkl",
        },
        "arousal": {
            "binary": "preprocessed_datasets/dreamer_binary_arousal_dataset.pkl",
            "multiclass": "preprocessed_datasets/dreamer_multi_arousal_dataset.pkl",
        },
        "dominance": {
            "binary": "preprocessed_datasets/dreamer_binary_dominance_dataset.pkl",
            "multiclass": "preprocessed_datasets/dreamer_multi_dominance_dataset.pkl",
        },
    },
    "seed": {
        "emotion": {
            "multiclass": "preprocessed_datasets/seed_multi_dataset.pkl"
        },
    },
}

dataset_name = "dreamer"
label_name = "valence"
task_type = "binary"

dataset_path = PREPROCESSED_DATASETS[dataset_name][label_name][task_type]

with open(dataset_path, "rb") as f:
    dataset = pickle.load(f)

print(f"Loaded dataset from {dataset_path}")


################################################################################
# SEED CONFIG 
################################################################################
SEED = 23
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


################################################################################
# TRAINING-PARAMETERS 
################################################################################
num_epochs = 300
kfolds = 5
initial_batch_size = 256
reduced_batch_size = 64
initial_learning_rate = 1e-3
minimum_learning_rate = 1e-6
weight_decay = 1e-3
label_smoothing = 0.12
num_workers = 4
data_drop_ratio = 0.10


################################################################################
# MODEL-CONFIG 
################################################################################
num_electrodes = 14
bde_dim = 4
embed_dim = 128
depth = 4
heads = 6
head_dim = 32
mlp_hidden_dim = 128
dropout = 0.1
num_classes = 2  # For DEAP and Dreamer Binary-Class Classification Training: 2
                 # For SEED Multi-Class Classification Training: 3
                 # For Dreamer Multi-Class Classification Training: 5
                 # For Deap Multi-Class Classification Training: 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


################################################################################
#  WANDB & HUGGINGFACE CONFIG 
################################################################################
load_dotenv()

# WANDB-CONFIGS
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
wandb.login(key=WANDB_API_KEY)

paper_task = "γ-eeg-recognition"
dataset_name = "dreamer"
benchmark_column = "valence"
task_type = "binary-classification"
random_hash = "sota-run"
test_run_num = "0001"
wandb_run_name = f"{paper_task}-{dataset_name}-{benchmark_column}-{task_type}-{random_hash}-{test_run_num}"


# HF-CONFIG
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

username = "<YOUR_USERNAME>"
base_repo_id = f"{username}/{dataset_name}-{benchmark_column}-{task_type}-Kfold"
def push_model_to_hub(model, repo_id, local_dir="rbtransformer", commit_message="Upload RBTransformer model"):
    model.save_pretrained(local_dir)
    model.push_to_hub(repo_id=repo_id, commit_message=commit_message)
    print(f"Model successfully uploaded to https://huggingface.co/{repo_id}")

################################################################################
# REGULARIZATION: DATA DROPOUT
################################################################################
X_full = []
y_full = []
for i in tqdm(range(len(dataset)), desc="Extracting data for SMOTE"):
    x, y = dataset[i]
    X_full.append(x.squeeze(0).numpy().flatten())
    y_full.append(y)

X_full = np.array(X_full)
y_full = np.array(y_full)

data_drop_ratio = data_drop_ratio
num_samples = len(X_full)
drop_count = int(num_samples * data_drop_ratio)
all_indices = np.arange(num_samples)

np.random.shuffle(all_indices)
kept_indices = all_indices[drop_count:]
X_full = X_full[kept_indices]
y_full = y_full[kept_indices]


################################################################################
# K-FOLD TRAINING  
################################################################################
kf = KFold(n_splits=kfolds, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(y_full)))):

    print(f"\nFold {fold + 1}/{kfolds}")

    X_train = X_full[train_idx]
    y_train = y_full[train_idx]
    X_val = X_full[val_idx]
    y_val = y_full[val_idx]

    smote = SMOTE(random_state=SEED)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    train_dataset = DatasetReshape(X_train_balanced, y_train_balanced, num_electrodes)
    val_dataset = DatasetReshape(X_val, y_val, num_electrodes)

    balanced_counts = Counter(y_train_balanced)
    print(f"\nFold {fold + 1} Training Set Class Balance (After SMOTE):")
    print(f"Total samples: {len(y_train_balanced)}")
    for cls, count in balanced_counts.items():
        print(
            f"Class {cls}: {count} samples ({count / len(y_train_balanced) * 100:.2f}%)"
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=initial_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = RBTransformer(
        num_electrodes=num_electrodes,
        bde_dim=bde_dim,
        embed_dim=embed_dim,
        depth=depth,
        heads=heads,
        head_dim=head_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_learning_rate, weight_decay=weight_decay
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=minimum_learning_rate
    )

    wandb.init(
        project=wandb_run_name,
        group='dreamer-binary-valence',
        name=f'Kfold-Run-{fold+1}',
        config={
            'learning_rate': initial_learning_rate,
            'min_learning_rate': minimum_learning_rate,
            'num_epochs': num_epochs,
            'model': 'RBTransformer',
            'num_folds': kfolds,
            'fold': fold+1,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
        }
    )

    for epoch in range(num_epochs):
        if epoch < 150:
            current_batch_size = initial_batch_size
        else:
            current_batch_size = reduced_batch_size

        train_loader = DataLoader(
            train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=4
        )

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(
            train_loader,
            desc=f"Fold {fold + 1} Epoch {epoch + 1}/{num_epochs} - Training",
        ):
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
        
        scheduler.step()

        train_accuracy = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        val_accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'lr': optimizer.param_groups[0]['lr'],
            'train_batch_size': current_batch_size
        })

        tqdm.write(f"\n[Fold {fold+1}] Epoch {epoch+1}/{num_epochs}")
        tqdm.write(f"Train Loss     : {avg_train_loss:.4f}")
        tqdm.write(f"Train Accuracy : {train_accuracy:.4f}")
        tqdm.write(f"Val Loss       : {avg_val_loss:.4f}")
        tqdm.write(f"Val Accuracy   : {val_accuracy:.4f}")
        tqdm.write(f"Precision      : {precision:.4f}")
        tqdm.write(f"Recall         : {recall:.4f}")
        tqdm.write(f"F1 Score       : {f1:.4f}")
        tqdm.write(f"Learning Rate  : {optimizer.param_groups[0]['lr']:.6f}")
        tqdm.write(f"Batch Size     : {current_batch_size}")
        
    push_model_to_hub(
        model=model,
        repo_id=f"{base_repo_id}-{fold+1}",
        commit_message=f"Upload of trained RBTransformer on Kfold-{fold+1} run"
    )
    
    wandb.finish()

