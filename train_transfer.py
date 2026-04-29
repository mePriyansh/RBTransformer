import os
import torch
import wandb
import pickle
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils.seed import set_seed
from huggingface_hub import login
from model.model import RBTransformer
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from utils.push_to_hf import push_model_to_hub
import torch.optim.lr_scheduler as lr_scheduler
from utils.messages import success, fail
from utils.pickle_patch import patch_pickle_loading
from preprocessing.transformations import DatasetReshape
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

################################################################################
# ARGPARSE CONFIGURATION
################################################################################


def parse_args():
    parser = argparse.ArgumentParser(
        description="RBTransformer Transfer Learning Training Script"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["scratch", "finetune", "frozen", "soup"],
        help=(
            "scratch: random init | "
            "finetune: load pretrained weights, train all | "
            "frozen: load pretrained weights, freeze encoder | "
            "soup: load full merged-teacher checkpoint (all weights) and train all"
        ),
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=1.0,
        choices=[1.0, 0.25, 0.10],
        help="Fraction of training data to use",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="HuggingFace repo ID or local path to pretrained model (required for finetune/frozen)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="preprocessed_datasets/physionet_mi_multi_motor_imagery_dataset.pkl",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--hf_username", type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--wandb_api_key", type=str, required=True)
    return parser.parse_args()


################################################################################
# TRANSFER LEARNING UTILITIES
################################################################################


def load_full_checkpoint(model, pretrained_path):
    """
    Loads ALL weights from a checkpoint that already matches the student's
    target shape (e.g., a merged-teacher 'soup' checkpoint produced by
    merge_teachers.py). Unlike load_pretrained_encoder, this transfers the
    electrode identity embedding and the deep classification head as well,
    relying on the producer of the checkpoint to have shaped everything to
    match the student.
    """
    pretrained = RBTransformer.from_pretrained(pretrained_path)
    pretrained_state = pretrained.state_dict()
    model_state = model.state_dict()

    transferred_keys = []
    skipped_keys = []
    for key, value in pretrained_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            transferred_keys.append(key)
        else:
            skipped_keys.append(key)

    model.load_state_dict(model_state)
    print(success(f"Soup init: transferred {len(transferred_keys)} parameter tensors from {pretrained_path}"))
    if skipped_keys:
        print(f"  Skipped (shape mismatch or missing in student): {skipped_keys}")
    return model


def load_pretrained_encoder(model, pretrained_path):
    """
    Loads pretrained weights into the encoder (BDE projection + transformer blocks).
    Electrode identity embedding and classification head are left as random init
    since they differ in shape (62->64 electrodes, 3->4 classes).
    """
    pretrained = RBTransformer.from_pretrained(pretrained_path)
    pretrained_state = pretrained.state_dict()
    model_state = model.state_dict()

    transferred_keys = []
    skipped_keys = []

    for key in pretrained_state:
        if "electrode_id_embedding" in key:
            skipped_keys.append(key)
            continue
        if "classification_head" in key:
            skipped_keys.append(key)
            continue

        if key in model_state and pretrained_state[key].shape == model_state[key].shape:
            model_state[key] = pretrained_state[key]
            transferred_keys.append(key)
        else:
            skipped_keys.append(key)

    model.load_state_dict(model_state)

    print(success(f"Transferred {len(transferred_keys)} parameter tensors from pretrained model"))
    print(f"  Transferred: {transferred_keys}")
    print(f"  Skipped (reinitialized): {skipped_keys}")

    return model


def freeze_encoder(model):
    """
    Freezes the encoder: BDE projection + all transformer blocks.
    Only electrode identity embedding and classification head remain trainable.
    """
    frozen_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        if "electrode_id_embedding" in name or "classification_head" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()

    print(success(f"Frozen encoder: {frozen_params:,} params frozen, {trainable_params:,} params trainable"))

    return model


################################################################################
# MAIN-FUNCTION
################################################################################
def main():
    args = parse_args()

    if args.mode in ["finetune", "frozen", "soup"] and args.pretrained_path is None:
        raise ValueError("--pretrained_path is required for finetune/frozen/soup modes")

    ################################################################################
    # RUN-CONFIG
    ################################################################################
    DATA_FRACTION = args.data_fraction
    MODE = args.mode
    run_name = f"physionet-mi-{MODE}-data{int(DATA_FRACTION*100)}pct"

    print(
        success(
            f"Training Initialized => Dataset: PHYSIONET_MI || Mode: {MODE.capitalize()} || Data: {int(DATA_FRACTION*100)}%"
        )
    )

    ################################################################################
    # SEED CONFIG
    ################################################################################
    SEED_VAL = args.seed
    set_seed(SEED_VAL)
    print(success(f"Seed value set for training run: {SEED_VAL}"))

    ################################################################################
    # TRAINING-HYPERPARAMETERS
    ################################################################################
    NUM_EPOCHS = 100

    INITIAL_BATCH_SIZE = 256

    REDUCED_BATCH_SIZE = 64

    INITIAL_LEARNING_RATE = 5e-4

    MINIMUM_LEARNING_RATE = 1e-6

    WEIGHT_DECAY = 1e-3

    LABEL_SMOOTHING = 0.12

    NUM_WORKERS = args.num_workers

    DATA_DROP_RATIO = 0.10

    ################################################################################
    # MODEL-CONFIG
    ################################################################################
    NUM_ELECTRODES = 64

    NUM_CLASSES = 4

    BDE_DIM = 4

    EMBED_DIM = 128

    DEPTH = 4

    HEADS = 6

    HEAD_DIM = 32

    MLP_HIDDEN_DIM = 128

    DROPOUT = 0.1

    DEVICE = torch.device(args.device)
    print(success(f"Device Set: {DEVICE}"))

    ################################################################################
    #  WANDB & HUGGINGFACE CONFIG
    ################################################################################
    try:
        wandb.login(key=args.wandb_api_key)
        print(success("Success: WandB API key authenticated."))
    except Exception as e:
        print(fail("Failed: WandB API key authentication failed."))
        raise e

    WANDB_RUN_NAME = f"α-rbtransformer-transfer-{run_name}"

    try:
        login(token=args.hf_token)
        print(success("Success: Hugging Face token authenticated."))
    except Exception as e:
        print(fail("Failed: Hugging Face token authentication failed."))
        raise e

    USERNAME = args.hf_username
    HF_REPO_ID = f"{USERNAME}/{run_name}"

    ################################################################################
    # LOAD PREPROCESSED DATASET
    ################################################################################
    patch_pickle_loading()

    try:
        with open(args.dataset_path, "rb") as f:
            dataset = pickle.load(f)
        print(success(f"Success: Dataset '{args.dataset_path}' successfully loaded"))
    except Exception as e:
        print(fail(f"Failed: Dataset '{args.dataset_path}' failed to load"))
        raise e

    ################################################################################
    # REGULARIZATION: DATA DROPOUT
    ################################################################################
    X_full = []
    y_full = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        X_full.append(x.squeeze(0).numpy().flatten())
        y_full.append(y)

    X_full = np.array(X_full)
    y_full = np.array(y_full)

    num_samples = len(X_full)
    drop_count = int(num_samples * DATA_DROP_RATIO)
    all_indices = np.arange(num_samples)

    np.random.shuffle(all_indices)
    kept_indices = all_indices[drop_count:]
    X_full = X_full[kept_indices]
    y_full = y_full[kept_indices]

    ################################################################################
    # TRAIN/TEST SPLIT
    ################################################################################
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=SEED_VAL, stratify=y_full
    )

    if DATA_FRACTION < 1.0:
        n_keep = int(len(X_train) * DATA_FRACTION)
        indices = np.random.choice(len(X_train), n_keep, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(success(f"Using {DATA_FRACTION*100:.0f}% of training data: {len(X_train)} samples"))

    smote = SMOTE(random_state=SEED_VAL)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    train_dataset = DatasetReshape(X_train_balanced, y_train_balanced, NUM_ELECTRODES)
    val_dataset = DatasetReshape(X_val, y_val, NUM_ELECTRODES)

    val_loader = DataLoader(
        val_dataset,
        batch_size=INITIAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    ################################################################################
    # MODEL INIT + TRANSFER LEARNING
    ################################################################################
    model = RBTransformer(
        num_electrodes=NUM_ELECTRODES,
        bde_dim=BDE_DIM,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        heads=HEADS,
        head_dim=HEAD_DIM,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
    )

    if MODE in ["finetune", "frozen"]:
        model = load_pretrained_encoder(model, args.pretrained_path)

    if MODE == "frozen":
        model = freeze_encoder(model)

    if MODE == "soup":
        model = load_full_checkpoint(model, args.pretrained_path)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=INITIAL_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=MINIMUM_LEARNING_RATE
    )

    wandb.init(
        project=WANDB_RUN_NAME,
        group=f"physionet_mi-{MODE}",
        name=run_name,
        config={
            "learning_rate": INITIAL_LEARNING_RATE,
            "minimum_learning_rate": MINIMUM_LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "model": "RBTransformer",
            "mode": MODE,
            "data_fraction": DATA_FRACTION,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "pretrained_path": args.pretrained_path,
        },
    )

    ################################################################################
    # TRAINING
    ################################################################################
    for epoch in range(NUM_EPOCHS):
        if epoch < 50:
            current_batch_size = INITIAL_BATCH_SIZE
        else:
            current_batch_size = REDUCED_BATCH_SIZE

        train_loader = DataLoader(
            train_dataset,
            batch_size=current_batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training",
        ):
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
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
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        val_accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(
            all_targets, all_preds, average="macro", zero_division=0
        )
        recall = recall_score(
            all_targets, all_preds, average="macro", zero_division=0
        )
        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "lr": optimizer.param_groups[0]["lr"],
                "train_batch_size": current_batch_size,
            }
        )

        tqdm.write(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
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
        repo_id=HF_REPO_ID,
        commit_message=f"Upload of trained RBTransformer: {run_name}",
    )

    wandb.finish()


if __name__ == "__main__":
    main()
