import os
import torch
import wandb
import pickle
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils.seed import set_seed
from model.model import RBTransformer
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler
from utils.messages import success, fail
from utils.pickle_patch import patch_pickle_loading
from preprocessing.transformations import DatasetReshape
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="RBTransformer Transfer Learning on PhysioNet MI"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["scratch", "finetune", "frozen"],
        help="scratch: random init | finetune: load SEED weights, train all | frozen: load SEED weights, freeze encoder",
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
        help="Path to SEED pretrained model directory (required for finetune/frozen)",
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
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=300)
    return parser.parse_args()


def load_pretrained_encoder(model, pretrained_path):
    """
    Loads SEED-pretrained weights into the encoder (BDE projection + transformer blocks).
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

    print(success(f"Transferred {len(transferred_keys)} parameter tensors from SEED pretrained model"))
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


def main():
    args = parse_args()

    if args.mode in ["finetune", "frozen"] and args.pretrained_path is None:
        raise ValueError("--pretrained_path is required for finetune/frozen modes")

    NUM_ELECTRODES = 64
    NUM_CLASSES = 4
    BDE_DIM = 4
    EMBED_DIM = 128
    DEPTH = 4
    HEADS = 6
    HEAD_DIM = 32
    MLP_HIDDEN_DIM = 128
    DROPOUT = 0.1

    NUM_EPOCHS = args.num_epochs
    INITIAL_BATCH_SIZE = 256
    REDUCED_BATCH_SIZE = 64
    INITIAL_LEARNING_RATE = 1e-3
    MINIMUM_LEARNING_RATE = 1e-6
    WEIGHT_DECAY = 1e-3
    LABEL_SMOOTHING = 0.12
    DATA_DROP_RATIO = 0.10

    SEED_VAL = args.seed
    set_seed(SEED_VAL)
    DEVICE = torch.device(args.device)

    run_name = f"physionet-mi-{args.mode}-data{int(args.data_fraction*100)}pct"
    print(success(f"Run: {run_name} | Device: {DEVICE}"))

    patch_pickle_loading()
    try:
        with open(args.dataset_path, "rb") as f:
            dataset = pickle.load(f)
        print(success(f"Dataset loaded: {len(dataset)} samples"))
    except Exception as e:
        print(fail(f"Failed to load dataset: {args.dataset_path}"))
        raise e

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

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=SEED_VAL, stratify=y_full
    )

    if args.data_fraction < 1.0:
        n_keep = int(len(X_train) * args.data_fraction)
        indices = np.random.choice(len(X_train), n_keep, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(success(f"Using {args.data_fraction*100:.0f}% of training data: {len(X_train)} samples"))

    smote = SMOTE(random_state=SEED_VAL)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(success(f"After SMOTE: {len(X_train_balanced)} train samples, {len(X_test)} test samples"))

    train_dataset = DatasetReshape(X_train_balanced, y_train_balanced, NUM_ELECTRODES)
    test_dataset = DatasetReshape(X_test, y_test, NUM_ELECTRODES)

    test_loader = DataLoader(
        test_dataset, batch_size=INITIAL_BATCH_SIZE, shuffle=False, num_workers=args.num_workers
    )

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

    if args.mode in ["finetune", "frozen"]:
        model = load_pretrained_encoder(model, args.pretrained_path)

    if args.mode == "frozen":
        model = freeze_encoder(model)

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

    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
        wandb.init(
            project="rbtransformer-transfer-learning",
            name=run_name,
            config={
                "mode": args.mode,
                "data_fraction": args.data_fraction,
                "num_epochs": NUM_EPOCHS,
                "num_electrodes": NUM_ELECTRODES,
                "num_classes": NUM_CLASSES,
                "pretrained_path": args.pretrained_path,
            },
        )

    best_val_accuracy = 0.0
    best_metrics = {}

    for epoch in range(NUM_EPOCHS):
        current_batch_size = INITIAL_BATCH_SIZE if epoch < 150 else REDUCED_BATCH_SIZE

        train_loader = DataLoader(
            train_dataset,
            batch_size=current_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training"):
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
            for batch in test_loader:
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_metrics = {
                "epoch": epoch + 1,
                "accuracy": val_accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
            torch.save(model.state_dict(), f"best_model_{run_name}.pt")

        if args.wandb_api_key:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "lr": optimizer.param_groups[0]["lr"],
                "batch_size": current_batch_size,
            })

        tqdm.write(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        tqdm.write(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        tqdm.write(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        tqdm.write(f"P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")

    print("\n" + "=" * 60)
    print(success(f"FINAL RESULTS: {run_name}"))
    print(f"  Best Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"  Best Precision: {best_metrics['precision']:.4f}")
    print(f"  Best Recall:    {best_metrics['recall']:.4f}")
    print(f"  Best F1 Score:  {best_metrics['f1_score']:.4f}")
    print(f"  Best Epoch:     {best_metrics['epoch']}")
    print("=" * 60)

    if args.wandb_api_key:
        wandb.log({"best_accuracy": best_metrics["accuracy"], "best_f1": best_metrics["f1_score"]})
        wandb.finish()


if __name__ == "__main__":
    main()
