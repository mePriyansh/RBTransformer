"""
Naive weight-averaging merge of K pretrained RBTransformer teacher checkpoints
into a single PhysioNet-MI student initialization.

For each parameter tensor in the student state_dict:
  - If all teachers have a tensor of identical shape, take the element-wise
    mean across teachers (the standard model-soup average).
  - The electrode_id_embedding tensor has a different (1, N, D) shape per
    teacher because each dataset has a different electrode count. Each
    teacher's embedding is first spatially interpolated to the target layout
    (PhysioNet-MI 64 channels) using a Gaussian kernel over scalp positions,
    after which the three (1, 64, D) embeddings are averaged.
  - The final classifier (classification_head.mlp.9) cannot be transferred
    (different num_classes per dataset) and is kept at the student's random
    init.

The resulting student is saved as a HuggingFace-format checkpoint that can be
loaded directly via RBTransformer.from_pretrained(<output_dir>).
"""

import argparse
import torch

from model.model import RBTransformer
from utils.electrode_layouts import (
    DATASET_CHANNELS,
    get_electrode_positions,
)
from utils.spatial_interpolation import interpolate_electrode_embedding
from utils.messages import success, fail


# Maps a CLI dataset key to (HF teacher repo id, layout key in DATASET_CHANNELS).
DEFAULT_TEACHERS = {
    "deap": ("nnilayy/deap-multi-valence-Kfold-1", "deap"),
    "dreamer": ("nnilayy/dreamer-multi-valence-Kfold-1", "dreamer"),
    "seed": ("nnilayy/seed-multi-emotion-Kfold-1", "seed"),
}


# Parameter keys that can never be merged: their shapes depend on the teacher's
# label space, which is unrelated to the student's task.
NON_MERGEABLE_KEYS = (
    "classification_head.mlp.9.weight",
    "classification_head.mlp.9.bias",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Naively merge RBTransformer teacher checkpoints into a single PhysioNet-MI student init."
    )
    parser.add_argument(
        "--deap_teacher",
        type=str,
        default=DEFAULT_TEACHERS["deap"][0],
        help="HF repo id of the DEAP teacher (32 electrodes).",
    )
    parser.add_argument(
        "--dreamer_teacher",
        type=str,
        default=DEFAULT_TEACHERS["dreamer"][0],
        help="HF repo id of the DREAMER teacher (14 electrodes).",
    )
    parser.add_argument(
        "--seed_teacher",
        type=str,
        default=DEFAULT_TEACHERS["seed"][0],
        help="HF repo id of the SEED teacher (62 electrodes).",
    )
    parser.add_argument(
        "--target_layout",
        type=str,
        default="physionet_mi",
        choices=list(DATASET_CHANNELS.keys()),
    )
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--bde_dim", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--head_dim", type=int, default=32)
    parser.add_argument("--mlp_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--interp_sigma",
        type=float,
        default=0.04,
        help="Gaussian-kernel bandwidth (meters) for electrode-embedding interpolation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./merged_checkpoints/physionet_mi_soup",
        help="Local directory to save the merged HF-format checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def load_teacher(repo_id, layout_key):
    print(f"Loading teacher {repo_id} ...")
    teacher = RBTransformer.from_pretrained(repo_id)
    teacher.eval()
    expected_n = len(DATASET_CHANNELS[layout_key])
    actual_n = teacher.electrode_id_embedding.embedding.shape[1]
    assert actual_n == expected_n, (
        f"Teacher {repo_id} has {actual_n} electrodes but layout {layout_key} "
        f"expects {expected_n}. Did the teacher's electrode count change?"
    )
    return teacher


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    target_channels = DATASET_CHANNELS[args.target_layout]
    target_positions = get_electrode_positions(target_channels)
    n_target = len(target_channels)

    student = RBTransformer(
        num_electrodes=n_target,
        bde_dim=args.bde_dim,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        head_dim=args.head_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        dropout=args.dropout,
        num_classes=args.num_classes,
    )
    print(success(f"Initialized student with target layout '{args.target_layout}' (N={n_target}, classes={args.num_classes})"))

    teacher_specs = [
        ("deap", args.deap_teacher),
        ("dreamer", args.dreamer_teacher),
        ("seed", args.seed_teacher),
    ]
    teachers = []
    for layout_key, repo_id in teacher_specs:
        teacher = load_teacher(repo_id, layout_key)
        teachers.append(
            {
                "layout_key": layout_key,
                "repo_id": repo_id,
                "model": teacher,
                "state_dict": teacher.state_dict(),
                "channels": DATASET_CHANNELS[layout_key],
                "positions": get_electrode_positions(DATASET_CHANNELS[layout_key]),
            }
        )
    print(success(f"Loaded {len(teachers)} teachers"))

    interpolated_embeddings = []
    for t in teachers:
        src_emb = t["state_dict"]["electrode_id_embedding.embedding"]
        interp = interpolate_electrode_embedding(
            source_embedding=src_emb,
            source_positions=t["positions"],
            target_positions=target_positions,
            sigma=args.interp_sigma,
        )
        assert interp.shape == (1, n_target, args.embed_dim), (
            f"Interpolated shape {tuple(interp.shape)} != expected (1, {n_target}, {args.embed_dim})"
        )
        interpolated_embeddings.append(interp)
        print(
            f"  Interpolated {t['layout_key']:8s} ({len(t['channels']):2d} ch -> {n_target} ch) ok"
        )

    student_sd = student.state_dict()
    merged_sd = {}
    averaged_keys = []
    spatial_interp_keys = []
    skipped_keys = []

    for key, target_tensor in student_sd.items():
        if key == "electrode_id_embedding.embedding":
            stacked = torch.stack(interpolated_embeddings, dim=0)
            merged = stacked.mean(dim=0)
            assert merged.shape == target_tensor.shape, (
                f"Merged electrode embedding shape {tuple(merged.shape)} != student shape {tuple(target_tensor.shape)}"
            )
            merged_sd[key] = merged
            spatial_interp_keys.append(key)
            continue

        if key in NON_MERGEABLE_KEYS:
            merged_sd[key] = target_tensor.clone()
            skipped_keys.append(key)
            continue

        teacher_tensors = []
        all_match = True
        for t in teachers:
            if key not in t["state_dict"]:
                all_match = False
                break
            tt = t["state_dict"][key]
            if tt.shape != target_tensor.shape:
                all_match = False
                break
            teacher_tensors.append(tt)

        if all_match and len(teacher_tensors) == len(teachers):
            stacked = torch.stack(teacher_tensors, dim=0).to(dtype=target_tensor.dtype)
            merged_sd[key] = stacked.mean(dim=0)
            averaged_keys.append(key)
        else:
            merged_sd[key] = target_tensor.clone()
            skipped_keys.append(key)

    student.load_state_dict(merged_sd, strict=True)

    print()
    print(success(f"Merge summary"))
    print(f"  Averaged across {len(teachers)} teachers : {len(averaged_keys)} tensors")
    print(f"  Spatially interpolated + averaged       : {len(spatial_interp_keys)} tensors  -> {spatial_interp_keys}")
    print(f"  Kept at random init (non-mergeable)     : {len(skipped_keys)} tensors  -> {skipped_keys}")

    output_dir = args.output_dir
    student.save_pretrained(output_dir)
    print()
    print(success(f"Merged student saved to: {output_dir}"))
    print(
        f"Use it with: python train_transfer.py --mode soup --pretrained_path {output_dir} ..."
    )


if __name__ == "__main__":
    main()
