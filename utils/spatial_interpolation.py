"""
Spatial interpolation of per-electrode embeddings between EEG layouts.

Used to remap a teacher's learned electrode_id_embedding (shape (1, N_src, D))
to a target layout (shape (1, N_tgt, D)) so it can be merged with other
teachers and a target student. The interpolation uses a Gaussian kernel over
3D scalp positions: each target electrode receives a softmax-weighted sum of
the source-side embeddings, where weights are induced by squared scalp
distance with bandwidth sigma.
"""

import numpy as np
import torch


def gaussian_kernel_weights(target_positions, source_positions, sigma):
    """
    Computes a (N_tgt, N_src) row-stochastic weight matrix using a Gaussian
    kernel over Euclidean scalp distance.

    Args:
        target_positions (np.ndarray): (N_tgt, 3) target scalp coords.
        source_positions (np.ndarray): (N_src, 3) source scalp coords.
        sigma (float): kernel bandwidth in the same units as positions
            (meters when positions come from MNE).

    Returns:
        np.ndarray of shape (N_tgt, N_src). Each row sums to 1.
    """
    diff = target_positions[:, None, :] - source_positions[None, :, :]
    sq_dist = np.sum(diff * diff, axis=-1)
    logits = -sq_dist / (2.0 * sigma * sigma)
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights


def interpolate_electrode_embedding(
    source_embedding,
    source_positions,
    target_positions,
    sigma=0.04,
):
    """
    Interpolates a per-electrode embedding tensor from a source layout to a
    target layout via Gaussian-weighted scalp-distance kernels.

    Args:
        source_embedding (Tensor): shape (1, N_src, D) or (N_src, D).
        source_positions (np.ndarray): (N_src, 3) scalp coords.
        target_positions (np.ndarray): (N_tgt, 3) scalp coords.
        sigma (float): kernel bandwidth in meters. Default 0.04 (~4 cm) is a
            reasonable scalp-smoothing scale for typical EEG montages.

    Returns:
        Tensor of shape (1, N_tgt, D) (or (N_tgt, D) if input had no batch
        dim), dtype matching the source.
    """
    squeeze_batch = False
    if source_embedding.dim() == 2:
        squeeze_batch = True
        source_embedding = source_embedding.unsqueeze(0)

    assert source_embedding.dim() == 3 and source_embedding.size(0) == 1, (
        f"Expected source_embedding shape (1, N_src, D), got {tuple(source_embedding.shape)}"
    )
    n_src = source_embedding.size(1)
    assert source_positions.shape == (n_src, 3), (
        f"source_positions {source_positions.shape} does not match source N={n_src}"
    )

    weights_np = gaussian_kernel_weights(
        target_positions=target_positions,
        source_positions=source_positions,
        sigma=sigma,
    )
    weights = torch.from_numpy(weights_np).to(
        dtype=source_embedding.dtype, device=source_embedding.device
    )

    interpolated = torch.einsum("ts,bsd->btd", weights, source_embedding)

    if squeeze_batch:
        interpolated = interpolated.squeeze(0)
    return interpolated
