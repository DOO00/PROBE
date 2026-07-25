from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from typing import Literal


def svd_purify_features(embeddings: torch.Tensor, keep_ratio: float = 0.8) -> torch.Tensor:
    """Purify node embeddings by removing low-energy singular directions.

    Args:
        embeddings: Node feature matrix with shape [N, D].
        keep_ratio: Ratio of largest singular values to keep. The smallest
            (1 - keep_ratio) proportion will be zeroed out.

    Returns:
        A purified feature matrix with the same shape [N, D].
    """
    if embeddings.dim() != 2:
        raise ValueError(f"`embeddings` must be a 2D tensor [N, D], got shape {tuple(embeddings.shape)}")
    if not (0.0 <= keep_ratio <= 1.0):
        raise ValueError(f"`keep_ratio` must be in [0, 1], got {keep_ratio}")
    if keep_ratio >= 1.0 - 1e-12:
        return embeddings
    use_large_graph_path = embeddings.size(0) > 200_000

    x = embeddings
    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    if use_large_graph_path:
        compute_dtype = torch.float32
    else:
        compute_dtype = torch.float64 if x.dtype in (torch.float32, torch.float64) else torch.float32
    x = x.to(compute_dtype)

    # Normalize matrix scale to improve numerical stability of SVD.
    scale = x.abs().amax()
    if torch.isfinite(scale) and scale > 1.0:
        x = x / scale
    else:
        scale = torch.tensor(1.0, dtype=x.dtype, device=x.device)

    if use_large_graph_path:
        total = x.size(1)
        keep_k = int(math.ceil(total * keep_ratio))
        keep_k = max(0, min(total, keep_k))
        purified = _purify_large_graph_with_covariance(x=x, keep_k=keep_k)
    else:
        # Robust SVD: retry with tiny jitter; fallback to low-rank SVD approximation.
        u, s, vh = _safe_svd(x)
        total = s.numel()
        keep_k = int(math.ceil(total * keep_ratio))
        keep_k = max(0, min(total, keep_k))

        if keep_k == 0:
            s_filtered = torch.zeros_like(s)
        elif keep_k == total:
            s_filtered = s
        else:
            s_filtered = torch.zeros_like(s)
            s_filtered[:keep_k] = s[:keep_k]

        # Reconstruct to original [N, D].
        purified = (u * s_filtered.unsqueeze(0)) @ vh
    purified = purified * scale
    purified = torch.nan_to_num(purified, nan=0.0, posinf=1e6, neginf=-1e6).to(dtype=embeddings.dtype)

    # Straight-through: keep purified features in forward while routing gradients through
    # the original embeddings to avoid unstable SVD backward on some server stacks.
    return embeddings + (purified - embeddings).detach()


def _purify_large_graph_with_covariance(x: torch.Tensor, keep_k: int) -> torch.Tensor:
    """Scalable top-k subspace projection for tall matrices [N, D], N >> D."""
    n, d = x.shape
    keep_k = int(max(0, min(d, keep_k)))
    if keep_k == 0:
        return torch.zeros_like(x)
    if keep_k == d:
        return x

    eye = None
    last_err: Exception | None = None
    for jitter in (0.0, 1e-8, 1e-6):
        try:
            cov = (x.transpose(0, 1) @ x) / max(1, n)  # [D, D]
            if jitter > 0.0:
                if eye is None:
                    eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
                cov = cov + jitter * eye
            evals, evecs = torch.linalg.eigh(cov)
            top_idx = torch.argsort(evals, descending=True)[:keep_k]
            v_top = evecs[:, top_idx]  # [D, keep_k]
            return (x @ v_top) @ v_top.transpose(0, 1)
        except RuntimeError as err:
            last_err = err
            continue

    warnings.warn(
        f"Large-graph covariance purification failed; using identity fallback. Last error: {last_err}",
        RuntimeWarning,
    )
    return x


def _safe_svd(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Numerically stable SVD with retries and a low-rank fallback."""
    last_err: Exception | None = None
    for jitter in (0.0, 1e-7, 1e-6, 1e-5):
        x_try = x if jitter == 0.0 else x + jitter * torch.randn_like(x)
        try:
            return torch.linalg.svd(x_try, full_matrices=False)
        except RuntimeError as err:
            last_err = err
            continue

    # Fallback: randomized low-rank SVD usually survives ill-conditioned cases.
    try:
        q = min(x.shape)
        q = max(1, q)
        u, s, v = torch.svd_lowrank(x, q=q, niter=4)
        vh = v.transpose(0, 1).contiguous()
        return u, s, vh
    except RuntimeError as err:
        last_err = err

    warnings.warn(
        f"SVD failed repeatedly; returning identity-like decomposition fallback. Last error: {last_err}",
        RuntimeWarning,
    )
    # Ultimate fallback: do not crash training; keep original features.
    m, n = x.shape
    k = min(m, n)
    u = torch.zeros((m, k), dtype=x.dtype, device=x.device)
    vh = torch.zeros((k, n), dtype=x.dtype, device=x.device)
    for i in range(k):
        u[i, i] = 1.0
        vh[i, i] = 1.0
    s = torch.ones((k,), dtype=x.dtype, device=x.device)
    return u, s, vh


def gaussian_feature_augment(
    purified_embeddings: torch.Tensor,
    noise_std: float = 0.1,
) -> torch.Tensor:
    """Generate Gaussian-perturbed variants with the same shape [N, D]."""
    if purified_embeddings.dim() != 2:
        raise ValueError(
            f"`purified_embeddings` must be 2D [N, D], got {tuple(purified_embeddings.shape)}"
        )
    if noise_std < 0:
        raise ValueError(f"`noise_std` must be >= 0, got {noise_std}")

    noise = torch.randn_like(purified_embeddings) * noise_std
    return purified_embeddings + noise


def build_augmented_embeddings(
    purified_embeddings: torch.Tensor,
    noise_std: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build [2N, D] embeddings by concatenating purified and augmented features."""
    augmented_embeddings = gaussian_feature_augment(purified_embeddings, noise_std=noise_std)
    concat_embeddings = torch.cat([purified_embeddings, augmented_embeddings], dim=0)
    return concat_embeddings, augmented_embeddings


def classify_concat_embeddings(
    purified_embeddings: torch.Tensor,
    classifier: nn.Module,
    noise_std: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create augmented nodes, concat to [2N, D], then send into linear classifier.

    Returns:
        logits_2n: [2N, C]
        concat_embeddings: [2N, D]
        augmented_embeddings: [N, D]
    """
    concat_embeddings, augmented_embeddings = build_augmented_embeddings(
        purified_embeddings, noise_std=noise_std
    )
    logits_2n = classifier(concat_embeddings)
    return logits_2n, concat_embeddings, augmented_embeddings


def compute_dual_loss(
    logits_2n: torch.Tensor,
    labels: torch.Tensor,
    num_supervised_nodes: int | None = None,
    supervised_node_ids: torch.Tensor | None = None,
    consistency_type: Literal["mse", "infonce"] = "mse",
    consistency_input: torch.Tensor | None = None,
    temperature: float = 0.2,
    bce_weight: float = 1.0,
    consistency_weight: float = 1.0,
    bce_pos_weight: float | None = None,
    bce_sample_weight: torch.Tensor | None = None,
    bce_loss_type: Literal["bce", "focal"] = "bce",
    focal_gamma: float = 2.0,
    rank_loss_weight: float = 0.0,
    rank_neg_ratio: float = 1.0,
    rank_hard_negative: bool = False,
    rank_hard_positive: bool = False,
    rank_neg_top_fraction: float = 1.0,
    rank_pos_bottom_fraction: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute dual loss: BCE on original nodes + consistency on paired nodes.

    Args:
        logits_2n: Classifier output for concatenated nodes, shape [2N, C] or [2N].
        labels: Binary labels of original nodes. Usually [N] / [N, 1], or at least
            first num_supervised_nodes labels.
        num_supervised_nodes: Only the first K original nodes are used for BCE.
            If None, use min(N, labels.shape[0]).
        supervised_node_ids: Optional node-id tensor for supervised BCE nodes.
            If provided, it takes precedence over num_supervised_nodes.
        consistency_type: "mse" or "infonce".
        consistency_input: Optional representation for consistency loss with shape
            [2N, D]. If None, uses sigmoid(logits) for MSE and logits for InfoNCE.
    """
    if logits_2n.dim() == 1:
        logits_2n = logits_2n.unsqueeze(-1)
    if logits_2n.dim() != 2:
        raise ValueError(f"`logits_2n` must be [2N, C] or [2N], got {tuple(logits_2n.shape)}")
    if logits_2n.size(0) % 2 != 0:
        raise ValueError(f"First dimension of `logits_2n` must be even (2N), got {logits_2n.size(0)}")

    n = logits_2n.size(0) // 2
    logits_ori = logits_2n[:n]
    logits_aug = logits_2n[n:]

    if labels.dim() == 1:
        labels = labels.unsqueeze(-1)
    if labels.dim() != 2:
        raise ValueError(f"`labels` must be [M] or [M, 1], got {tuple(labels.shape)}")

    if supervised_node_ids is not None:
        if supervised_node_ids.dim() != 1:
            raise ValueError(
                f"`supervised_node_ids` must be 1D, got shape {tuple(supervised_node_ids.shape)}"
            )
        if supervised_node_ids.numel() == 0:
            raise ValueError("`supervised_node_ids` cannot be empty.")

        supervised_node_ids = supervised_node_ids.to(device=logits_ori.device, dtype=torch.long)
        if supervised_node_ids.min().item() < 0 or supervised_node_ids.max().item() >= n:
            raise ValueError(
                f"`supervised_node_ids` must be within [0, {n - 1}], got range "
                f"[{supervised_node_ids.min().item()}, {supervised_node_ids.max().item()}]"
            )

        sup_logits = logits_ori[supervised_node_ids]
        if labels.size(0) == n:
            sup_labels = labels[supervised_node_ids]
        elif labels.size(0) == supervised_node_ids.numel():
            sup_labels = labels
        else:
            raise ValueError(
                "`labels` rows mismatch when using `supervised_node_ids`: expected either full-N labels "
                f"({n}) or exactly |supervised_node_ids| ({supervised_node_ids.numel()}), got {labels.size(0)}."
            )
        sup_labels = sup_labels.to(dtype=sup_logits.dtype, device=sup_logits.device)
    else:
        if num_supervised_nodes is None:
            num_supervised_nodes = min(n, labels.size(0))
        if not (0 <= num_supervised_nodes <= n):
            raise ValueError(f"`num_supervised_nodes` must be in [0, {n}], got {num_supervised_nodes}")
        if labels.size(0) < num_supervised_nodes:
            raise ValueError(
                f"`labels` has only {labels.size(0)} rows, but num_supervised_nodes={num_supervised_nodes}"
            )

        sup_logits = logits_ori[:num_supervised_nodes]
        sup_labels = labels[:num_supervised_nodes].to(dtype=sup_logits.dtype, device=sup_logits.device)

    if sup_labels.size(1) != sup_logits.size(1):
        if sup_labels.size(1) == 1 and sup_logits.size(1) == 1:
            pass
        else:
            raise ValueError(
                f"Shape mismatch for BCE: logits {tuple(sup_logits.shape)} vs labels {tuple(sup_labels.shape)}"
            )

    pos_weight_tensor = None
    if bce_pos_weight is not None:
        pos_weight_tensor = torch.tensor(
            [max(float(bce_pos_weight), 1e-8)],
            dtype=sup_logits.dtype,
            device=sup_logits.device,
        )

    bce_raw = F.binary_cross_entropy_with_logits(
        sup_logits,
        sup_labels,
        pos_weight=pos_weight_tensor,
        reduction="none",
    )
    if bce_loss_type == "focal":
        gamma = max(float(focal_gamma), 0.0)
        with torch.no_grad():
            prob = torch.sigmoid(sup_logits)
            p_t = prob * sup_labels + (1.0 - prob) * (1.0 - sup_labels)
            focal_factor = (1.0 - p_t).clamp_min(1e-8).pow(gamma)
        bce_raw = bce_raw * focal_factor
    elif bce_loss_type != "bce":
        raise ValueError(f"Unsupported bce_loss_type: {bce_loss_type}")
    if bce_sample_weight is not None:
        if bce_sample_weight.dim() == 1:
            bce_sample_weight = bce_sample_weight.unsqueeze(-1)
        if bce_sample_weight.shape != bce_raw.shape:
            raise ValueError(
                f"`bce_sample_weight` shape mismatch: expected {tuple(bce_raw.shape)}, "
                f"got {tuple(bce_sample_weight.shape)}"
            )
        sample_weight = bce_sample_weight.to(dtype=bce_raw.dtype, device=bce_raw.device).clamp_min(1e-8)
        bce_loss = (bce_raw * sample_weight).sum() / sample_weight.sum()
    else:
        bce_loss = bce_raw.mean()

    rank_loss = torch.zeros((), dtype=bce_loss.dtype, device=bce_loss.device)
    if rank_loss_weight > 0.0:
        logits_flat = sup_logits.view(-1)
        labels_flat = sup_labels.view(-1)
        pos_logits = logits_flat[labels_flat > 0.5]
        neg_logits = logits_flat[labels_flat <= 0.5]
        if pos_logits.numel() > 0 and neg_logits.numel() > 0:
            target_neg = int(max(1, round(pos_logits.numel() * max(rank_neg_ratio, 1e-6))))

            pos_pool = pos_logits
            neg_pool = neg_logits

            neg_top_frac = float(max(1e-6, min(1.0, rank_neg_top_fraction)))
            pos_bottom_frac = float(max(1e-6, min(1.0, rank_pos_bottom_fraction)))

            if rank_hard_negative and neg_logits.numel() > 1:
                neg_pool_n = int(max(1, round(neg_logits.numel() * neg_top_frac)))
                neg_pool_n = min(neg_pool_n, neg_logits.numel())
                # Hard negatives: highest-logit negatives.
                neg_pool = torch.topk(neg_logits, k=neg_pool_n, largest=True).values

            if rank_hard_positive and pos_logits.numel() > 1:
                pos_pool_n = int(max(1, round(pos_logits.numel() * pos_bottom_frac)))
                pos_pool_n = min(pos_pool_n, pos_logits.numel())
                # Hard positives: lowest-logit positives.
                pos_pool = torch.topk(pos_logits, k=pos_pool_n, largest=False).values

            pair_n = min(pos_pool.numel(), neg_pool.numel(), target_neg)
            if pair_n > 0:
                pos_perm = torch.randperm(pos_pool.numel(), device=pos_pool.device)[:pair_n]
                neg_perm = torch.randperm(neg_pool.numel(), device=neg_pool.device)[:pair_n]
                pos_sel = pos_pool[pos_perm]
                neg_sel = neg_pool[neg_perm]
                # Encourage positive nodes to have higher logits than negatives.
                rank_loss = F.softplus(-(pos_sel - neg_sel)).mean()

    if consistency_input is not None:
        if consistency_input.dim() == 1:
            consistency_input = consistency_input.unsqueeze(-1)
        if consistency_input.dim() != 2 or consistency_input.size(0) != 2 * n:
            raise ValueError(
                f"`consistency_input` must be [2N, D], got {tuple(consistency_input.shape)}"
            )
        rep_ori = consistency_input[:n]
        rep_aug = consistency_input[n:]
    else:
        if consistency_type == "mse":
            rep_ori = torch.sigmoid(logits_ori)
            rep_aug = torch.sigmoid(logits_aug)
        else:
            rep_ori = logits_ori
            rep_aug = logits_aug

    if consistency_type == "mse":
        consistency_loss = F.mse_loss(rep_ori, rep_aug)
    elif consistency_type == "infonce":
        if temperature <= 0:
            raise ValueError(f"`temperature` must be > 0 for InfoNCE, got {temperature}")
        rep_ori = F.normalize(rep_ori, p=2, dim=1)
        rep_aug = F.normalize(rep_aug, p=2, dim=1)
        sim = (rep_ori @ rep_aug.t()) / temperature  # [N, N], positive pairs on diagonal
        targets = torch.arange(n, device=sim.device)
        loss_i = F.cross_entropy(sim, targets)
        loss_j = F.cross_entropy(sim.t(), targets)
        consistency_loss = 0.5 * (loss_i + loss_j)
    else:
        raise ValueError(f"Unsupported consistency_type: {consistency_type}")

    total_loss = (
        bce_weight * bce_loss
        + consistency_weight * consistency_loss
        + float(rank_loss_weight) * rank_loss
    )
    loss_dict = {
        "loss_total": total_loss.detach(),
        "loss_bce": bce_loss.detach(),
        "loss_consistency": consistency_loss.detach(),
        "loss_rank": rank_loss.detach(),
    }
    return total_loss, loss_dict
