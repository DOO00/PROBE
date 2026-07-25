from __future__ import annotations

import argparse
import os
import uuid
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from backbone import build_backbone
from bmm import fit_two_component_beta_mixture
from dataloader import load_data
from graph_cluster import cluster_graph, drop_subgraph
from svd_utils import classify_concat_embeddings, compute_dual_loss, svd_purify_features
from tracker import Tracker
from utils import get_logger, get_training_config, metrics, set_seed


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch GAD training framework (Stage1/2/3)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--semi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use semi-supervised setting",
    )
    parser.add_argument("--num_exp", type=int, default=10, help="Repeat how many experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="yelp",
        choices=[
            "reddit",
            "weibo",
            "amazon",
            "yelp",
            "tfinance",
            "elliptic",
            "tolokers",
            "questions",
            "dgraphfin",
            "tsocial",
        ],
    )
    parser.add_argument("--model_name", type=str, default="sage", choices=["sage", "graphsage", "gcn"])
    parser.add_argument("--exp_setting", type=str, default="tran", choices=["tran", "ind"])
    parser.add_argument("--config_path", type=str, default="semi_train.conf.yaml")
    args = parser.parse_args()
    return args


def dgl_to_edge_index(g) -> torch.Tensor:
    src, dst = g.edges()
    return torch.stack([src.long(), dst.long()], dim=0)


def prepare_binary_labels(y: torch.Tensor) -> torch.Tensor:
    y = y.view(-1).float()
    if y.min().item() < 0:
        raise ValueError("Labels contain negative values, cannot form binary targets.")
    if y.max().item() > 1:
        y = (y > 0).float()
    else:
        y = y.clamp(0.0, 1.0)
    return y


def sample_labeled_nodes(idx_train: torch.Tensor, budget: int | None, seed: int) -> torch.Tensor:
    if idx_train.numel() == 0:
        raise ValueError("Empty training index.")
    if budget is None or budget <= 0 or budget >= idx_train.numel():
        return idx_train

    gen = torch.Generator(device=idx_train.device)
    gen.manual_seed(seed)
    perm = torch.randperm(idx_train.numel(), generator=gen, device=idx_train.device)
    return idx_train[perm[:budget]]


def build_cluster_ids(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_clusters: int,
    method: str,
    seed: int,
) -> torch.Tensor:
    try:
        return cluster_graph(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_clusters=num_clusters,
            method=method,  # type: ignore[arg-type]
            seed=seed,
        )
    except Exception:
        # Fallback if metis/scipy dependencies are unavailable.
        gen = torch.Generator(device=edge_index.device)
        gen.manual_seed(seed)
        perm = torch.randperm(num_nodes, generator=gen, device=edge_index.device)
        cluster_ids = torch.empty(num_nodes, dtype=torch.long, device=edge_index.device)
        cluster_ids[perm] = torch.arange(num_nodes, device=edge_index.device) % num_clusters
        return cluster_ids


def minmax_to_unit_interval(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    x_min = x.min()
    x_max = x.max()
    if (x_max - x_min).abs().item() < eps:
        return torch.full_like(x, 0.5).clamp(eps, 1.0 - eps)
    out = (x - x_min) / (x_max - x_min)
    return out.clamp(eps, 1.0 - eps)


def downsample_edge_index_stride(
    edge_index: torch.Tensor,
    keep_ratio: float,
    seed: int,
) -> torch.Tensor:
    """Lightweight edge downsampling by strided slicing to reduce train-time memory."""
    keep_ratio = float(keep_ratio)
    if keep_ratio >= 1.0 - 1e-12:
        return edge_index
    if keep_ratio <= 0.0:
        raise ValueError(f"`train_edge_keep_ratio` must be > 0, got {keep_ratio}")

    step = int(round(1.0 / keep_ratio))
    step = max(1, step)
    offset = int(seed % step)
    out = edge_index[:, offset::step]
    if out.size(1) == 0:
        return edge_index[:, :1]
    return out


def forward_purified_logits(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    keep_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, embeddings = model(x, edge_index)
    purified_embeddings = svd_purify_features(embeddings, keep_ratio=keep_ratio)
    logits = model.classifier(purified_embeddings)
    return logits, purified_embeddings


def tracker_to_soft_labels(tracker: Tracker, conf: dict[str, Any]) -> tuple[torch.Tensor, int]:
    sensitivity = tracker.get_sensitivity(metric=conf["tracker_metric"], normalize=False).detach().cpu()
    sensitivity_norm = minmax_to_unit_interval(sensitivity, eps=conf["bmm_eps"])

    bmm_model = fit_two_component_beta_mixture(
        x=sensitivity_norm,
        max_iters=int(conf["bmm_max_iters"]),
        tol=float(conf["bmm_tol"]),
        eps=float(conf["bmm_eps"]),
        min_shape=float(conf["bmm_min_shape"]),
        max_shape=float(conf["bmm_max_shape"]),
        verbose=conf.get("bmm_verbose", False),
    )
    comp_b = int(torch.argmax(bmm_model.get_component_means()).item())
    soft_labels = bmm_model.get_soft_labels(sensitivity_norm, component_index=comp_b).float()
    return soft_labels, comp_b


def evaluate_split(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y_true: torch.Tensor,
    idx: torch.Tensor,
    keep_ratio: float,
    use_amp: bool = False,
    eval_num_neighbors: list[int] | None = None,
    eval_batch_size: int = 65536,
    eval_data_cpu: Any | None = None,
    eval_loader: Any | None = None,
    return_probs: bool = False,
) -> dict[str, float] | tuple[dict[str, float], torch.Tensor]:
    model.eval()
    if eval_num_neighbors is not None and len(eval_num_neighbors) > 0:
        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader

        idx_cpu = idx.detach().cpu()
        y_cpu = y_true.detach().cpu()
        if eval_loader is None:
            if eval_data_cpu is None:
                data = Data(x=x.detach().cpu(), edge_index=edge_index.detach().cpu())
            else:
                data = eval_data_cpu
            loader = NeighborLoader(
                data=data,
                input_nodes=idx_cpu,
                num_neighbors=eval_num_neighbors,
                batch_size=int(eval_batch_size),
                shuffle=False,
                num_workers=0,
            )
        else:
            loader = eval_loader
        probs_buf = torch.zeros(idx_cpu.numel(), dtype=torch.float32)
        offset = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(x.device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits, _ = forward_purified_logits(
                        model, batch.x, batch.edge_index, keep_ratio=keep_ratio
                    )
                probs_seed = (
                    torch.sigmoid(logits.squeeze(-1)[: batch.batch_size]).detach().float().cpu()
                )
                if hasattr(batch, "input_id"):
                    probs_buf[batch.input_id.cpu()] = probs_seed
                else:
                    end = min(offset + probs_seed.numel(), probs_buf.numel())
                    probs_buf[offset:end] = probs_seed[: end - offset]
                    offset = end
        score = metrics(y_cpu[idx_cpu], probs_buf)
        if return_probs:
            return score, probs_buf
        return score

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, _ = forward_purified_logits(model, x, edge_index, keep_ratio=keep_ratio)
        probs = torch.sigmoid(logits.squeeze(-1))
        score = metrics(y_true[idx], probs[idx])
        if return_probs:
            return score, probs[idx].detach().float().cpu()
        return score


def compute_pos_weight(
    labels: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    max_pos_weight: float | None = None,
    eps: float = 1e-8,
) -> float:
    labels = labels.view(-1, 1).float()
    if sample_weight is None:
        sample_weight = torch.ones_like(labels)
    else:
        sample_weight = sample_weight.view(-1, 1).float().to(labels.device)

    pos_mass = (sample_weight * labels).sum()
    neg_mass = (sample_weight * (1.0 - labels)).sum()
    if pos_mass.item() <= eps:
        return 1.0

    pos_weight = float((neg_mass / pos_mass).item())
    pos_weight = max(1.0, pos_weight)
    if max_pos_weight is not None:
        pos_weight = min(pos_weight, float(max_pos_weight))
    return pos_weight


def build_pseudo_sample_weight(
    stage3_targets: torch.Tensor,
    labeled_idx: torch.Tensor,
    conf_low: float,
    conf_high: float,
    min_weight: float,
) -> torch.Tensor:
    targets = stage3_targets.view(-1, 1)
    confidence = (targets - 0.5).abs() * 2.0  # [0,1], larger means more confident.

    conf_low = float(max(0.0, min(1.0, conf_low)))
    conf_high = float(max(0.0, min(1.0, conf_high)))
    if conf_high <= conf_low:
        conf_high = min(1.0, conf_low + 1e-6)
    min_weight = float(max(0.0, min(1.0, min_weight)))

    scaled = ((confidence - conf_low) / (conf_high - conf_low)).clamp(0.0, 1.0)
    sample_weight = min_weight + (1.0 - min_weight) * scaled
    sample_weight[labeled_idx] = 1.0
    return sample_weight


def select_val_scalar(score: dict[str, float], metric: str) -> float:
    metric = metric.lower()
    if metric == "auroc":
        return score["AUROC"]
    if metric == "auprc":
        return score["AUPRC"]
    if metric == "reck":
        return score["RecK"]
    if metric == "composite":
        return 0.4 * score["AUROC"] + 0.4 * score["AUPRC"] + 0.2 * score["RecK"]
    raise ValueError(f"Unsupported val_select_metric: {metric}")


def recall_at_k_scalar(labels: torch.Tensor, probs: torch.Tensor) -> float:
    labels = labels.view(-1).long()
    probs = probs.view(-1).float()
    k = int(labels.sum().item())
    if k <= 0:
        return 0.0
    top_idx = torch.topk(probs, k=min(k, probs.numel()), largest=True).indices
    hit = labels[top_idx].sum().item()
    return float(hit / k * 100.0)


def run(
    conf: dict[str, Any],
    edge_index: torch.Tensor,
    x: torch.Tensor,
    y_true: torch.Tensor,
    idx_train: torch.Tensor,
    idx_val: torch.Tensor,
    idx_test: torch.Tensor,
) -> tuple[float, float, float, float]:
    """One complete run: Stage1 warm-up -> Stage2 pseudo labels -> Stage3 alternating training."""
    set_seed(conf["seed"])

    device = x.device
    num_nodes = x.size(0)
    y_target = y_true.view(-1, 1).float()

    labeled_idx = sample_labeled_nodes(idx_train, conf.get("labeled_budget", None), conf["seed"])

    disable_subgraph_drop = bool(conf.get("disable_subgraph_drop", False))
    drop_subgraph_on_cpu = bool(conf.get("drop_subgraph_on_cpu", False))
    tracker_drop_only = bool(conf.get("tracker_drop_only", False))
    train_edge_keep_ratio = float(conf.get("train_edge_keep_ratio", 1.0))
    base_train_edge_index = downsample_edge_index_stride(
        edge_index=edge_index,
        keep_ratio=train_edge_keep_ratio,
        seed=conf["seed"],
    )
    edge_index_for_drop = base_train_edge_index
    if disable_subgraph_drop:
        cluster_ids = None
    else:
        cluster_ids = build_cluster_ids(
            edge_index=edge_index_for_drop,
            num_nodes=num_nodes,
            num_clusters=conf["num_clusters"],
            method=conf["cluster_method"],
            seed=conf["seed"],
        )
        if drop_subgraph_on_cpu:
            edge_index_for_drop = edge_index_for_drop.detach().cpu()
            cluster_ids = cluster_ids.cpu()
        else:
            cluster_ids = cluster_ids.to(device)

    model = build_backbone(
        model_name=conf["model_name"],
        in_channels=x.size(1),
        hidden_channels=conf["hidden_channels"],
        num_classes=1,
        embed_channels=conf.get("embed_channels", conf["hidden_channels"]),
        dropout=conf["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=conf["wd"])

    tracker = Tracker(num_nodes=num_nodes, device=device)
    if conf.get("tracker_metric", "variance") == "mean_bias":
        with torch.no_grad():
            model.eval()
            clean_logits, _ = forward_purified_logits(model, x, edge_index, keep_ratio=conf["keep_ratio"])
            tracker.set_baseline(clean_logits.squeeze(-1), from_logits=True)

    best_epoch = 0
    best_val_scalar = -1.0
    best_val_score = {"AUROC": -1.0, "AUPRC": 0.0, "RecK": 0.0}
    stage = "stage1"
    stage3_targets = None
    stage3_sample_weight = None
    stage3_start_epoch = None
    use_amp = bool(conf.get("use_amp", False) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    train_num_neighbors = conf.get("train_num_neighbors", None)
    if train_num_neighbors is not None:
        train_num_neighbors = [int(v) for v in train_num_neighbors]
    train_batch_size = int(conf.get("train_batch_size", 32768))
    mini_batch_train = train_num_neighbors is not None and len(train_num_neighbors) > 0
    eval_num_neighbors = conf.get("eval_num_neighbors", None)
    if eval_num_neighbors is not None:
        eval_num_neighbors = [int(v) for v in eval_num_neighbors]
    eval_batch_size = int(conf.get("eval_batch_size", 65536))
    skip_tracker_update = bool(conf.get("skip_tracker_update", False))
    bce_loss_type = str(conf.get("bce_loss_type", "bce")).lower()
    focal_gamma = float(conf.get("focal_gamma", 2.0))
    rank_loss_weight = float(conf.get("rank_loss_weight", 0.0))
    rank_neg_ratio = float(conf.get("rank_neg_ratio", 1.0))
    rank_hard_negative = bool(conf.get("rank_hard_negative", False))
    rank_hard_positive = bool(conf.get("rank_hard_positive", False))
    rank_neg_top_fraction = float(conf.get("rank_neg_top_fraction", 1.0))
    rank_pos_bottom_fraction = float(conf.get("rank_pos_bottom_fraction", 1.0))
    eval_data_cpu = None
    val_eval_loader = None
    test_eval_loader = None
    if eval_num_neighbors is not None and len(eval_num_neighbors) > 0:
        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader

        # Build CPU graph once and reuse in each validation/test evaluation to
        # avoid repeated full-graph GPU->CPU copies on large datasets.
        eval_data_cpu = Data(x=x.detach().cpu(), edge_index=edge_index.detach().cpu())
        val_eval_loader = NeighborLoader(
            data=eval_data_cpu,
            input_nodes=idx_val.detach().cpu(),
            num_neighbors=eval_num_neighbors,
            batch_size=int(eval_batch_size),
            shuffle=False,
            num_workers=0,
        )
        test_eval_loader = NeighborLoader(
            data=eval_data_cpu,
            input_nodes=idx_test.detach().cpu(),
            num_neighbors=eval_num_neighbors,
            batch_size=int(eval_batch_size),
            shuffle=False,
            num_workers=0,
        )
    stage1_pos_weight = compute_pos_weight(
        labels=y_target[labeled_idx],
        max_pos_weight=conf.get("max_pos_weight", None),
    )
    train_loader = None
    if mini_batch_train:
        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader

        train_data_cpu = Data(
            x=x.detach().cpu(),
            edge_index=base_train_edge_index.detach().cpu(),
        )
        train_loader = NeighborLoader(
            data=train_data_cpu,
            input_nodes=idx_train.detach().cpu(),
            num_neighbors=train_num_neighbors,
            batch_size=int(train_batch_size),
            shuffle=True,
            num_workers=0,
        )

    os.makedirs("snapshots", exist_ok=True)
    snapshot_path = os.path.join(
        "snapshots",
        f"{conf['dataset']}_seed{conf['seed']}_{uuid.uuid4().hex}.pt",
    )
    torch.save(model.state_dict(), snapshot_path)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(conf["seed"] + 2026)

    with tqdm(
        total=conf["epochs"],
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
        ) as pbar:
        for epoch in range(1, 1 + conf["epochs"]):
            if stage == "stage1" and epoch > conf["warmup_epochs"] and int(tracker.count.sum().item()) > 0:
                soft_labels, _ = tracker_to_soft_labels(tracker, conf)
                stage3_targets = soft_labels.to(device).view(-1, 1).clamp(0.0, 1.0)
                stage3_targets[labeled_idx] = y_target[labeled_idx]
                stage3_sample_weight = build_pseudo_sample_weight(
                    stage3_targets=stage3_targets,
                    labeled_idx=labeled_idx,
                    conf_low=float(conf.get("pseudo_conf_low", 0.15)),
                    conf_high=float(conf.get("pseudo_conf_high", 0.85)),
                    min_weight=float(conf.get("pseudo_min_weight", 0.2)),
                )
                stage = "stage3"
                stage3_start_epoch = epoch
                if conf.get("tracker_reset_on_refresh", True):
                    tracker.reset()

            if (
                stage == "stage3"
                and conf["pseudo_update_interval"] > 0
                and epoch > conf["warmup_epochs"] + 1
                and ((epoch - (conf["warmup_epochs"] + 1)) % conf["pseudo_update_interval"] == 0)
                and int(tracker.count.sum().item()) > 0
            ):
                soft_labels, _ = tracker_to_soft_labels(tracker, conf)
                stage3_targets = soft_labels.to(device).view(-1, 1).clamp(0.0, 1.0)
                stage3_targets[labeled_idx] = y_target[labeled_idx]
                stage3_sample_weight = build_pseudo_sample_weight(
                    stage3_targets=stage3_targets,
                    labeled_idx=labeled_idx,
                    conf_low=float(conf.get("pseudo_conf_low", 0.15)),
                    conf_high=float(conf.get("pseudo_conf_high", 0.85)),
                    min_weight=float(conf.get("pseudo_min_weight", 0.2)),
                )
                if conf.get("tracker_reset_on_refresh", True):
                    tracker.reset()

            model.train()
            tracker_logits = None
            if mini_batch_train:
                target_cluster_id = -1
                dropped_edge_index = base_train_edge_index
                consistency_weight = float(conf["consistency_weight"])
                rampup_epochs = int(conf.get("consistency_rampup_epochs", 0))
                if stage == "stage3" and stage3_start_epoch is not None and rampup_epochs > 0:
                    progress = min(1.0, (epoch - stage3_start_epoch + 1) / max(1, rampup_epochs))
                    consistency_weight = consistency_weight * progress

                for batch in train_loader:  # type: ignore[union-attr]
                    batch = batch.to(device)
                    seed_n = int(batch.batch_size)
                    if seed_n <= 0:
                        continue
                    seed_ids = torch.arange(seed_n, device=device, dtype=torch.long)
                    seed_global = batch.n_id[:seed_n].to(device)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        _, embeddings = model(batch.x, batch.edge_index)
                        purified_embeddings = svd_purify_features(embeddings, keep_ratio=conf["keep_ratio"])
                        if consistency_weight > 0.0:
                            logits_2n, concat_embeddings, _ = classify_concat_embeddings(
                                purified_embeddings=purified_embeddings,
                                classifier=model.classifier,
                                noise_std=conf["noise_std"],
                            )
                        else:
                            logits = model.classifier(purified_embeddings)
                            logits_2n = torch.cat([logits, logits], dim=0)
                            concat_embeddings = None

                        if stage == "stage1" or stage3_targets is None:
                            batch_labels = y_target[seed_global]
                            loss, _ = compute_dual_loss(
                                logits_2n=logits_2n,
                                labels=batch_labels,
                                supervised_node_ids=seed_ids,
                                consistency_type=conf["consistency_type"],
                                consistency_input=concat_embeddings,
                                bce_weight=conf["bce_weight"],
                                consistency_weight=consistency_weight,
                                bce_pos_weight=stage1_pos_weight,
                                bce_loss_type=bce_loss_type,
                                focal_gamma=focal_gamma,
                                rank_loss_weight=rank_loss_weight,
                                rank_neg_ratio=rank_neg_ratio,
                                rank_hard_negative=rank_hard_negative,
                                rank_hard_positive=rank_hard_positive,
                                rank_neg_top_fraction=rank_neg_top_fraction,
                                rank_pos_bottom_fraction=rank_pos_bottom_fraction,
                            )
                        else:
                            batch_targets = stage3_targets[seed_global]
                            batch_sample_weight = (
                                stage3_sample_weight[seed_global]
                                if stage3_sample_weight is not None
                                else torch.ones_like(batch_targets)
                            )
                            stage3_pos_weight = compute_pos_weight(
                                labels=stage3_targets,
                                sample_weight=stage3_sample_weight,
                                max_pos_weight=conf.get("max_pos_weight", None),
                            )
                            loss, _ = compute_dual_loss(
                                logits_2n=logits_2n,
                                labels=batch_targets,
                                supervised_node_ids=seed_ids,
                                consistency_type=conf["consistency_type"],
                                consistency_input=concat_embeddings,
                                bce_weight=conf["bce_weight"],
                                consistency_weight=consistency_weight,
                                bce_pos_weight=stage3_pos_weight,
                                bce_sample_weight=batch_sample_weight,
                                bce_loss_type=bce_loss_type,
                                focal_gamma=focal_gamma,
                                rank_loss_weight=rank_loss_weight,
                                rank_neg_ratio=rank_neg_ratio,
                                rank_hard_negative=rank_hard_negative,
                                rank_hard_positive=rank_hard_positive,
                                rank_neg_top_fraction=rank_neg_top_fraction,
                                rank_pos_bottom_fraction=rank_pos_bottom_fraction,
                            )

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                if disable_subgraph_drop:
                    target_cluster_id = -1
                    dropped_edge_index = base_train_edge_index
                    train_edge_index = base_train_edge_index
                else:
                    target_cluster_id = int(torch.randint(0, conf["num_clusters"], (1,), generator=rng).item())
                    dropped_edge_index = drop_subgraph(
                        edge_index=edge_index_for_drop,
                        cluster_ids=cluster_ids,
                        target_cluster_id=target_cluster_id,
                    )
                    if drop_subgraph_on_cpu:
                        dropped_edge_index = dropped_edge_index.to(device, non_blocking=True)
                    train_edge_index = base_train_edge_index if tracker_drop_only else dropped_edge_index

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    _, embeddings = model(x, train_edge_index)
                    purified_embeddings = svd_purify_features(embeddings, keep_ratio=conf["keep_ratio"])
                    consistency_weight = float(conf["consistency_weight"])
                    rampup_epochs = int(conf.get("consistency_rampup_epochs", 0))
                    if stage == "stage3" and stage3_start_epoch is not None and rampup_epochs > 0:
                        progress = min(1.0, (epoch - stage3_start_epoch + 1) / max(1, rampup_epochs))
                        consistency_weight = consistency_weight * progress

                    if consistency_weight > 0.0:
                        logits_2n, concat_embeddings, _ = classify_concat_embeddings(
                            purified_embeddings=purified_embeddings,
                            classifier=model.classifier,
                            noise_std=conf["noise_std"],
                        )
                    else:
                        logits = model.classifier(purified_embeddings)
                        logits_2n = torch.cat([logits, logits], dim=0)
                        concat_embeddings = None

                    if stage == "stage1" or stage3_targets is None:
                        loss, _ = compute_dual_loss(
                            logits_2n=logits_2n,
                            labels=y_target,
                            supervised_node_ids=labeled_idx,
                            consistency_type=conf["consistency_type"],
                            consistency_input=concat_embeddings,
                            bce_weight=conf["bce_weight"],
                            consistency_weight=consistency_weight,
                            bce_pos_weight=stage1_pos_weight,
                            bce_loss_type=bce_loss_type,
                            focal_gamma=focal_gamma,
                            rank_loss_weight=rank_loss_weight,
                            rank_neg_ratio=rank_neg_ratio,
                            rank_hard_negative=rank_hard_negative,
                            rank_hard_positive=rank_hard_positive,
                            rank_neg_top_fraction=rank_neg_top_fraction,
                            rank_pos_bottom_fraction=rank_pos_bottom_fraction,
                        )
                    else:
                        if stage3_sample_weight is None:
                            stage3_sample_weight = torch.ones_like(stage3_targets)
                        stage3_pos_weight = compute_pos_weight(
                            labels=stage3_targets,
                            sample_weight=stage3_sample_weight,
                            max_pos_weight=conf.get("max_pos_weight", None),
                        )
                        loss, _ = compute_dual_loss(
                            logits_2n=logits_2n,
                            labels=stage3_targets,
                            num_supervised_nodes=num_nodes,
                            consistency_type=conf["consistency_type"],
                            consistency_input=concat_embeddings,
                            bce_weight=conf["bce_weight"],
                            consistency_weight=consistency_weight,
                            bce_pos_weight=stage3_pos_weight,
                            bce_sample_weight=stage3_sample_weight,
                            bce_loss_type=bce_loss_type,
                            focal_gamma=focal_gamma,
                            rank_loss_weight=rank_loss_weight,
                            rank_neg_ratio=rank_neg_ratio,
                            rank_hard_negative=rank_hard_negative,
                            rank_hard_positive=rank_hard_positive,
                            rank_neg_top_fraction=rank_neg_top_fraction,
                            rank_pos_bottom_fraction=rank_pos_bottom_fraction,
                        )

                if not skip_tracker_update and target_cluster_id >= 0 and not tracker_drop_only:
                    tracker_logits = logits_2n[:num_nodes].detach()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            with torch.no_grad():
                model.eval()
                if not mini_batch_train and not skip_tracker_update and target_cluster_id >= 0:
                    if tracker_drop_only:
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            dropped_logits, _ = forward_purified_logits(
                                model, x, dropped_edge_index, keep_ratio=conf["keep_ratio"]
                            )
                        tracker.update(
                            preds=dropped_logits.squeeze(-1),
                            dropped_cluster_id=target_cluster_id,
                            from_logits=True,
                        )
                    else:
                        tracker.update(
                            preds=tracker_logits.squeeze(-1),
                            dropped_cluster_id=target_cluster_id,
                            from_logits=True,
                        )

                val_score = evaluate_split(
                    model=model,
                    x=x,
                    edge_index=edge_index,
                    y_true=y_true,
                    idx=idx_val,
                    keep_ratio=conf["keep_ratio"],
                    use_amp=use_amp,
                    eval_num_neighbors=eval_num_neighbors,
                    eval_batch_size=eval_batch_size,
                    eval_data_cpu=eval_data_cpu,
                    eval_loader=val_eval_loader,
                )
                val_metric = conf.get("val_select_metric", "auroc")
                val_scalar = select_val_scalar(val_score, val_metric)
                if not np.isfinite(val_scalar):
                    val_scalar = -1.0
                if val_scalar > best_val_scalar:
                    best_epoch = epoch
                    best_val_scalar = val_scalar
                    best_val_score = val_score
                    torch.save(model.state_dict(), snapshot_path)
                else:
                    if epoch - best_epoch > conf["patience"]:
                        break

            pbar.set_postfix(
                {"Val|AUROC": best_val_score["AUROC"], "AUPRC": best_val_score["AUPRC"], "RecK": best_val_score["RecK"]}
            )
            pbar.update()

    model.load_state_dict(torch.load(snapshot_path, map_location=device))
    model.eval()
    final_eval_full_graph = bool(conf.get("final_eval_full_graph", False))
    final_eval_num_neighbors = conf.get("final_eval_num_neighbors", None)
    if final_eval_num_neighbors is not None:
        final_eval_num_neighbors = [int(v) for v in final_eval_num_neighbors]

    if final_eval_full_graph:
        final_eval_num_neighbors = None
        final_eval_loader = None
        final_val_loader = None
    else:
        if final_eval_num_neighbors is None:
            final_eval_num_neighbors = eval_num_neighbors
        if final_eval_num_neighbors == eval_num_neighbors:
            final_eval_loader = test_eval_loader
            final_val_loader = val_eval_loader
        elif final_eval_num_neighbors is not None and len(final_eval_num_neighbors) > 0:
            if eval_data_cpu is None:
                from torch_geometric.data import Data

                eval_data_cpu = Data(x=x.detach().cpu(), edge_index=edge_index.detach().cpu())
            from torch_geometric.loader import NeighborLoader

            final_eval_loader = NeighborLoader(
                data=eval_data_cpu,
                input_nodes=idx_test.detach().cpu(),
                num_neighbors=final_eval_num_neighbors,
                batch_size=int(eval_batch_size),
                shuffle=False,
                num_workers=0,
            )
            final_val_loader = NeighborLoader(
                data=eval_data_cpu,
                input_nodes=idx_val.detach().cpu(),
                num_neighbors=final_eval_num_neighbors,
                batch_size=int(eval_batch_size),
                shuffle=False,
                num_workers=0,
            )
        else:
            final_eval_loader = None
            final_val_loader = None
    with torch.no_grad():
        enable_score_blend = bool(conf.get("enable_score_blend", False))
        if enable_score_blend:
            val_score_raw, val_probs = evaluate_split(
                model=model,
                x=x,
                edge_index=edge_index,
                y_true=y_true,
                idx=idx_val,
                keep_ratio=conf["keep_ratio"],
                use_amp=use_amp,
                eval_num_neighbors=final_eval_num_neighbors,
                eval_batch_size=eval_batch_size,
                eval_data_cpu=eval_data_cpu,
                eval_loader=final_val_loader,
                return_probs=True,
            )
            _, test_probs = evaluate_split(
                model=model,
                x=x,
                edge_index=edge_index,
                y_true=y_true,
                idx=idx_test,
                keep_ratio=conf["keep_ratio"],
                use_amp=use_amp,
                eval_num_neighbors=final_eval_num_neighbors,
                eval_batch_size=eval_batch_size,
                eval_data_cpu=eval_data_cpu,
                eval_loader=final_eval_loader,
                return_probs=True,
            )

            labels_val_cpu = y_true[idx_val].detach().cpu().view(-1)
            labels_test_cpu = y_true[idx_test].detach().cpu().view(-1)

            blend_step = float(conf.get("score_blend_step", 0.0))
            use_degree_aux = bool(conf.get("blend_use_degree", True))
            use_feat_aux = bool(conf.get("blend_use_featnorm", False))
            val_metric_name = str(conf.get("val_select_metric", "auroc")).lower()

            if val_metric_name == "reck":
                best_val_scalar_local = recall_at_k_scalar(labels_val_cpu, val_probs)
            else:
                best_val_scalar_local = select_val_scalar(val_score_raw, val_metric_name)
            best_deg_w = 0.0
            best_feat_w = 0.0
            best_test_probs = test_probs

            degree_val = None
            degree_test = None
            if use_degree_aux:
                edge_cpu = (
                    eval_data_cpu.edge_index
                    if eval_data_cpu is not None and hasattr(eval_data_cpu, "edge_index")
                    else edge_index.detach().cpu()
                )
                degree_all = torch.bincount(edge_cpu[0], minlength=num_nodes).float()
                degree_all += torch.bincount(edge_cpu[1], minlength=num_nodes).float()
                degree_all = minmax_to_unit_interval(torch.log1p(degree_all))
                degree_val = degree_all[idx_val.detach().cpu()]
                degree_test = degree_all[idx_test.detach().cpu()]

            feat_val = None
            feat_test = None
            if use_feat_aux:
                x_cpu = (
                    eval_data_cpu.x
                    if eval_data_cpu is not None and hasattr(eval_data_cpu, "x")
                    else x.detach().cpu()
                )
                feat_all = minmax_to_unit_interval(torch.norm(x_cpu.float(), p=2, dim=1))
                feat_val = feat_all[idx_val.detach().cpu()]
                feat_test = feat_all[idx_test.detach().cpu()]

            if blend_step > 0.0 and (degree_val is not None or feat_val is not None):
                steps = int(max(1, round(1.0 / blend_step)))
                cand_weights = [i * blend_step for i in range(0, steps + 1)]
                for deg_w in cand_weights:
                    if degree_val is None and deg_w > 0:
                        continue
                    for feat_w in cand_weights:
                        if feat_val is None and feat_w > 0:
                            continue
                        if deg_w + feat_w > 1.0 + 1e-9:
                            continue
                        base_w = 1.0 - deg_w - feat_w
                        if base_w < -1e-9:
                            continue
                        blended_val = base_w * val_probs
                        if degree_val is not None and deg_w > 0:
                            blended_val = blended_val + deg_w * degree_val
                        if feat_val is not None and feat_w > 0:
                            blended_val = blended_val + feat_w * feat_val

                        if val_metric_name == "reck":
                            val_scalar_local = recall_at_k_scalar(labels_val_cpu, blended_val)
                        else:
                            val_score_local = metrics(labels_val_cpu, blended_val)
                            val_scalar_local = select_val_scalar(val_score_local, val_metric_name)

                        if val_scalar_local > best_val_scalar_local:
                            best_val_scalar_local = val_scalar_local
                            best_deg_w = float(deg_w)
                            best_feat_w = float(feat_w)
                            blended_test = base_w * test_probs
                            if degree_test is not None and deg_w > 0:
                                blended_test = blended_test + deg_w * degree_test
                            if feat_test is not None and feat_w > 0:
                                blended_test = blended_test + feat_w * feat_test
                            best_test_probs = blended_test

                print(
                    "Blend| "
                    f"degree_w={best_deg_w:.2f}, feat_w={best_feat_w:.2f}, "
                    f"val_scalar={best_val_scalar_local:.2f}"
                )

            test_score = metrics(labels_test_cpu, best_test_probs)
        else:
            test_score = evaluate_split(
                model=model,
                x=x,
                edge_index=edge_index,
                y_true=y_true,
                idx=idx_test,
                keep_ratio=conf["keep_ratio"],
                use_amp=use_amp,
                eval_num_neighbors=final_eval_num_neighbors,
                eval_batch_size=eval_batch_size,
                eval_data_cpu=eval_data_cpu,
                eval_loader=final_eval_loader,
            )
        auroc, auprc, reck = test_score["AUROC"], test_score["AUPRC"], test_score["RecK"]
        print(f"Test| AUROC={auroc:.2f}, AUPRC={auprc:.2f}, RecK={reck:.2f}")
    if os.path.exists(snapshot_path):
        os.remove(snapshot_path)
    return auroc, auprc, reck, best_val_scalar


def main():
    args = get_args()

    if args.semi:
        conf_file = args.config_path
        log_dir = "./logs/semi"
    else:
        if args.config_path != "semi_train.conf.yaml":
            conf_file = args.config_path
        else:
            conf_file = "full_train.conf.yaml" if os.path.exists("full_train.conf.yaml") else args.config_path
        log_dir = "./logs/full"

    conf_from_file = get_training_config(args.dataset, config_path=conf_file)
    conf = dict(args.__dict__, **conf_from_file)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(f"{log_dir}/{args.dataset}.log")
    logger.info(str(conf))

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the current environment. "
            "Please run with a GPU-enabled PyTorch environment."
        )
    device = torch.device(f"cuda:{conf['device']}")

    g, x, y, train_masks, val_masks, test_masks = load_data(
        args.dataset, semi=args.semi, feat_trans=conf["feat_trans"]
    )
    edge_index = dgl_to_edge_index(g).to(device)
    x = x.to(device).float()
    y = prepare_binary_labels(y).to(device)

    train_masks = train_masks.bool().to(device)
    val_masks = val_masks.bool().to(device)
    test_masks = test_masks.bool().to(device)
    indices = torch.arange(x.shape[0], device=device)

    aurocs, auprcs, recks = [], [], []
    num_restarts = int(conf.get("num_restarts", 1))
    num_restarts = max(1, num_restarts)
    for i in range(min(conf["num_exp"], train_masks.shape[1])):
        idx_train = indices[train_masks[:, i]]
        idx_val = indices[val_masks[:, i]]
        idx_test = indices[test_masks[:, i]]
        base_seed = int(i if not args.semi else conf["seed"])

        best_restart = None
        best_restart_val = -1.0
        for restart_id in range(num_restarts):
            conf_run = dict(conf)
            conf_run["seed"] = base_seed + restart_id * 9973
            auroc, auprc, reck, val_scalar = run(
                conf_run,
                edge_index,
                x,
                y,
                idx_train,
                idx_val,
                idx_test,
            )
            if val_scalar > best_restart_val:
                best_restart_val = val_scalar
                best_restart = (auroc, auprc, reck)

        if best_restart is None:
            raise RuntimeError("No valid restart result found.")
        aurocs.append(best_restart[0])
        auprcs.append(best_restart[1])
        recks.append(best_restart[2])

    res = (
        f"Test| AUROC={np.mean(aurocs):.2f}+-{np.std(aurocs):.2f}, "
        f"AUPRC={np.mean(auprcs):.2f}+-{np.std(auprcs):.2f}, "
        f"RecK={np.mean(recks):.2f}+-{np.std(recks):.2f}\n"
    )
    logger.info(res)


if __name__ == "__main__":
    main()
