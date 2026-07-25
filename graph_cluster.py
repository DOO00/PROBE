from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F


def cluster_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_clusters: int,
    method: Literal["auto", "metis", "spectral", "random"] = "auto",
    seed: int = 42,
    kmeans_max_iter: int = 100,
) -> torch.Tensor:
    """Partition a PyG graph into K non-overlapping subgraphs.

    Args:
        edge_index: Graph edges in PyG format [2, E].
        num_nodes: Number of nodes.
        num_clusters: K, number of subgraphs.
        method: "metis", "spectral", or "auto" (metis first, then spectral).
        seed: Random seed for deterministic clustering.
        kmeans_max_iter: Iterations for spectral clustering K-Means stage.

    Returns:
        cluster_ids: LongTensor with shape [num_nodes], values in [0, K-1].
    """
    _validate_inputs(edge_index=edge_index, num_nodes=num_nodes, num_clusters=num_clusters)

    if num_clusters == 1:
        return torch.zeros(num_nodes, dtype=torch.long)
    if num_clusters == num_nodes:
        return torch.arange(num_nodes, dtype=torch.long)

    if method not in {"auto", "metis", "spectral", "random"}:
        raise ValueError(f"Unsupported method: {method}")

    if method == "random":
        g = torch.Generator(device=edge_index.device)
        g.manual_seed(seed)
        perm = torch.randperm(num_nodes, generator=g, device=edge_index.device)
        cluster_ids = torch.empty(num_nodes, dtype=torch.long, device=edge_index.device)
        cluster_ids[perm] = torch.arange(num_nodes, device=edge_index.device) % num_clusters
        return cluster_ids.cpu()

    if method in {"auto", "metis"}:
        try:
            return _cluster_with_metis(edge_index=edge_index, num_nodes=num_nodes, num_clusters=num_clusters)
        except Exception:
            if method == "metis":
                raise

    return _cluster_with_spectral(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_clusters=num_clusters,
        seed=seed,
        kmeans_max_iter=kmeans_max_iter,
    )


def drop_subgraph(
    edge_index: torch.Tensor,
    cluster_ids: torch.Tensor,
    target_cluster_id: int,
) -> torch.Tensor:
    """Drop a target subgraph from PyG edge_index.

    Remove:
    1) edges inside the target cluster,
    2) cross-subgraph edges between target cluster and non-target nodes.

    This is equivalent to removing all edges incident to nodes whose
    cluster id equals target_cluster_id.

    Args:
        edge_index: [2, E] graph connectivity.
        cluster_ids: [N] cluster assignment for each node.
        target_cluster_id: The cluster id to drop.

    Returns:
        new_edge_index: [2, E'] after subgraph dropout.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"`edge_index` must be [2, E], got {tuple(edge_index.shape)}")
    if cluster_ids.dim() != 1:
        raise ValueError(f"`cluster_ids` must be [N], got {tuple(cluster_ids.shape)}")
    if cluster_ids.numel() == 0:
        raise ValueError("`cluster_ids` cannot be empty")

    if not torch.is_tensor(target_cluster_id):
        target_cluster_id = int(target_cluster_id)

    # Make sure indexing happens on the same device.
    if cluster_ids.device != edge_index.device:
        cluster_ids = cluster_ids.to(edge_index.device)

    src, dst = edge_index[0], edge_index[1]
    if src.numel() == 0:
        return edge_index
    if src.max().item() >= cluster_ids.numel() or dst.max().item() >= cluster_ids.numel():
        raise ValueError("`edge_index` contains node ids out of range of `cluster_ids`")

    src_in_target = cluster_ids[src] == target_cluster_id
    dst_in_target = cluster_ids[dst] == target_cluster_id

    # Keep only edges where both endpoints are outside the target cluster.
    keep_mask = (~src_in_target) & (~dst_in_target)
    new_edge_index = edge_index[:, keep_mask]
    return new_edge_index


def _validate_inputs(edge_index: torch.Tensor, num_nodes: int, num_clusters: int) -> None:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"`edge_index` must be [2, E], got {tuple(edge_index.shape)}")
    if num_nodes <= 0:
        raise ValueError(f"`num_nodes` must be > 0, got {num_nodes}")
    if num_clusters <= 0:
        raise ValueError(f"`num_clusters` must be > 0, got {num_clusters}")
    if num_clusters > num_nodes:
        raise ValueError(f"`num_clusters` ({num_clusters}) cannot exceed num_nodes ({num_nodes})")


def _edge_index_to_undirected_pairs(edge_index: torch.Tensor) -> np.ndarray:
    edge_np = edge_index.detach().cpu().numpy()
    row = edge_np[0]
    col = edge_np[1]

    # Remove self-loops, then merge (u, v) and (v, u) as one undirected edge.
    mask = row != col
    row = row[mask]
    col = col[mask]

    if row.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    u = np.minimum(row, col)
    v = np.maximum(row, col)
    pairs = np.stack([u, v], axis=1)
    pairs = np.unique(pairs, axis=0).astype(np.int64, copy=False)
    return pairs


def _cluster_with_metis(edge_index: torch.Tensor, num_nodes: int, num_clusters: int) -> torch.Tensor:
    import pymetis  # type: ignore

    pairs = _edge_index_to_undirected_pairs(edge_index)
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in pairs:
        adjacency[int(u)].append(int(v))
        adjacency[int(v)].append(int(u))

    # Deduplicate adjacency list.
    adjacency = [list(dict.fromkeys(neigh)) for neigh in adjacency]
    _, membership = pymetis.part_graph(num_clusters, adjacency=adjacency)
    return torch.tensor(membership, dtype=torch.long)


def _cluster_with_spectral(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_clusters: int,
    seed: int,
    kmeans_max_iter: int,
) -> torch.Tensor:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh

    pairs = _edge_index_to_undirected_pairs(edge_index)

    if pairs.shape[0] == 0:
        # No edges: fall back to simple contiguous assignment.
        cluster_ids = torch.arange(num_nodes, dtype=torch.long) % num_clusters
        return cluster_ids

    row = np.concatenate([pairs[:, 0], pairs[:, 1]])
    col = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.ones(row.shape[0], dtype=np.float64)
    adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    adj.sum_duplicates()

    # Normalized graph Laplacian.
    lap = laplacian(adj, normed=True)
    evals, evecs = eigsh(lap, k=num_clusters, which="SM")
    order = np.argsort(evals)
    evecs = evecs[:, order]

    spectral_feat = torch.from_numpy(evecs).float()
    spectral_feat = F.normalize(spectral_feat, p=2, dim=1, eps=1e-12)
    cluster_ids = _kmeans_torch(
        x=spectral_feat,
        k=num_clusters,
        seed=seed,
        max_iter=kmeans_max_iter,
    )
    return cluster_ids.cpu()


def _kmeans_torch(
    x: torch.Tensor,
    k: int,
    seed: int = 42,
    max_iter: int = 100,
) -> torch.Tensor:
    n = x.size(0)
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)

    perm = torch.randperm(n, generator=g, device=x.device)
    centers = x[perm[:k]].clone()
    labels = torch.zeros(n, dtype=torch.long, device=x.device)

    for i in range(max_iter):
        dist = torch.cdist(x, centers, p=2)  # [N, K]
        new_labels = dist.argmin(dim=1)
        if i > 0 and torch.equal(new_labels, labels):
            break
        labels = new_labels

        new_centers = centers.clone()
        for cid in range(k):
            mask = labels == cid
            if mask.any():
                new_centers[cid] = x[mask].mean(dim=0)
            else:
                random_idx = torch.randint(0, n, (1,), generator=g, device=x.device)
                new_centers[cid] = x[random_idx]
        centers = new_centers

    return labels
