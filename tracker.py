from __future__ import annotations

from typing import Literal

import torch


class Tracker:
    """Track node-wise prediction fluctuation under subgraph-drop perturbations.

    This tracker accumulates prediction probabilities for each node and provides:
    1) variance-based sensitivity
    2) mean-bias-based sensitivity (w.r.t. a clean baseline prediction)
    """

    def __init__(
        self,
        num_nodes: int,
        device: torch.device | str | None = None,
        eps: float = 1e-12,
    ) -> None:
        if num_nodes <= 0:
            raise ValueError(f"`num_nodes` must be > 0, got {num_nodes}")
        self.num_nodes = int(num_nodes)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.eps = eps

        self.count = torch.zeros(self.num_nodes, dtype=torch.long, device=self.device)
        self.sum_probs = torch.zeros(self.num_nodes, dtype=torch.float32, device=self.device)
        self.sum_probs_sq = torch.zeros(self.num_nodes, dtype=torch.float32, device=self.device)

        # For mean-bias sensitivity.
        self.baseline_probs: torch.Tensor | None = None
        self.sum_abs_bias = torch.zeros(self.num_nodes, dtype=torch.float32, device=self.device)
        self.count_bias = torch.zeros(self.num_nodes, dtype=torch.long, device=self.device)

        # Optional metadata.
        self.drop_cluster_history: list[int | None] = []

    def reset(self) -> None:
        self.count.zero_()
        self.sum_probs.zero_()
        self.sum_probs_sq.zero_()
        self.sum_abs_bias.zero_()
        self.count_bias.zero_()
        self.drop_cluster_history.clear()

    @torch.no_grad()
    def set_baseline(
        self,
        preds: torch.Tensor,
        from_logits: bool = False,
        positive_class_index: int = 1,
        node_ids: torch.Tensor | None = None,
    ) -> None:
        """Set clean-graph baseline probabilities used by mean-bias sensitivity."""
        probs = self._to_prob_vector(
            preds=preds,
            from_logits=from_logits,
            positive_class_index=positive_class_index,
        )
        node_ids = self._resolve_node_ids(node_ids=node_ids, num_items=probs.numel(), device=probs.device)

        if self.baseline_probs is None:
            self.baseline_probs = torch.zeros(self.num_nodes, dtype=torch.float32, device=self.device)
        self.baseline_probs[node_ids] = probs.to(self.device)

    @torch.no_grad()
    def update(
        self,
        preds: torch.Tensor,
        dropped_cluster_id: int | None = None,
        from_logits: bool = False,
        positive_class_index: int = 1,
        node_ids: torch.Tensor | None = None,
    ) -> None:
        """Update tracker with one perturbation run's prediction probabilities.

        Args:
            preds: Prediction tensor. Supported shapes:
                - [N] / [M]: probability or logit of positive class
                - [N, 1] / [M, 1]
                - [N, C] / [M, C]
            dropped_cluster_id: Which target cluster was dropped in this run.
            from_logits: If True, convert logits to probabilities.
            positive_class_index: Positive class index when preds has C>1.
            node_ids: Optional node ids for partial updates. If None, assumes
                preds contains all nodes in [0..N-1] order.
        """
        probs = self._to_prob_vector(
            preds=preds,
            from_logits=from_logits,
            positive_class_index=positive_class_index,
        )
        node_ids = self._resolve_node_ids(node_ids=node_ids, num_items=probs.numel(), device=probs.device)

        probs = probs.to(self.device)
        node_ids = node_ids.to(self.device)

        self.count[node_ids] += 1
        self.sum_probs[node_ids] += probs
        self.sum_probs_sq[node_ids] += probs * probs

        if self.baseline_probs is not None:
            bias = torch.abs(probs - self.baseline_probs[node_ids])
            self.sum_abs_bias[node_ids] += bias
            self.count_bias[node_ids] += 1

        self.drop_cluster_history.append(None if dropped_cluster_id is None else int(dropped_cluster_id))

    @torch.no_grad()
    def get_variance_sensitivity(self) -> torch.Tensor:
        """Return node-wise prediction variance as 1D tensor [N]."""
        mean = torch.where(
            self.count > 0,
            self.sum_probs / self.count.clamp_min(1).float(),
            torch.zeros_like(self.sum_probs),
        )
        second_moment = torch.where(
            self.count > 0,
            self.sum_probs_sq / self.count.clamp_min(1).float(),
            torch.zeros_like(self.sum_probs_sq),
        )
        var = (second_moment - mean * mean).clamp_min(0.0)
        return var

    @torch.no_grad()
    def get_mean_bias_sensitivity(self) -> torch.Tensor:
        """Return node-wise mean absolute bias as 1D tensor [N].

        Baseline must be set via `set_baseline` first.
        """
        if self.baseline_probs is None:
            raise RuntimeError("Baseline is not set. Please call `set_baseline(...)` first.")
        mean_bias = torch.where(
            self.count_bias > 0,
            self.sum_abs_bias / self.count_bias.clamp_min(1).float(),
            torch.zeros_like(self.sum_abs_bias),
        )
        return mean_bias

    @torch.no_grad()
    def get_sensitivity(
        self,
        metric: Literal["variance", "mean_bias"] = "variance",
        normalize: bool = False,
    ) -> torch.Tensor:
        """Return a length-N 1D sensitivity tensor for downstream weighting."""
        if metric == "variance":
            sens = self.get_variance_sensitivity()
        elif metric == "mean_bias":
            sens = self.get_mean_bias_sensitivity()
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if normalize:
            sens = sens / sens.max().clamp_min(self.eps)
        return sens

    def summary(self) -> dict[str, float]:
        observed_ratio = float((self.count > 0).float().mean().item())
        mean_count = float(self.count.float().mean().item())
        return {"observed_ratio": observed_ratio, "mean_observations_per_node": mean_count}

    def _resolve_node_ids(
        self,
        node_ids: torch.Tensor | None,
        num_items: int,
        device: torch.device,
    ) -> torch.Tensor:
        if node_ids is None:
            if num_items != self.num_nodes:
                raise ValueError(
                    f"When `node_ids` is None, preds must contain all {self.num_nodes} nodes; got {num_items}."
                )
            return torch.arange(self.num_nodes, device=device, dtype=torch.long)

        if node_ids.dim() != 1:
            raise ValueError(f"`node_ids` must be 1D, got {tuple(node_ids.shape)}")
        if node_ids.numel() != num_items:
            raise ValueError(
                f"`node_ids` length ({node_ids.numel()}) must match number of preds ({num_items})."
            )
        if node_ids.min().item() < 0 or node_ids.max().item() >= self.num_nodes:
            raise ValueError("`node_ids` contains out-of-range node index.")
        return node_ids.long()

    def _to_prob_vector(
        self,
        preds: torch.Tensor,
        from_logits: bool,
        positive_class_index: int,
    ) -> torch.Tensor:
        if preds.dim() == 1:
            probs = preds.float()
            if from_logits:
                probs = torch.sigmoid(probs)
            return probs

        if preds.dim() != 2:
            raise ValueError(f"`preds` must be 1D or 2D, got shape {tuple(preds.shape)}")

        if preds.size(1) == 1:
            probs = preds[:, 0].float()
            if from_logits:
                probs = torch.sigmoid(probs)
            return probs

        # Multi-class [N, C]
        if positive_class_index < 0 or positive_class_index >= preds.size(1):
            raise ValueError(
                f"`positive_class_index` must be in [0, {preds.size(1)-1}], got {positive_class_index}"
            )
        if from_logits:
            probs_all = torch.softmax(preds.float(), dim=1)
        else:
            probs_all = preds.float()
        return probs_all[:, positive_class_index]

