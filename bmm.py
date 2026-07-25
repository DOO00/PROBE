from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

ArrayLike1D = Union[np.ndarray, torch.Tensor]


@dataclass
class BetaMixtureParams:
    pi: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor


class TwoComponentBetaMixtureEM:
    """1D two-component Beta Mixture Model fitted by EM.

    Input x must be a 1D normalized sensitivity vector in (0, 1).
    """

    def __init__(
        self,
        max_iters: int = 200,
        tol: float = 1e-6,
        eps: float = 1e-6,
        min_shape: float = 1e-2,
        max_shape: float = 1e4,
        verbose: bool = False,
    ) -> None:
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.eps = float(eps)
        self.min_shape = float(min_shape)
        self.max_shape = float(max_shape)
        self.verbose = bool(verbose)

        self.params: BetaMixtureParams | None = None
        self.fitted_: bool = False
        self.n_iters_: int = 0
        self.log_likelihood_: float = float("-inf")
        self.log_likelihood_trace_: list[float] = []
        self.x_fit_: torch.Tensor | None = None

    def fit(self, x: ArrayLike1D) -> "TwoComponentBetaMixtureEM":
        x_t = self._prepare_input(x)
        n = x_t.numel()
        if n < 2:
            raise ValueError("Need at least 2 samples to fit a two-component mixture.")
        self.x_fit_ = x_t.clone()

        params = self._init_params(x_t)
        prev_ll = None
        self.log_likelihood_trace_ = []

        for i in range(self.max_iters):
            # E-step: compute responsibilities r[n, k]
            log_prob = self._log_component_pdf(x_t, params.alpha, params.beta)
            log_weighted = log_prob + torch.log(params.pi.clamp_min(self.eps)).unsqueeze(0)
            log_norm = torch.logsumexp(log_weighted, dim=1, keepdim=True)
            r = torch.exp(log_weighted - log_norm)

            # M-step
            nk = r.sum(dim=0).clamp_min(self.eps)  # [2]
            pi = (nk / n).clamp_min(self.eps)
            pi = pi / pi.sum()

            alpha = torch.empty(2, dtype=x_t.dtype, device=x_t.device)
            beta = torch.empty(2, dtype=x_t.dtype, device=x_t.device)
            for k in range(2):
                a_k, b_k = self._weighted_beta_moments(x_t, r[:, k])
                alpha[k] = a_k
                beta[k] = b_k

            params = BetaMixtureParams(pi=pi, alpha=alpha, beta=beta)

            ll = float(log_norm.sum().item())
            self.log_likelihood_trace_.append(ll)
            if self.verbose:
                print(
                    f"[EM] iter={i + 1:03d}, ll={ll:.6f}, "
                    f"pi={params.pi.tolist()}, alpha={params.alpha.tolist()}, beta={params.beta.tolist()}"
                )

            if prev_ll is not None and abs(ll - prev_ll) < self.tol:
                self.n_iters_ = i + 1
                self.log_likelihood_ = ll
                self.params = params
                self.fitted_ = True
                return self
            prev_ll = ll

        self.n_iters_ = self.max_iters
        self.log_likelihood_ = self.log_likelihood_trace_[-1] if self.log_likelihood_trace_ else float("-inf")
        self.params = params
        self.fitted_ = True
        return self

    def predict_proba(self, x: ArrayLike1D) -> torch.Tensor:
        """Posterior responsibilities p(z=k|x), shape [N, 2]."""
        self._check_fitted()
        x_t = self._prepare_input(x)
        assert self.params is not None

        log_prob = self._log_component_pdf(x_t, self.params.alpha, self.params.beta)
        log_weighted = log_prob + torch.log(self.params.pi.clamp_min(self.eps)).unsqueeze(0)
        log_norm = torch.logsumexp(log_weighted, dim=1, keepdim=True)
        resp = torch.exp(log_weighted - log_norm)
        return resp

    def predict(self, x: ArrayLike1D) -> torch.Tensor:
        """Hard assignments in {0, 1}, shape [N]."""
        return self.predict_proba(x).argmax(dim=1)

    def get_component_means(self) -> torch.Tensor:
        """Return Beta means of the two components, shape [2]."""
        self._check_fitted()
        assert self.params is not None
        means = self.params.alpha / (self.params.alpha + self.params.beta).clamp_min(self.eps)
        return means

    def get_soft_labels(
        self,
        x: ArrayLike1D | None = None,
        component_index: int | None = None,
    ) -> torch.Tensor:
        """Return posterior probability of belonging to high-volatility component.

        Args:
            x: 1D input array/tensor in [0, 1]. If None, use data seen in `fit`.
            component_index: Target component id (0 or 1). If None, automatically
                picks the component with larger Beta mean as high-volatility
                Component B.

        Returns:
            soft_labels: 1D tensor [N], posterior p(z=Component B | x).
        """
        self._check_fitted()
        if x is None:
            if self.x_fit_ is None:
                raise RuntimeError("No cached training input found. Pass `x` explicitly.")
            x_in = self.x_fit_
        else:
            x_in = x

        resp = self.predict_proba(x_in)  # [N, 2]
        if component_index is None:
            means = self.get_component_means()
            component_index = int(torch.argmax(means).item())

        if component_index not in (0, 1):
            raise ValueError(f"`component_index` must be 0 or 1, got {component_index}")
        return resp[:, component_index]

    def get_params(self) -> dict[str, np.ndarray]:
        self._check_fitted()
        assert self.params is not None
        return {
            "pi": self.params.pi.detach().cpu().numpy(),
            "alpha": self.params.alpha.detach().cpu().numpy(),
            "beta": self.params.beta.detach().cpu().numpy(),
        }

    def _check_fitted(self) -> None:
        if not self.fitted_ or self.params is None:
            raise RuntimeError("Model is not fitted. Call `fit(x)` first.")

    def _prepare_input(self, x: ArrayLike1D) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            x_t = x
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")

        if x_t.dim() != 1:
            raise ValueError(f"Input must be 1D, got shape {tuple(x_t.shape)}")

        x_t = x_t.detach().to(dtype=torch.float64, device="cpu")
        if torch.any(torch.isnan(x_t)) or torch.any(torch.isinf(x_t)):
            raise ValueError("Input contains NaN or Inf.")

        # Ensure values are in (0, 1) with clipping for numerical stability.
        if torch.any(x_t < 0.0) or torch.any(x_t > 1.0):
            raise ValueError("Input must be normalized to [0, 1].")
        x_t = x_t.clamp(min=self.eps, max=1.0 - self.eps)
        return x_t

    def _init_params(self, x: torch.Tensor) -> BetaMixtureParams:
        median = torch.median(x)
        idx0 = x <= median
        idx1 = ~idx0

        if idx0.sum() == 0 or idx1.sum() == 0:
            perm = torch.randperm(x.numel())
            half = x.numel() // 2
            idx0 = torch.zeros_like(x, dtype=torch.bool)
            idx0[perm[:half]] = True
            idx1 = ~idx0

        x0 = x[idx0]
        x1 = x[idx1]

        a0, b0 = self._moments_to_beta(x0.mean(), x0.var(unbiased=False))
        a1, b1 = self._moments_to_beta(x1.mean(), x1.var(unbiased=False))

        pi0 = idx0.float().mean().clamp_min(self.eps).to(dtype=x.dtype)
        pi1 = (1.0 - pi0).to(dtype=x.dtype)
        pi = torch.stack([pi0, pi1], dim=0)
        pi = (pi / pi.sum()).clamp_min(self.eps)
        pi = pi / pi.sum()

        alpha = torch.stack([a0, a1], dim=0).to(dtype=x.dtype)
        beta = torch.stack([b0, b1], dim=0).to(dtype=x.dtype)
        return BetaMixtureParams(pi=pi, alpha=alpha, beta=beta)

    def _weighted_beta_moments(self, x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = w.clamp_min(self.eps)
        w_sum = w.sum().clamp_min(self.eps)
        mu = (w * x).sum() / w_sum
        var = (w * (x - mu) ** 2).sum() / w_sum
        return self._moments_to_beta(mu, var)

    def _moments_to_beta(self, mean: torch.Tensor, var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = mean.clamp(self.eps, 1.0 - self.eps)

        # Valid Beta variance is strictly less than mean * (1-mean).
        max_var = (mean * (1.0 - mean)).clamp_min(self.eps) * (1.0 - self.eps)
        var = var.clamp(min=self.eps, max=max_var)

        common = mean * (1.0 - mean) / var - 1.0
        common = common.clamp(min=self.min_shape, max=self.max_shape)

        alpha = (mean * common).clamp(min=self.min_shape, max=self.max_shape)
        beta = ((1.0 - mean) * common).clamp(min=self.min_shape, max=self.max_shape)
        return alpha, beta

    def _log_component_pdf(self, x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Return log p(x|k) for each k, shape [N, 2]."""
        x = x.unsqueeze(1)  # [N, 1]
        a = alpha.unsqueeze(0).clamp_min(self.min_shape)
        b = beta.unsqueeze(0).clamp_min(self.min_shape)

        log_norm = torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(b)
        log_pdf = (a - 1.0) * torch.log(x) + (b - 1.0) * torch.log(1.0 - x) + log_norm
        return log_pdf


def fit_two_component_beta_mixture(
    x: ArrayLike1D,
    max_iters: int = 200,
    tol: float = 1e-6,
    eps: float = 1e-6,
    min_shape: float = 1e-2,
    max_shape: float = 1e4,
    verbose: bool = False,
) -> TwoComponentBetaMixtureEM:
    """Functional API for fitting a 1D two-component Beta Mixture Model."""
    model = TwoComponentBetaMixtureEM(
        max_iters=max_iters,
        tol=tol,
        eps=eps,
        min_shape=min_shape,
        max_shape=max_shape,
        verbose=verbose,
    )
    model.fit(x)
    return model
