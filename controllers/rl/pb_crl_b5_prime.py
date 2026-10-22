"""B5′ — PPO-Lagrangian on the shared price-and-budget packer (PB-CRL v3.1).

Not the original panel B5 (residual/pack-logit). Fair algorithmic comparator.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple  # noqa: F401 — Optional used for c2_K

import torch
from torch import nn
from torch.distributions import Normal


class B5PrimePolicy(nn.Module):
    """Maps a fixed-size state vector → (λ, β_prop) with Gaussian pre-squash heads."""

    is_pb_crl_b5_prime = True

    def __init__(
        self,
        state_dim: int = 16,
        hidden: int = 64,
        lam_max: float = 50.0,
        log_std_lam: float = -1.0,
        log_std_beta: float = -1.0,
    ):
        super().__init__()
        self.lam_max = float(lam_max)
        self.state_dim = int(state_dim)
        # Calibrated pathwise C2 budget intervention state
        self.c2_K: Optional[int] = None
        self.c2_Z: int = 0
        self.c2_forced_zero_epochs: int = 0
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu_lam = nn.Linear(hidden, 1)
        self.mu_beta = nn.Linear(hidden, 1)
        self.log_std_lam = nn.Parameter(torch.tensor(float(log_std_lam)))
        self.log_std_beta = nn.Parameter(torch.tensor(float(log_std_beta)))
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.cost_head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        # Start near (λ≈3.8, β≈0.98) aligning with empirical density distribution
        nn.init.zeros_(self.mu_lam.weight)
        nn.init.constant_(self.mu_lam.bias, -2.5)  # sigmoid(-2.5)*50.0 ≈ 3.79
        nn.init.zeros_(self.mu_beta.weight)
        nn.init.constant_(self.mu_beta.bias, 4.0)   # sigmoid(4)≈0.98

    def _fit_state(self, x: torch.Tensor) -> torch.Tensor:
        """Pad/truncate to state_dim (WiLab residual obs is often 8–10)."""
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        d = x.shape[-1]
        if d < self.state_dim:
            pad = torch.zeros(x.shape[0], self.state_dim - d, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        elif d > self.state_dim:
            x = x[..., : self.state_dim]
        return x

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self._fit_state(x))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(self._trunk(x)).squeeze(-1)

    def cost_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.cost_head(self._trunk(x)).squeeze(-1)

    def forward_actions(
        self, x: torch.Tensor, *, deterministic: bool = False,
    ):
        h = self._trunk(x)
        mu_l = self.mu_lam(h).squeeze(-1)
        mu_b = self.mu_beta(h).squeeze(-1)
        std_l = self.log_std_lam.exp().expand_as(mu_l)
        std_b = self.log_std_beta.exp().expand_as(mu_b)
        dist_l = Normal(mu_l, std_l)
        dist_b = Normal(mu_b, std_b)
        if deterministic:
            z_l, z_b = mu_l, mu_b
        else:
            z_l, z_b = dist_l.rsample(), dist_b.rsample()
        lam = torch.sigmoid(z_l) * self.lam_max
        beta = torch.sigmoid(z_b)
        logp = dist_l.log_prob(z_l) + dist_b.log_prob(z_b)
        ent = dist_l.entropy() + dist_b.entropy()
        return lam, beta, logp, z_l, z_b, ent

    def evaluate_actions(self, state_vec, z_l: float, z_b: float):
        h = self._trunk(torch.tensor(state_vec, dtype=torch.float32))
        mu_l = self.mu_lam(h).squeeze(-1)
        mu_b = self.mu_beta(h).squeeze(-1)
        dist_l = Normal(mu_l, self.log_std_lam.exp().expand_as(mu_l))
        dist_b = Normal(mu_b, self.log_std_beta.exp().expand_as(mu_b))
        zl = torch.tensor(float(z_l), dtype=torch.float32)
        zb = torch.tensor(float(z_b), dtype=torch.float32)
        logp = dist_l.log_prob(zl) + dist_b.log_prob(zb)
        ent = dist_l.entropy() + dist_b.entropy()
        return logp.reshape(()), ent.reshape(())

    def reset_c2(self, K: Optional[int] = None):
        self.c2_K = None if K is None else int(K)
        self.c2_Z = 0
        self.c2_forced_zero_epochs = 0

    def note_window(self, epoch: dict):
        """Call after each audit epoch; increments Z on valid violations."""
        if self.c2_K is None:
            return
        from controllers.rl.pb_crl_c2 import window_violated
        if window_violated(epoch):
            self.c2_Z += 1

    def act_numpy(self, state_vec, *, deterministic: bool = False):
        x = torch.tensor(state_vec, dtype=torch.float32)
        with torch.no_grad():
            lam, beta, logp, z_l, z_b, _ent = self.forward_actions(x, deterministic=deterministic)
        lam_f = float(lam.reshape(()).detach())
        beta_f = float(beta.reshape(()).detach())
        if self.c2_K is not None and self.c2_Z >= self.c2_K:
            beta_f = 0.0
            self.c2_forced_zero_epochs += 1
        return (
            lam_f, beta_f, logp.reshape(()),
            float(z_l.reshape(()).detach()), float(z_b.reshape(()).detach()),
        )


def state_vec_from_wilab(state: dict, layer, *, dim: int = 16) -> list:
    """Compact equal-info stub (budget block zeros until shield wired)."""
    cbr = float(state.get("cbr") or 0.0)
    resel = float(state.get("resel_t") or state.get("reselection_rate") or 0.0)
    epoch = float(getattr(layer, "current_epoch", 0) or 0)
    ce = float(getattr(layer, "controller_epochs", 100) or 100)
    vec = [cbr, resel, epoch / max(ce, 1.0), 0.0, 0.0, 0.0]  # last 3: kappa/K, Tleft/T, m_rem
    # pad / truncate
    if len(vec) < dim:
        vec = vec + [0.0] * (dim - len(vec))
    return vec[:dim]
