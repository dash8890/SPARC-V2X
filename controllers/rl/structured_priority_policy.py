"""Structured Priority & Opportunity-Cost Controller (SPARC-V2X Redesign).

Implements the formal mathematical specification for candidate-dependent priority control:
  1. DeepSets Task Encoder (MLP 64 -> mean+max pool, dim 128)
  2. Helper Fleet Backlog & Geometry Encoder (MLP 32 -> mean pool, dim 32)
  3. Causal Radio History GRU (64 units over last 10 epochs of [CBR, resel, bytes/Bmax, z])
  4. Predictive Risk Budget Block (MLP 16 on [kappa/K, T_left/T, m_rem/nu(K)])
  5. Fused Shared Trunk (dim 240 -> 128 -> 128)
  6. Action Heads:
     - Priority weights w(t) = [w_v, w_q, w_o, w_r] in R_+^4
     - Volume throttle beta_t in [0, 1]
  7. Deterministic Bipartite Priority Matching Algorithm:
     S(k, j) = w_v * V_kj - w_q * L_kj - w_o * O_kj - w_r * R_kj
  8. Twin critics: V_r (utility value) and V_c (constraint cost value)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class DeepSetsTaskEncoder(nn.Module):
    def __init__(self, in_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, task_feats: torch.Tensor) -> torch.Tensor:
        # task_feats: (batch, n_tasks, in_dim) or (n_tasks, in_dim)
        if task_feats.dim() == 2:
            task_feats = task_feats.unsqueeze(0)
        if task_feats.shape[1] == 0:
            return torch.zeros(task_feats.shape[0], 2 * 64, dtype=task_feats.dtype, device=task_feats.device)
        h = self.phi(task_feats)  # (batch, n_tasks, hidden_dim)
        h_mean = h.mean(dim=1)
        h_max, _ = h.max(dim=1)
        return torch.cat([h_mean, h_max], dim=-1)  # (batch, 128)


class HelperFleetEncoder(nn.Module):
    def __init__(self, in_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, helper_feats: torch.Tensor) -> torch.Tensor:
        if helper_feats.dim() == 2:
            helper_feats = helper_feats.unsqueeze(0)
        if helper_feats.shape[1] == 0:
            return torch.zeros(helper_feats.shape[0], 32, dtype=helper_feats.dtype, device=helper_feats.device)
        h = self.phi(helper_feats)
        return h.mean(dim=1)  # (batch, 32)


class StructuredPriorityPolicy(nn.Module):
    """Structured Priority & Opportunity-Cost Controller Network."""

    is_structured_priority_policy = True
    is_pb_crl_policy = True

    def __init__(
        self,
        task_in_dim: int = 5,
        helper_in_dim: int = 4,
        radio_hist_dim: int = 4,
        budget_dim: int = 3,
        hidden: int = 128,
        w_max: float = 5.0,
        init_log_std_w: Optional[Sequence[float] | float] = None,
        init_log_std_beta: Optional[float] = None,
        scale_w: float = 3.0,
        scale_beta: float = 1.5,
    ):
        super().__init__()
        self.w_max = float(w_max)
        self.scale_w = float(scale_w)
        self.scale_beta = float(scale_beta)

        self.task_enc = DeepSetsTaskEncoder(in_dim=task_in_dim, hidden_dim=64)
        self.helper_enc = HelperFleetEncoder(in_dim=helper_in_dim, hidden_dim=32)
        self.radio_gru = nn.GRU(input_size=radio_hist_dim, hidden_size=64, batch_first=True)
        self.budget_mlp = nn.Sequential(nn.Linear(budget_dim, 16), nn.Tanh())

        # Total fusion dim = 128 (tasks) + 32 (helpers) + 64 (radio) + 16 (budget) = 240
        fusion_dim = 128 + 32 + 64 + 16
        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Baseline anchors: w_ref = [2.5, 1.887, 1.887, 1.887], beta_ref = 0.881
        # Pre-sigmoid logit equivalents: sigmoid(0.0)*5 = 2.5; sigmoid(-0.5)*5 = 1.887; sigmoid(2.0) = 0.881
        self.register_buffer("base_z_w", torch.tensor([0.0, -0.5, -0.5, -0.5], dtype=torch.float32))
        self.register_buffer("base_z_beta", torch.tensor(2.0, dtype=torch.float32))

        # Action heads output bounded residual offsets in logit space:
        # delta_z_w = scale_w * tanh(head_w(trunk))
        # delta_z_beta = scale_beta * tanh(head_beta(trunk))
        self.head_w = nn.Linear(hidden, 4)
        self.head_beta = nn.Linear(hidden, 1)

        # Component-specific exploration standard deviations:
        # sigma = (sigma_v=0.2, sigma_q=0.8, sigma_o=0.8, sigma_r=0.2, sigma_beta=0.2)
        # Prioritizes informative exploration of latency and helper-occupancy tradeoffs
        # while keeping restrained exploration on utility ranking and budget throttle.
        if init_log_std_w is None:
            log_std_w_tensor = torch.tensor(
                [math.log(0.2), math.log(0.8), math.log(0.8), math.log(0.2)],
                dtype=torch.float32,
            )
        elif isinstance(init_log_std_w, (list, tuple, torch.Tensor)):
            log_std_w_tensor = torch.as_tensor(init_log_std_w, dtype=torch.float32)
        else:
            log_std_w_tensor = torch.full((4,), float(init_log_std_w), dtype=torch.float32)

        if init_log_std_beta is None:
            log_std_beta_val = math.log(0.2)
        else:
            log_std_beta_val = float(init_log_std_beta)

        self.log_std_w = nn.Parameter(log_std_w_tensor)
        self.log_std_beta = nn.Parameter(torch.tensor(log_std_beta_val, dtype=torch.float32))

        # Twin critics
        self.v_r_head = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(), nn.Linear(64, 1))
        self.v_c_head = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(), nn.Linear(64, 1))

        self._init_defaults()

    def _init_defaults(self):
        # Initialize residual heads to strictly 0.0 offset so initial deterministic policy
        # reproduces documented baseline B4 configuration exactly.
        nn.init.zeros_(self.head_w.weight)
        nn.init.zeros_(self.head_w.bias)
        nn.init.zeros_(self.head_beta.weight)
        nn.init.zeros_(self.head_beta.bias)

    def forward_trunk(
        self,
        task_feats: torch.Tensor,
        helper_feats: torch.Tensor,
        radio_hist: torch.Tensor,
        budget_block: torch.Tensor,
    ) -> torch.Tensor:
        e_task = self.task_enc(task_feats)
        e_hlp = self.helper_enc(helper_feats)

        if radio_hist.dim() == 2:
            radio_hist = radio_hist.unsqueeze(0)
        gru_out, _ = self.radio_gru(radio_hist)
        e_radio = gru_out[:, -1, :]

        if budget_block.dim() == 1:
            budget_block = budget_block.unsqueeze(0)
        e_budget = self.budget_mlp(budget_block)

        fused = torch.cat([e_task, e_hlp, e_radio, e_budget], dim=-1)
        return self.trunk(fused)

    def forward_actions(
        self,
        task_feats: torch.Tensor,
        helper_feats: torch.Tensor,
        radio_hist: torch.Tensor,
        budget_block: torch.Tensor,
        *,
        deterministic: bool = False,
    ):
        trunk_h = self.forward_trunk(task_feats, helper_feats, radio_hist, budget_block)
        delta_w = self.scale_w * torch.tanh(self.head_w(trunk_h))
        delta_b = self.scale_beta * torch.tanh(self.head_beta(trunk_h).squeeze(-1))

        mu_w = self.base_z_w.to(trunk_h.device) + delta_w
        mu_b = self.base_z_beta.to(trunk_h.device) + delta_b

        std_w = self.log_std_w.exp().expand_as(mu_w)
        std_b = self.log_std_beta.exp().expand_as(mu_b)

        dist_w = Normal(mu_w, std_w)
        dist_b = Normal(mu_b, std_b)

        if deterministic:
            z_w, z_b = mu_w, mu_b
        else:
            z_w, z_b = dist_w.sample(), dist_b.sample()

        w_vec = torch.sigmoid(z_w) * self.w_max
        beta = torch.sigmoid(z_b)

        logp = dist_w.log_prob(z_w).sum(dim=-1) + dist_b.log_prob(z_b)
        ent = dist_w.entropy().sum(dim=-1) + dist_b.entropy()
        return w_vec, beta, logp, z_w, z_b, ent

    def evaluate_actions(
        self,
        task_feats: torch.Tensor,
        helper_feats: torch.Tensor,
        radio_hist: torch.Tensor,
        budget_block: torch.Tensor,
        z_w: torch.Tensor,
        z_b: torch.Tensor,
    ):
        trunk_h = self.forward_trunk(task_feats, helper_feats, radio_hist, budget_block)
        delta_w = self.scale_w * torch.tanh(self.head_w(trunk_h))
        delta_b = self.scale_beta * torch.tanh(self.head_beta(trunk_h).squeeze(-1))

        mu_w = self.base_z_w.to(trunk_h.device) + delta_w
        mu_b = self.base_z_beta.to(trunk_h.device) + delta_b

        dist_w = Normal(mu_w, self.log_std_w.exp().expand_as(mu_w))
        dist_b = Normal(mu_b, self.log_std_beta.exp().expand_as(mu_b))

        zw = torch.as_tensor(z_w, dtype=torch.float32, device=mu_w.device)
        zb = torch.as_tensor(z_b, dtype=torch.float32, device=mu_b.device)

        logp = dist_w.log_prob(zw).sum(dim=-1) + dist_b.log_prob(zb)
        ent = dist_w.entropy().sum(dim=-1) + dist_b.entropy()
        return logp, ent

    def value(
        self,
        task_feats: torch.Tensor,
        helper_feats: Optional[torch.Tensor] = None,
        radio_hist: Optional[torch.Tensor] = None,
        budget_block: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if helper_feats is None or radio_hist is None or budget_block is None:
            return torch.tensor(0.0)
        trunk_h = self.forward_trunk(task_feats, helper_feats, radio_hist, budget_block)
        return self.v_r_head(trunk_h).squeeze(-1)

    def cost_value(
        self,
        task_feats: torch.Tensor,
        helper_feats: Optional[torch.Tensor] = None,
        radio_hist: Optional[torch.Tensor] = None,
        budget_block: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if helper_feats is None or radio_hist is None or budget_block is None:
            return torch.tensor(0.0)
        trunk_h = self.forward_trunk(task_feats, helper_feats, radio_hist, budget_block)
        return self.v_c_head(trunk_h).squeeze(-1)

    def act_numpy(self, state, layer=None, *, deterministic: bool = True):
        """Causal state feature extraction and priority weight inference."""
        if layer is not None:
            cands = layer.candidates(cbr=float(state.get("cbr", 0.0)))
            if cands:
                t_list = []
                for c in cands:
                    t_list.append([
                        float(c.get("norm_value", 0.5)),
                        float(c.get("payload_bytes", 0)) / 175000.0,
                        float(c.get("cycles", 5e7)) / 1e8,
                        float(c.get("lat_pressure", 0.5)),
                        float(c.get("finish_ms", 10.0)) / 100.0,
                    ])
                task_t = torch.tensor(t_list, dtype=torch.float32).unsqueeze(0)

                h_list = []
                for h in layer.helpers:
                    h_list.append([
                        float(h.cpu_cycles_s) / 1e9,
                        float(h.queue_cycles) / 1e8,
                        0.5,
                        0.5,
                    ])
                hlp_t = torch.tensor(h_list, dtype=torch.float32).unsqueeze(0)

                radio_t = torch.tensor([[float(state.get("cbr", 0.04)), 0.0, 0.2, 0.0]] * 10, dtype=torch.float32).unsqueeze(0)
                budget_t = torch.tensor([1.0, 0.5, 1.0], dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    w_vec, beta, logp, z_w, z_b, _ = self.forward_actions(
                        task_t, hlp_t, radio_t, budget_t, deterministic=deterministic
                    )
                    v_r = self.value(task_t, hlp_t, radio_t, budget_t)
                    v_c = self.cost_value(task_t, hlp_t, radio_t, budget_t)

                sp_obs = {
                    "task_t": task_t.detach().cpu(),
                    "hlp_t": hlp_t.detach().cpu(),
                    "radio_t": radio_t.detach().cpu(),
                    "budget_t": budget_t.detach().cpu(),
                    "value": float(v_r.item()),
                    "cost_value": float(v_c.item()),
                }
                return (
                    w_vec.squeeze(0).cpu().tolist(),
                    float(beta.item()),
                    logp,
                    z_w.squeeze(0).cpu().tolist(),
                    float(z_b.item()),
                    sp_obs,
                )

        return [2.5, 1.8877, 1.8877, 1.8877], 0.8808, torch.tensor(0.0), [0.0, -0.5, -0.5, -0.5], 2.0, None



def score_and_match(
    candidates: Sequence[Dict[str, Any]],
    weights: Sequence[float] | np.ndarray | torch.Tensor,
    beta_throttle: float,
    b_max: int = 175000,
    unique_sources: bool = True,
    unique_helpers: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Deterministic Priority Matching Algorithm (Algorithm 1 in specification).

    Candidate score:
      S(k, j) = w_v * V_kj - w_q * L_kj - w_o * O_kj - w_r * R_kj
    """
    if not candidates:
        return [], {"n_pack": 0, "bytes": 0, "scores": []}

    w = [float(x) for x in weights]
    w_v, w_q, w_o, w_r = w[0], w[1], w[2], w[3]
    b_limit = int(float(beta_throttle) * b_max)

    scored = []
    for c in candidates:
        # Check physical feasibility (C4)
        l_kj = float(c.get("lat_pressure", c.get("L_kj", 0.0)))
        contact_feas = bool(c.get("contact_feasible", True))
        if l_kj > 1.0 or not contact_feas:
            continue

        v_kj = float(c.get("norm_value", c.get("V_kj", 0.0)))
        o_kj = float(c.get("occ_cost", c.get("O_kj", 0.0)))
        r_kj = float(c.get("rad_footprint", c.get("R_kj", 0.0)))

        score = w_v * v_kj - w_q * l_kj - w_o * o_kj - w_r * r_kj
        scored.append((score, c))

    # Sort candidates descending by priority score
    scored.sort(key=lambda x: x[0], reverse=True)

    packed = []
    used_sources = set()
    used_helpers = set()
    total_bytes = 0
    scores_kept = []

    for s, c in scored:
        sid = int(c.get("source_id", -1))
        hid = int(c.get("helper_id", -1))
        b = int(c.get("payload_bytes", 0))

        if unique_sources and sid in used_sources:
            continue
        if unique_helpers and hid in used_helpers:
            continue
        if total_bytes + b > b_limit:
            continue

        packed.append(c)
        scores_kept.append(float(s))
        if unique_sources:
            used_sources.add(sid)
        if unique_helpers:
            used_helpers.add(hid)
        total_bytes += b

    meta = {
        "n_pack": len(packed),
        "bytes": total_bytes,
        "b_limit": b_limit,
        "weights": [round(x, 4) for x in w],
        "beta_throttle": float(beta_throttle),
        "scores": scores_kept,
    }
    return packed, meta
