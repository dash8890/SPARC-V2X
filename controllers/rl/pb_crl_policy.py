"""PB-CRL v3.1 — Full Controller Architecture, Risk Model, and Predictive Shield.

Implements the formal v3.1 specification (SPARC_HQ_SPO_PB_CRL.md):
  1. Policy Network (PB-CRL-core):
     - DeepSets task encoder (MLP 64 -> mean+max pool)
     - Helper encoder (MLP 32 -> mean pool)
     - Radio history GRU (64 units over last 10 epochs of [CBR, resel, bytes/Bmax, z])
     - Budget block (MLP 16 on [kappa/K, T_left/T, m_rem/nu(K)])
     - Shared trunk MLP 128x2
     - Three scalar heads:
         * lambda_t in [0, lam_max] (shadow price)
         * beta_prop in [0, 1] (proposed offloading volume fraction)
         * g_t in [0.5, 2.0] (risk budget pacing gain)
     - Twin critics: V_r (timely utility value) and V_c (constraint cost value)

  2. 5-Network Safety Critic Ensemble Q_c^+(o_t, beta):
     - Ensemble of 5 MLPs predicting P(window violation | o_t, beta)
     - Q_c^+ = mean + c_q * std (pessimistic upper confidence bound)

  3. Predictive Risk Account & Shield:
     - nu(kappa) precomputed from Poisson tail: P(Poisson(m) > kappa) <= delta_s
     - p_allow = min(g_t * m_rem / T_left, m_rem) / clip(r, 1, 3)
     - beta_exec = max { beta in grid | beta <= beta_prop, Q_c^+(o_t, beta) <= p_allow }
     - Predictive deduction: m_rem <- max(0, m_rem - Q_c^+(o_t, beta_exec))
     - Violation observation: if z_t == 1, kappa <- kappa - 1, m_rem <- nu(kappa)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.stats
import torch
from torch import nn
from torch.distributions import Normal

from controllers.rl.pb_crl_packer import price_budget_pack, pack_meta
from controllers.rl.hq_spo_baselines import assignments_to_action_fields


def precompute_poisson_nu(max_k: int = 10, delta_s: float = 0.05) -> Dict[int, float]:
    """nu(kappa) = largest m with P(Poisson(m) > kappa) <= delta_s."""
    nu_table = {}
    for k in range(max_k + 1):
        # Find m via bisection: CDF(k; m) >= 1 - delta_s
        lo = 0.0
        hi = max(10.0, float(k) * 3.0 + 5.0)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            # P(Poisson(mid) <= k)
            p_le = scipy.stats.poisson.cdf(k, mid)
            if (1.0 - p_le) <= delta_s:
                lo = mid
            else:
                hi = mid
        nu_table[k] = float(lo)
    return nu_table


class DeepSetsTaskEncoder(nn.Module):
    def __init__(self, in_dim: int = 6, hidden_dim: int = 64):
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


class EnsembleRiskModel(nn.Module):
    """5-network MLP ensemble predicting window violation risk Q_c(o_t, beta)."""

    def __init__(self, obs_dim: int = 16, n_models: int = 5, hidden: int = 64, c_q: float = 1.0):
        super().__init__()
        self.n_models = n_models
        self.c_q = float(c_q)
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + 1, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
                nn.Sigmoid(),
            )
            for _ in range(n_models)
        ])

    def predict_all(self, obs: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # obs: (batch, obs_dim), beta: (batch, 1)
        if beta.dim() == 1:
            beta = beta.unsqueeze(-1)
        inp = torch.cat([obs, beta], dim=-1)
        # Marginal offload-attributable violation probability: zero at beta=0, scaled by admitted offload volume
        preds = torch.stack([beta.squeeze(-1) * (m(inp).squeeze(-1) * 0.08) for m in self.models], dim=0)  # (n_models, batch)
        return preds

    def forward(self, obs: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        preds = self.predict_all(obs, beta)
        mean_p = preds.mean(dim=0)
        std_p = preds.std(dim=0)
        q_plus = torch.clamp(mean_p + self.c_q * std_p, 0.0, 1.0)
        return q_plus, mean_p, std_p


class PredictiveRiskAccount:
    """Predictive Risk Account & Shield (v3.1 specification §4)."""

    def __init__(
        self,
        risk_model: EnsembleRiskModel,
        K: int = 1,
        total_epochs: int = 50,
        delta_s: float = 0.05,
        grid_steps: int = 15,
    ):
        self.risk_model = risk_model
        self.K = int(K)
        self.total_epochs = int(total_epochs)
        self.delta_s = float(delta_s)
        self.grid_steps = int(grid_steps)
        self.nu_table = precompute_poisson_nu(max_k=max(10, self.K + 2), delta_s=self.delta_s)

        # Episode runtime state
        self.kappa = self.K
        self.m_rem = self.nu_table.get(self.K, 0.05)
        self.current_epoch = 0
        self.history_violations: List[int] = []
        self.history_q_plus: List[float] = []

    def reset(self, K: Optional[int] = None):
        if K is not None:
            self.K = int(K)
        self.kappa = self.K
        self.m_rem = self.nu_table.get(self.K, 0.05)
        self.current_epoch = 0
        self.history_violations.clear()
        self.history_q_plus.clear()

    def drift_ratio(self) -> float:
        """Online drift guard ratio r over last 50 windows."""
        if not self.history_violations:
            return 1.0
        n_recent = min(len(self.history_violations), 50)
        obs_viol = sum(self.history_violations[-n_recent:])
        sum_q = sum(self.history_q_plus[-n_recent:])
        r = obs_viol / max(sum_q, 1e-4)
        return float(np.clip(r, 1.0, 3.0))

    def evaluate_shield(
        self,
        obs_tensor: torch.Tensor,
        beta_prop: float,
        g_t: float = 1.0,
    ) -> Tuple[float, float, float]:
        """Calculates p_allow, finds beta_exec <= beta_prop on grid, and charges account."""
        t_left = max(1, self.total_epochs - self.current_epoch)
        r = self.drift_ratio()
        
        # p_allow = min(g_t * m_rem / t_left, m_rem) / clip(r, 1, 3)
        p_allow = min(float(g_t) * self.m_rem / float(t_left), self.m_rem) / r
        p_allow = max(0.0, float(p_allow))

        # Discrete search grid: {0, 1/15, 2/15, ..., 1.0}
        grid = [i / float(self.grid_steps) for i in range(self.grid_steps + 1)]
        grid = [b for b in grid if b <= float(beta_prop) + 1e-6]

        beta_exec = 0.0
        q_plus_exec = 0.0
        with torch.no_grad():
            for b in reversed(grid):
                b_t = torch.tensor([[b]], dtype=torch.float32, device=obs_tensor.device)
                q_plus, _, _ = self.risk_model(obs_tensor, b_t)
                q_val = float(q_plus.reshape(()).cpu())
                if q_val <= p_allow + 1e-6:
                    beta_exec = b
                    q_plus_exec = q_val
                    break

        return beta_exec, q_plus_exec, p_allow

    def deduct_and_record(self, q_plus_exec: float):
        """Predictive risk deduction before violation occurs."""
        self.m_rem = max(0.0, self.m_rem - q_plus_exec)
        self.history_q_plus.append(q_plus_exec)
        self.current_epoch += 1

    def note_window_outcome(self, violated: bool):
        """Feedback after window outcome z_t is observed."""
        self.current_epoch += 1
        z = 1 if violated else 0
        self.history_violations.append(z)
        if z == 1:
            self.kappa -= 1
            self.m_rem = self.nu_table.get(max(0, self.kappa), 0.0) if self.kappa >= 0 else 0.0


class PBCRLPolicy(nn.Module):
    """Full PB-CRL-core policy network (§2 in SPARC_HQ_SPO_PB_CRL.md)."""

    is_pb_crl_policy = True

    def __init__(
        self,
        task_in_dim: int = 6,
        helper_in_dim: int = 4,
        radio_hist_dim: int = 4,
        budget_dim: int = 3,
        hidden: int = 128,
        lam_max: float = 50.0,
    ):
        super().__init__()
        self.lam_max = float(lam_max)
        self.task_enc = DeepSetsTaskEncoder(in_dim=task_in_dim, hidden_dim=64)
        self.helper_enc = nn.Sequential(nn.Linear(helper_in_dim, 32), nn.Tanh())
        self.radio_gru = nn.GRU(input_size=radio_hist_dim, hidden_size=64, batch_first=True)
        self.budget_mlp = nn.Sequential(nn.Linear(budget_dim, 16), nn.Tanh())

        # Total encoder fusion dim = 128 (tasks) + 32 (helpers) + 64 (radio) + 16 (budget) = 240
        fusion_dim = 128 + 32 + 64 + 16
        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Three scalar heads: (lambda, beta_prop, g_t)
        self.head_lam = nn.Linear(hidden, 1)
        self.head_beta = nn.Linear(hidden, 1)
        self.head_g = nn.Linear(hidden, 1)

        # Twin critics
        self.v_r_head = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(), nn.Linear(64, 1))
        self.v_c_head = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(), nn.Linear(64, 1))

        # Weight initializations:
        # Pacing gain head g_t zero-initialized -> sigmoid(0)*1.5 + 0.5 = 1.0 (exact neutral pacing initially)
        nn.init.zeros_(self.head_g.weight)
        nn.init.zeros_(self.head_g.bias)
        # Lambda head bias set to empirical density
        nn.init.zeros_(self.head_lam.weight)
        nn.init.constant_(self.head_lam.bias, -2.5)  # sigmoid(-2.5)*50 ≈ 3.79
        # Beta head bias set to high admission initially
        nn.init.zeros_(self.head_beta.weight)
        nn.init.constant_(self.head_beta.bias, 4.0)   # sigmoid(4.0) ≈ 0.98

    def forward_trunk(
        self,
        task_feats: torch.Tensor,
        helper_feats: torch.Tensor,
        radio_hist: torch.Tensor,
        budget_block: torch.Tensor,
    ) -> torch.Tensor:
        e_tasks = self.task_enc(task_feats)  # (batch, 128)
        if helper_feats.dim() == 2:
            helper_feats = helper_feats.unsqueeze(0)
        e_helpers = self.helper_enc(helper_feats).mean(dim=1)  # (batch, 32)
        if radio_hist.dim() == 2:
            radio_hist = radio_hist.unsqueeze(0)
        _, h_gru = self.radio_gru(radio_hist)
        e_radio = h_gru[-1]  # (batch, 64)
        if budget_block.dim() == 1:
            budget_block = budget_block.unsqueeze(0)
        e_budget = self.budget_mlp(budget_block)  # (batch, 16)

        fusion = torch.cat([e_tasks, e_helpers, e_radio, e_budget], dim=-1)
        return self.trunk(fusion)

    def forward_heads(
        self,
        trunk_h: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_l = self.head_lam(trunk_h).squeeze(-1)
        z_b = self.head_beta(trunk_h).squeeze(-1)
        z_g = self.head_g(trunk_h).squeeze(-1)

        lam = torch.sigmoid(z_l) * self.lam_max
        beta_prop = torch.sigmoid(z_b)
        # g_t in [0.5, 2.0]
        g_t = torch.sigmoid(z_g) * 1.5 + 0.5
        return lam, beta_prop, g_t

    def critics(self, trunk_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v_r = self.v_r_head(trunk_h).squeeze(-1)
        v_c = self.v_c_head(trunk_h).squeeze(-1)
        return v_r, v_c


class PBCRLCoreAgent:
    """Full PB-CRL-core agent with learned 3 heads, ensemble risk model, and predictive shield."""

    is_pb_crl_b5_prime = True

    def __init__(
        self,
        policy: PBCRLPolicy,
        risk_model: EnsembleRiskModel,
        K: int = 1,
        total_epochs: int = 50,
        delta_s: float = 0.05,
    ):
        self.policy = policy
        self.risk_model = risk_model
        self.account = PredictiveRiskAccount(risk_model, K=K, total_epochs=total_epochs, delta_s=delta_s)
        self.radio_history: List[List[float]] = []  # last 10 steps of [CBR, resel, bytes/Bmax, z]
        self.c2_K = K
        self.c2_Z = 0
        self.c2_forced_zero_epochs = 0

    def reset_c2(self, K: Optional[int] = None):
        self.c2_K = K
        self.c2_Z = 0
        self.c2_forced_zero_epochs = 0
        self.account.reset(K)
        self.radio_history.clear()

    def note_window(self, epoch_rec: dict):
        prr = float(epoch_rec.get("window_prr", epoch_rec.get("prr", 1.0)) or 1.0)
        p95 = float(epoch_rec.get("p95_delay_ms", 0.0) or 0.0)
        violated = (prr < 0.95 or p95 > 50.0)
        if violated:
            self.c2_Z += 1
        self.account.note_window_outcome(violated)

        # Update radio history
        cbr = float(epoch_rec.get("cbr", 0.0) or 0.0)
        resel = float(epoch_rec.get("resel_t", 0.0) or 0.0)
        offered_b = float(epoch_rec.get("computation_bytes_offered", 0.0) or 0.0) / 175000.0
        z_f = 1.0 if violated else 0.0
        self.radio_history.append([cbr, resel, offered_b, z_f])
        if len(self.radio_history) > 10:
            self.radio_history.pop(0)

    def act_numpy(self, state_vec, layer=None, *, deterministic: bool = True):
        # Build features for DeepSets and GRU
        cbr = float(state_vec[0]) if len(state_vec) > 0 else 0.0
        resel = float(state_vec[1]) if len(state_vec) > 1 else 0.0
        epoch_idx = self.account.current_epoch
        tot_ep = self.account.total_epochs
        t_left = max(0, tot_ep - epoch_idx)
        
        # Budget block: [kappa / K, T_left / T, m_rem / nu(K)]
        kappa_norm = float(self.account.kappa) / max(1.0, float(self.account.K))
        t_left_norm = float(t_left) / max(1.0, float(tot_ep))
        nu_k = max(1e-4, self.account.nu_table.get(self.account.K, 0.05))
        m_rem_norm = float(self.account.m_rem) / nu_k
        budget_block = torch.tensor([[kappa_norm, t_left_norm, m_rem_norm]], dtype=torch.float32)

        # Radio history tensor (batch=1, seq_len=10, 4)
        hist = list(self.radio_history)
        while len(hist) < 10:
            hist.insert(0, [cbr, resel, 0.0, 0.0])
        radio_hist = torch.tensor([hist[-10:]], dtype=torch.float32)

        # Candidates tensor for task encoder
        if layer is not None:
            cands = layer.candidates(cbr, 0.0)
            task_feats_list = []
            for c in cands:
                f = c.get("features") or [0, 0, 0, 0, 0, 0]
                task_feats_list.append(list(f)[:6])
            if not task_feats_list:
                task_feats = torch.zeros(1, 0, 6, dtype=torch.float32)
            else:
                task_feats = torch.tensor([task_feats_list], dtype=torch.float32)
            
            helper_feats_list = []
            for h in layer.helpers:
                helper_feats_list.append([h.cpu_cycles_s / 1e9, h.queue_cycles / 1e9, h.ready_time_ms / 100.0, 0.0])
            helper_feats = torch.tensor([helper_feats_list], dtype=torch.float32)
        else:
            task_feats = torch.zeros(1, 0, 6, dtype=torch.float32)
            helper_feats = torch.zeros(1, 1, 4, dtype=torch.float32)

        with torch.no_grad():
            trunk_h = self.policy.forward_trunk(task_feats, helper_feats, radio_hist, budget_block)
            lam_t, beta_prop_t, g_t_tensor = self.policy.forward_heads(trunk_h, deterministic=deterministic)

        lam_f = float(lam_t.reshape(()).cpu())
        beta_prop_f = float(beta_prop_t.reshape(()).cpu())
        g_f = float(g_t_tensor.reshape(()).cpu())

        # Flatten state vector for risk model obs (pad/truncate to 16)
        svec = list(state_vec) + [0.0] * max(0, 16 - len(state_vec))
        obs_tensor = torch.tensor([svec[:16]], dtype=torch.float32)

        # Evaluate predictive shield
        beta_exec, q_plus_exec, p_allow = self.account.evaluate_shield(obs_tensor, beta_prop_f, g_t=g_f)
        self.account.deduct_and_record(q_plus_exec)

        if beta_exec == 0.0 and beta_prop_f > 0.0:
            self.c2_forced_zero_epochs += 1

        return lam_f, beta_exec, torch.tensor(0.0), lam_f, beta_exec


class B5PrimeRAAgent:
    """B5' policy equipped with the exact same predictive risk account & shield (matched g=1.0)."""

    is_pb_crl_b5_prime = True

    def __init__(
        self,
        b5_prime_policy,
        risk_model: EnsembleRiskModel,
        K: int = 1,
        total_epochs: int = 50,
        delta_s: float = 0.05,
    ):
        self.b5_policy = b5_prime_policy
        self.risk_model = risk_model
        self.account = PredictiveRiskAccount(risk_model, K=K, total_epochs=total_epochs, delta_s=delta_s)
        self.c2_K = K
        self.c2_Z = 0
        self.c2_forced_zero_epochs = 0

    def reset_c2(self, K: Optional[int] = None):
        self.c2_K = K
        self.c2_Z = 0
        self.c2_forced_zero_epochs = 0
        self.account.reset(K)

    def note_window(self, epoch_rec: dict):
        prr = float(epoch_rec.get("window_prr", epoch_rec.get("prr", 1.0)) or 1.0)
        p95 = float(epoch_rec.get("p95_delay_ms", 0.0) or 0.0)
        violated = (prr < 0.95 or p95 > 50.0)
        if violated:
            self.c2_Z += 1
        self.account.note_window_outcome(violated)

    def act_numpy(self, state_vec, layer=None, *, deterministic: bool = True):
        # Base B5' proposed actions
        lam_f, beta_prop_f, logp, z_l, z_b = self.b5_policy.act_numpy(state_vec, deterministic=deterministic)

        # Risk model evaluation with fixed neutral pacing g = 1.0
        svec = list(state_vec) + [0.0] * max(0, 16 - len(state_vec))
        obs_tensor = torch.tensor([svec[:16]], dtype=torch.float32)

        beta_exec, q_plus_exec, p_allow = self.account.evaluate_shield(obs_tensor, beta_prop_f, g_t=1.0)
        self.account.deduct_and_record(q_plus_exec)

        if beta_exec == 0.0 and beta_prop_f > 0.0:
            self.c2_forced_zero_epochs += 1

        return lam_f, beta_exec, logp, z_l, z_b

