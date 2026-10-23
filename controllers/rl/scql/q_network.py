"""Graph-Conditioned Dueling Q-Network for SCQL (Sequential Capacity-Aware Q-Learning).

Architecture:
1. Task Encoder MLP: maps task state [v_i, deadline_rem, bytes, cycles, wait_time] -> h_i^T in R^64
2. Helper Fleet Encoder MLP: maps helper state [f_j, queue_cycles, t_avail, rel_pos, rel_vel] -> h_j^H in R^64
3. Causal History GRU: processes trailing radio/traffic history -> h^R in R^64
4. Edge Encoder MLP: combines (h_i^T, h_j^H, h^R, raw_edge_features) -> u_{ij} in R^64
5. Partial Matching Aggregator: sum-pools selected edges -> m_t in R^64
6. Dueling Q-Streams:
   - State Value Head V(s, M) in R^1
   - Advantage Stream A(s, M, (i, j)) in R^1 per candidate edge
   - Advantage Head A(s, M, STOP) in R^1 for stopping action
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn

from .action_mask import CandidateEdge


class SCQLDuelingQNetwork(nn.Module):
    def __init__(
        self,
        task_dim: int = 5,
        helper_dim: int = 7,
        radio_hist_dim: int = 4,
        edge_raw_dim: int = 10,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Encoders
        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.helper_encoder = nn.Sequential(
            nn.Linear(helper_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.history_gru = nn.GRU(
            input_size=radio_hist_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # 2. Edge representation
        edge_in_dim = hidden_dim * 3 + edge_raw_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # 3. Partial matching encoder
        self.partial_match_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # 4. Joint State Context: [h^R, m_t, b_rem_norm]
        joint_dim = hidden_dim * 2 + 1
        self.joint_trunk = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
        )

        # 5. Dueling Streams
        # Value stream: V(s, M)
        self.v_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Advantage stream for candidate edges: A(s, M, (i, j))
        self.a_edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Advantage head for STOP action: A(s, M, STOP)
        self.a_stop_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def extract_features(
        self,
        candidates: Sequence[CandidateEdge],
        history: torch.Tensor,
        b_rem: float,
        b_max: float = 175000.0,
        partial_matching: Optional[Sequence[CandidateEdge]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts edge embeddings u_ij, partial matching embedding m_t, and joint state s_joint."""
        device = next(self.parameters()).device

        # 1. History context h^R
        if history.dim() == 2:
            history = history.unsqueeze(0)
        history = history.to(device)
        _, h_gru = self.history_gru(history)
        h_r = h_gru[-1]  # (1, hidden_dim)

        # 2. Candidate edge embeddings
        u_edges = []
        for c in candidates:
            # Task features: [v_i, deadline_rem_s, bytes_norm, cycles_norm, wait_s]
            t_feat = torch.tensor([
                c.task_value / 5.0,
                c.deadline_ms / 1000.0,
                c.payload_bytes / 175000.0,
                c.cycles / 1e9,
                (c.features[8] if len(c.features) > 8 else 0.0),
            ], dtype=torch.float32, device=device).unsqueeze(0)
            h_t = self.task_encoder(t_feat)

            # Helper features (dim 7) directly observing capacity opportunity cost:
            # 1. Remaining queued computation: c.queue_cycles / 1e8
            # 2. Estimated release time: c.t_release_s / 10.0
            # 3. Remaining contact duration: c.t_contact_sec / 100.0
            # 4. Current tasks feasibly competing for helper: c.competing_tasks / 10.0
            # 5. Total value of competing tasks: c.competing_value / 25.0
            # 6. Future processing capacity remaining post-assignment: c.post_slack_sec / 100.0
            # 7. Helper CPU compute speed: c.cpu_cycles_s / 1e9
            h_feat = torch.tensor([
                c.queue_cycles / 1e8,
                min(1.0, c.t_release_s / 10.0),
                min(1.0, c.t_contact_sec / 100.0),
                min(1.0, float(c.competing_tasks) / 10.0),
                min(1.0, float(c.competing_value) / 25.0),
                min(1.0, c.post_slack_sec / 100.0),
                c.cpu_cycles_s / 1e9,
            ], dtype=torch.float32, device=device).unsqueeze(0)
            h_h = self.helper_encoder(h_feat)

            # Raw edge features (dim 10)
            raw_edge = torch.tensor([
                c.task_value / 5.0,
                c.deadline_ms / 1000.0,
                c.payload_bytes / 175000.0,
                c.cycles / 1e9,
                c.finish_ms / 1000.0,
                min(1.0, c.t_contact_sec / 100.0),
                min(1.0, c.post_slack_sec / 100.0),
                c.occ_cost,
                c.lat_pressure,
                c.risk,
            ], dtype=torch.float32, device=device).unsqueeze(0)

            fused_in = torch.cat([h_t, h_h, h_r, raw_edge], dim=-1)
            u_ij = self.edge_encoder(fused_in)
            u_edges.append(u_ij)

        if u_edges:
            u_tensor = torch.cat(u_edges, dim=0)  # (N_candidates, hidden_dim)
        else:
            u_tensor = torch.zeros(0, self.hidden_dim, device=device)

        # 3. Partial matching embedding m_t
        if partial_matching and len(partial_matching) > 0:
            m_embeds = []
            for pm in partial_matching:
                # Find or compute embedding for committed edge
                t_feat = torch.tensor([
                    pm.task_value / 5.0, pm.deadline_ms / 1000.0, pm.payload_bytes / 175000.0,
                    pm.cycles / 1e9, 0.0
                ], dtype=torch.float32, device=device).unsqueeze(0)
                h_feat = torch.tensor([
                    pm.queue_cycles / 1e8,
                    min(1.0, pm.t_release_s / 10.0),
                    min(1.0, pm.t_contact_sec / 100.0),
                    min(1.0, float(pm.competing_tasks) / 10.0),
                    min(1.0, float(pm.competing_value) / 25.0),
                    min(1.0, pm.post_slack_sec / 100.0),
                    pm.cpu_cycles_s / 1e9,
                ], dtype=torch.float32, device=device).unsqueeze(0)
                raw = torch.tensor([
                    pm.task_value / 5.0, pm.deadline_ms / 1000.0, pm.payload_bytes / 175000.0,
                    pm.cycles / 1e9, pm.finish_ms / 1000.0, min(1.0, pm.t_contact_sec / 100.0),
                    min(1.0, pm.post_slack_sec / 100.0), pm.occ_cost, pm.lat_pressure, pm.risk
                ], dtype=torch.float32, device=device).unsqueeze(0)
                u_pm = self.edge_encoder(torch.cat([self.task_encoder(t_feat), self.helper_encoder(h_feat), h_r, raw], dim=-1))
                m_embeds.append(self.partial_match_proj(u_pm))
            m_t = torch.stack(m_embeds).sum(dim=0)
        else:
            m_t = torch.zeros(1, self.hidden_dim, device=device)

        # 4. Joint state context
        b_norm = torch.tensor([[float(b_rem) / float(b_max)]], dtype=torch.float32, device=device)
        joint_in = torch.cat([h_r, m_t, b_norm], dim=-1)
        s_joint = self.joint_trunk(joint_in)  # (1, hidden_dim * 2)

        return u_tensor, s_joint, h_r

    def forward(
        self,
        candidates: Sequence[CandidateEdge],
        history: torch.Tensor,
        b_rem: float,
        action_mask: Optional[Sequence[bool]] = None,
        partial_matching: Optional[Sequence[CandidateEdge]] = None,
        b_max: float = 175000.0,
    ) -> torch.Tensor:
        """Computes Q-values for all candidate edges + STOP action.
        
        Returns:
          q_values: torch.Tensor of shape (N_candidates + 1,)
        """
        device = next(self.parameters()).device
        n_cands = len(candidates)
        u_tensor, s_joint, _ = self.extract_features(
            candidates, history, b_rem, b_max=b_max, partial_matching=partial_matching,
        )

        # 1. State Value V(s, M)
        val = self.v_head(s_joint).squeeze()  # scalar

        # 2. Advantage values
        adv_edges = []
        if n_cands > 0:
            s_expanded = s_joint.expand(n_cands, -1)
            pair_in = torch.cat([s_expanded, u_tensor], dim=-1)
            adv_edges = self.a_edge_head(pair_in).squeeze(-1)  # (N_candidates,)
        else:
            adv_edges = torch.empty(0, device=device)

        adv_stop = self.a_stop_head(s_joint).squeeze()  # scalar

        # Combine advantages: [adv_edge_0, ..., adv_edge_{N-1}, adv_stop]
        adv_all = torch.cat([adv_edges, adv_stop.unsqueeze(0)])  # (N_candidates + 1,)

        # 3. Dueling Q-values: Q = V + (A - mean(A_admissible))
        if action_mask is not None:
            mask_t = torch.tensor(action_mask, dtype=torch.bool, device=device)
            # Compute mean only over admissible actions
            admissible_adv = adv_all[mask_t]
            adv_mean = admissible_adv.mean() if len(admissible_adv) > 0 else adv_all.mean()
            q_values = val + (adv_all - adv_mean)
            # Mask inadmissible actions with large negative value
            q_values[~mask_t] = -1e9
        else:
            q_values = val + (adv_all - adv_all.mean())

        return q_values

    def evaluate_matching(
        self,
        matching: Sequence[CandidateEdge],
        history: torch.Tensor,
        b_rem: float,
        b_max: float = 175000.0,
    ) -> torch.Tensor:
        """Evaluates Q_theta(s, M) for a complete proposed matching M and its resulting resource commitments."""
        device = next(self.parameters()).device
        if history.dim() == 2:
            history = history.unsqueeze(0)
        history = history.to(device)
        _, h_gru = self.history_gru(history)
        h_r = h_gru[-1]  # (1, hidden_dim)

        if matching and len(matching) > 0:
            m_embeds = []
            for pm in matching:
                t_feat = torch.tensor([
                    pm.task_value / 5.0, pm.deadline_ms / 1000.0, pm.payload_bytes / 175000.0,
                    pm.cycles / 1e9, 0.0
                ], dtype=torch.float32, device=device).unsqueeze(0)
                h_feat = torch.tensor([
                    pm.queue_cycles / 1e8,
                    min(1.0, pm.t_release_s / 10.0),
                    min(1.0, pm.t_contact_sec / 100.0),
                    min(1.0, float(pm.competing_tasks) / 10.0),
                    min(1.0, float(pm.competing_value) / 25.0),
                    min(1.0, pm.post_slack_sec / 100.0),
                    pm.cpu_cycles_s / 1e9,
                ], dtype=torch.float32, device=device).unsqueeze(0)
                raw = torch.tensor([
                    pm.task_value / 5.0, pm.deadline_ms / 1000.0, pm.payload_bytes / 175000.0,
                    pm.cycles / 1e9, pm.finish_ms / 1000.0, min(1.0, pm.t_contact_sec / 100.0),
                    min(1.0, pm.post_slack_sec / 100.0), pm.occ_cost, pm.lat_pressure, pm.risk
                ], dtype=torch.float32, device=device).unsqueeze(0)
                u_pm = self.edge_encoder(torch.cat([self.task_encoder(t_feat), self.helper_encoder(h_feat), h_r, raw], dim=-1))
                m_embeds.append(self.partial_match_proj(u_pm))
            m_t = torch.stack(m_embeds).sum(dim=0)
        else:
            m_t = torch.zeros(1, self.hidden_dim, device=device)

        b_norm = torch.tensor([[float(b_rem) / float(b_max)]], dtype=torch.float32, device=device)
        joint_in = torch.cat([h_r, m_t, b_norm], dim=-1)
        s_joint = self.joint_trunk(joint_in)
        return self.v_head(s_joint).squeeze()

    def score_candidate_edges(
        self,
        candidates: Sequence[CandidateEdge],
        history: torch.Tensor,
        partial_matching: Sequence[CandidateEdge],
        b_rem: float,
        b_max: float = 175000.0,
    ) -> torch.Tensor:
        """Scores candidate edges for sequential construction of alternative matching M_R."""
        u_tensor, s_joint, _ = self.extract_features(
            candidates, history, b_rem, b_max=b_max, partial_matching=partial_matching,
        )
        n_cands = len(candidates)
        if n_cands == 0:
            return torch.empty(0, device=next(self.parameters()).device)
        s_expanded = s_joint.expand(n_cands, -1)
        pair_in = torch.cat([s_expanded, u_tensor], dim=-1)
        return self.a_edge_head(pair_in).squeeze(-1)
