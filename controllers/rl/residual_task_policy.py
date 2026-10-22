import torch
from torch import nn


class ResidualTaskPolicy(nn.Module):
    """Value-Aware anchored residual policy with an ordinary rollout critic.

    WiLab analytic scores are O(1e-2) with top-1/top-2 gaps ~1e-3, so raw
    Categorical(logits=analytic+residual) is nearly uniform and PPO cannot
    move rankings. ``logit_scale`` puts decisions in a peaked logit regime
    while preserving the analytic argmax when the residual is zero.

    ``logit_mode``:
      - ``va_residual`` (SPARC/A0): alpha * (analytic + beta * residual)
      - ``direct`` (B3/A4): alpha * residual only (no additive VA)
      - ``residual_only`` (A5): same actor form as direct; training may still
        use VA-relative advantages separately
    """

    def __init__(self, dim=10, logit_scale=100.0, residual_gain=1.0,
                 logit_mode: str = "va_residual"):
        super().__init__()
        if logit_mode not in ("va_residual", "direct", "residual_only"):
            raise ValueError(f"unsupported logit_mode={logit_mode!r}")
        self.logit_scale = float(logit_scale)
        self.residual_gain = float(residual_gain)
        self.logit_mode = str(logit_mode)
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.value_head = nn.Sequential(nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.cost_head = nn.Sequential(nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, 1))

    def residual(self, x):
        return self.net(x).squeeze(-1)

    def value(self, x):
        return self.value_head(x).squeeze(-1)

    def cost_value(self, x):
        return self.cost_head(x).squeeze(-1)

    def score_logits(self, x, analytic):
        if not torch.is_tensor(analytic):
            analytic = torch.tensor(analytic, dtype=torch.float32)
        residual = self.residual_gain * self.residual(x)
        if self.logit_mode in ("direct", "residual_only"):
            return self.logit_scale * residual
        return self.logit_scale * (analytic + residual)
