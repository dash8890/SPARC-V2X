import torch
from torch import nn
class SafeAdmissionPolicy(nn.Module):
    def __init__(self, obs_dim=8, actions=5):
        super().__init__(); self.net=nn.Sequential(nn.Linear(obs_dim,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh()); self.actor=nn.Linear(64,actions); self.critic=nn.Linear(64,1)
    def forward(self,x):
        z=self.net(x); return self.actor(z),self.critic(z).squeeze(-1)

def conservative_safe_mask(obs, gamma=.95, dmax_ms=50., kappa=.5):
    """Causal lower-PRR/upper-delay action shield for actions 0..4."""
    cbr,norm_n,prr,p95=map(float,obs[:4]); n=norm_n*100.
    mask=[]
    for a in range(5):
        pred_prr=max(0.,prr-0.006*a*(n/10.0)); lower=pred_prr-kappa*.02
        pred_delay=p95*100.+5.*a*(n/10.0); upper=pred_delay+kappa*5.
        mask.append(lower>=gamma and upper<=dmax_ms)
    if not any(mask): mask[0]=True
    return torch.tensor(mask,dtype=torch.bool)
