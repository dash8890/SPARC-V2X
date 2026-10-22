import torch
from torch import nn
class TaskPairPolicy(nn.Module):
    def __init__(self,dim=10): super().__init__(); self.net=nn.Sequential(nn.Linear(dim,64),nn.Tanh(),nn.Linear(64,1))
    def score(self,x): return self.net(x).squeeze(-1)
