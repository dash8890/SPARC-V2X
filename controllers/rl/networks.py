import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNUtilityNetwork(nn.Module):
    """
    Parameter-shared recurrent utility network for each UV.
    h_{u,t} = GRU(h_{u,t-1}, o_{u,t}, a_{u,t-1})
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Input features: observation + one-hot previous action
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, prev_action, hidden_state):
        # Concatenate obs and one-hot encoded prev_action
        x = torch.cat([obs, prev_action], dim=-1)
        x = F.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h_out = self.rnn(x, h_in)
        q = self.fc2(h_out)
        return q, h_out
        
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim)
