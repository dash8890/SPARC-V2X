import torch
from controllers.rl.networks import RNNUtilityNetwork
from controllers.rl.action_lattice import action_lattice

class LearnedAdmissionPolicy:
    """Campaign evaluator adapter for a trained QMIX/QPLEX utility network."""
    def __init__(self, checkpoint, action_dim=16, obs_dim=16):
        state = torch.load(checkpoint, map_location="cpu")
        action_dim = int(state["fc2.bias"].shape[0])
        obs_dim = int(state["fc1.weight"].shape[1] - action_dim)
        self.action_dim = action_dim
        self.net = RNNUtilityNetwork(obs_dim, action_dim, hidden_dim=64)
        self.net.load_state_dict(state, strict=True)
        self.net.eval()
        self.hidden = {}

    def reset(self):
        self.hidden.clear()

    def select_action(self, observation, k, h_count, uv_id="default"):
        if h_count < k:
            return None
        x = torch.tensor(observation, dtype=torch.float32).reshape(1, -1)
        prev = torch.zeros(1, self.action_dim)
        hidden = self.hidden.get(uv_id, self.net.init_hidden(1))
        with torch.no_grad():
            q, hidden = self.net(x, prev, hidden)
        self.hidden[uv_id] = hidden
        lattice = action_lattice(k, h_count, self.action_dim)
        mask = torch.zeros(self.action_dim, dtype=torch.bool); mask[:len(lattice)] = True
        idx = int(q.masked_fill(~mask, -torch.inf).argmax(dim=1).item())
        return int(lattice[idx])
