import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from controllers.rl.pb_crl_policy import PBCRLPolicy
from controllers.rl.pb_crl_ppo import pb_crl_ppo_update, build_pb_crl_steps

policy = PBCRLPolicy()
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

trace = []
for i in range(10):
    fobs = {
        'task_feats': [[1, 2, 3, 4, 5, 6]],
        'helper_feats': [[1, 2, 3, 4]],
        'radio_hist': [[0.1, 0.2, 0.3, 0.0]] * 10,
        'budget_block': [1.0, 0.5, 0.2],
    }
    tf = torch.tensor([fobs['task_feats']], dtype=torch.float32)
    hf = torch.tensor([fobs['helper_feats']], dtype=torch.float32)
    rh = torch.tensor([fobs['radio_hist']], dtype=torch.float32)
    bb = torch.tensor([fobs['budget_block']], dtype=torch.float32)
    
    with torch.no_grad():
        _, _, _, logp_init, z_l, z_b, z_g, _ = policy.forward_actions(tf, hf, rh, bb, deterministic=False)
        v_init = policy.value(tf, hf, rh, bb)
        c_init = policy.cost_value(tf, hf, rh, bb)

    item = {
        'reward': 2.0,
        'value': v_init,
        'cost_value': c_init,
        'logp': logp_init,
        'cbr_valid': True,
        'delay_valid': True,
        'prr': 0.98,
        'p95_delay_ms': 20.0,
        'pb_z': [float(z_l), float(z_b), float(z_g)],
        'pb_full_obs': fobs,
    }
    trace.append(item)

stats = pb_crl_ppo_update(policy, optimizer, [trace], lam_L=0.0, epochs=2, minibatch_size=4)
print('PPO Update Output:', stats)
assert stats is not None
assert stats['approx_kl'] < 0.1, f"Expected small KL, got {stats['approx_kl']}"
print('ALL TESTS PASSED!')
