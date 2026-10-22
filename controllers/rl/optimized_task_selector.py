"""Behavior-preserving task/helper candidate selector.

The candidate order and score equations intentionally match the frozen
evaluator. Only Python object construction and repeated dictionary lookups are
reduced.
"""
import numpy as np
import torch
from controllers.rl.safe_admission import conservative_safe_mask


def prepare(env):
    obs = env.obs()
    k = int(conservative_safe_mask(obs).sum().item())
    tasks = [t for t in env.tasks if not t['done']]
    helpers = list(env.helpers.items())
    n = len(tasks) * len(helpers)
    features = np.empty((max(n, 0), 10), dtype=np.float32)
    task_ids, helper_ids = [], []
    finish_values, task_values = [], []
    if n == 0 or k <= 0:
        return obs, k, task_ids, helper_ids, features, finish_values, task_values

    pos = 0
    cbr = env.mac.get_cbr_history_mean()
    tfrac = env.t / env.horizon
    nfrac = env.n / 100.0
    for t in tasks:
        payload = t['payload']; cycles = t['cycles']; deadline = t['deadline']; value = t['value']
        for hid, helper in helpers:
            cpu = helper['cpu']; finish = cycles / cpu
            features[pos] = (payload, cycles / 1e8, deadline, value / 5.0,
                             cpu / 1e9, helper['queue'] / 1e8, nfrac, cbr, tfrac, 1.0)
            task_ids.append(t['id']); helper_ids.append(hid)
            finish_values.append(finish); task_values.append(value)
            pos += 1

    return obs, k, task_ids, helper_ids, features, finish_values, task_values


def score_select(prepared, method, residual=None, task_policy=None):
    obs, k, task_ids, helper_ids, features, finish_values, task_values = prepared
    n = len(task_ids)
    if n == 0 or k <= 0:
        return [], n, []
    x = torch.from_numpy(features)
    finish = torch.tensor(finish_values, dtype=torch.float32)
    value = torch.tensor(task_values, dtype=torch.float32)
    with torch.no_grad():
        analytic = value / (finish + 1e-6)
        if method == 'value_aware':
            scores = analytic
        elif method == 'minfinish':
            scores = -finish
        elif method in ('shielded_residual_rl', 'ungated_residual_rl'):
            scores = analytic + residual.residual(x)
            mf = int(torch.argmin(finish)); top = int(torch.argmax(scores))
            if method == 'shielded_residual_rl' and float((scores[top] - analytic[mf]).detach()) <= .05:
                scores = analytic
        elif method == 'task_rl':
            scores = task_policy.score(x)
        elif method == 'edf':
            scores = -torch.tensor([features[i, 2] for i in range(n)], dtype=torch.float32)
        elif method == 'random':
            scores = torch.arange(n, dtype=torch.float32).flip(0)
        elif method == 'shield_greedy':
            scores = analytic
        else:
            raise ValueError(method)
        order = torch.argsort(scores, descending=True).tolist()
    pairs, used = [], set()
    for ix in order:
        tid = task_ids[ix]
        if tid not in used:
            pairs.append((tid, helper_ids[ix])); used.add(tid)
        if len(pairs) >= k:
            break
    return pairs, n, order


def select(env, method, residual=None, task_policy=None):
    return score_select(prepare(env), method, residual, task_policy)[:2]
