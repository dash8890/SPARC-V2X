"""Recurrent service-continuity actor-critic.

The controlled object is an already-active service.  The policy chooses
retain or a handoff operation for each active service, then scores feasible
target hosts from causal residence, contact, resource, and radio state.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from services import LifecycleCommand


@dataclass
class ContinuityAction:
    operation: str
    service: str
    source: str
    target: str | None = None
    target_index: int | None = None


class ServiceContinuityPolicy(nn.Module):
    def __init__(self, feature_dim=12, hidden=128):
        super().__init__()
        self.service_encoder = nn.Sequential(nn.Linear(feature_dim, hidden),
                                             nn.LayerNorm(hidden), nn.Tanh())
        self.target_encoder = nn.Sequential(nn.Linear(8, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.operation_head = nn.Linear(hidden, 3)  # retain, migrate, replicate
        self.target_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(),
                                         nn.Linear(hidden, 1))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    @staticmethod
    def service_features(env, service, source):
        spec = env.registry.specs[service]
        inst = env.registry.instance(service, source)
        residual = max(0.0, 10.0 - inst.residence_s) / 10.0
        return torch.tensor([
            spec.transfer_bytes / (400 << 10),
            spec.deadline_s,
            float(inst.queue_length) / 10.0,
            residual,
            env.mac.get_cbr_history_mean(),
            float(inst.state == "warm"),
            float(env.pending_packets.__len__()) / 100.0,
            float(env.registry.memory_used(source)) / max(env.registry.host_memory[source], 1),
            0.0, 0.0, 0.0, 0.0,
        ], dtype=torch.float32)

    @staticmethod
    def target_features(env, service, source, target):
        spec = env.registry.specs[service]
        free = (env.registry.host_memory[target] - env.registry.memory_used(target)) / max(env.registry.host_memory[target], 1)
        contact = float(env.world.contact_at(source, target, env.time_s))
        residence = max(0.0, 10.0 - env.registry.instance(service, source).residence_s) / 10.0
        deliverable = (contact * env.mac.slot_capacity_bits / 8.0 *
                       env.mac.predicted_success_probability(env.mac.slot_capacity_bits))
        return torch.tensor([
            free, contact, residence, min(deliverable / max(spec.transfer_bytes, 1), 2.0),
            env.mac.get_cbr_history_mean(), float(env.registry.can_place(service, target)),
            float(env.helpers[target].estimate_completion_time(spec.cycles_per_request)) / 1.0,
            float(env.registry.instance(service, target).state == "warm"),
        ], dtype=torch.float32)

    def score(self, env, service, source, hidden=None):
        sf = self.service_features(env, service, source).unsqueeze(0).unsqueeze(0)
        x, hidden = self.rnn(self.service_encoder(sf), hidden)
        context = x[:, -1]
        ops = self.operation_head(context)
        targets = [h for h in env.helpers if h != source and
                   env.registry.can_place(service, h)]
        if targets:
            tf = torch.stack([self.target_features(env, service, source, h) for h in targets])
            logits = self.target_head(torch.cat([context.expand(len(targets), -1),
                                                 self.target_encoder(tf)], dim=1)).squeeze(-1)
        else:
            logits = torch.empty(0)
        return ops.squeeze(0), logits, targets, hidden

    def act(self, env, service, source, deterministic=True, hidden=None):
        ops, target_logits, targets, hidden = self.score(env, service, source, hidden)
        op_mask = torch.tensor([True, bool(targets), bool(targets)])
        masked = ops.masked_fill(~op_mask, -1e9)
        op_dist = torch.distributions.Categorical(logits=masked)
        op_idx = int(torch.argmax(masked) if deterministic else op_dist.sample())
        operation = ("retain", "migrate", "replicate")[op_idx]
        target = None; target_index = None
        target_dist = None
        if operation != "retain":
            target_dist = torch.distributions.Categorical(logits=target_logits)
            target_index = int(torch.argmax(target_logits) if deterministic else target_dist.sample())
            target = targets[target_index]
        command = [] if operation == "retain" else [LifecycleCommand(operation, service, target, source)]
        return command, ContinuityAction(operation, service, source, target, target_index), op_dist, target_dist, hidden

    def value(self, env, service, source, hidden=None):
        sf = self.service_features(env, service, source).unsqueeze(0).unsqueeze(0)
        x, hidden = self.rnn(self.service_encoder(sf), hidden)
        return self.critic(x[:, -1]).squeeze(-1), hidden
