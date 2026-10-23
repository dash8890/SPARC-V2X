"""Train the residual policy through the isolated WiLab file bridge."""
from __future__ import annotations
import argparse, json, shutil, subprocess, time, sys, os, math, re
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from task_layer import TaskLayer
from metrics_logger import enrich_audit_epoch, write_episode_metrics
from wilab_closed_loop_protocol import atomic_json_write, wait_json
from controllers.rl.residual_task_policy import ResidualTaskPolicy
from controllers.rl.safe_admission import conservative_safe_mask
from controllers.rl.safety_governor import SafetyGovernor

def state_vector(state, epoch=0, controller_epochs=100, budget=250000):
    return torch.tensor([
        float(state.get('cbr') or 0.0), float(state.get('prr') or 0.0),
        float(state.get('p95_delay_ms') or 0.0)/100.0,
        float(state.get('wilab_pending_packets') or 0.0)/100000.0,
        float(state.get('safety_rx_cumulative') or 0.0)/100000.0,
        float(state.get('safety_error_cumulative') or 0.0)/10000.0,
        float(epoch)/max(1.0,float(controller_epochs)), float(budget)/1e6,
        float(bool(state.get('cbr_valid',False))), float(bool(state.get('delay_valid',False)))], dtype=torch.float32)


def sample_action(state, layer, policy, safety_only=False, forced_budget=None, all_eligible=False,
                  forced_byte_budget=None, policy_mode="residual", deterministic=False,
                  unique_sources=False, positive_admission_allowed=True, max_candidates=None):
    candidates = layer.candidates(float(state.get("cbr", 0.0)), 0.0)
    if max_candidates is not None and int(max_candidates) > 0 and len(candidates) > int(max_candidates):
        # Keep highest analytic-score candidates (stable cost/size sweep).
        candidates = sorted(candidates, key=lambda c: float(c.get("analytic_score") or 0.0), reverse=True)[
            : int(max_candidates)
        ]
    # Use the canonical SPARC shield implementation.  WiLab startup epochs
    # may not yet have valid sensing/delay measurements; withhold learned
    # computation admission deterministically until both are valid.
    cbr_valid = bool(state.get("cbr_valid", False))
    delay_valid = bool(state.get("delay_valid", False))
    byte_budget = None if forced_byte_budget is None else max(0, int(forced_byte_budget))
    baseline_modes = {
        "value_aware", "shield_greedy", "minfinish", "edf", "random", "cbr_rule", "rh_mpc",
        "greedy_vb", "cbr_threshold", "myopic_dual", "ce_milp",
    }
    if policy_mode in ("cbr_rule", "rh_mpc") and cbr_valid and delay_valid:
        cbr = float(state.get("cbr", 0.0) or 0.0)
        prr = float(state.get("prr", 0.0) or 0.0)
        p95 = float(state.get("p95_delay_ms", 0.0) or 0.0)
        if policy_mode == "cbr_rule":
            budget = max(0, min(5, int((1.0 - cbr) * 5.0)))
        else:
            feasible = [a for a in range(5) if prr - 0.006 * a * 3.5 >= 0.95 and p95 + 5.0 * a * 3.5 <= 50.0]
            budget = (max(feasible) + 1) if feasible else 0
        safe_actions = list(range(budget + 1))
    elif all_eligible or byte_budget is not None:
        # Calibration-only mode: remove the admission cap so the workload,
        # rather than the action cardinality, determines offered radio load.
        budget = len(candidates)
        safe_actions = list(range(budget + 1))
    elif forced_budget is not None:
        safe_actions = list(range(int(forced_budget) + 1))
        budget = int(forced_budget)
    elif not (cbr_valid and delay_valid):
        safe_actions = []
        budget = 0
    else:
        obs = [float(state.get("cbr", 0.0)),
               float(state.get("density", 0.0))/100.0,
               float(state.get("prr", 0.0)),
               float(state.get("p95_delay_ms", 0.0))/100.0]
        mask = conservative_safe_mask(obs)
        safe_actions = [i for i, ok in enumerate(mask.tolist()) if ok]
        # Action 0 is the no-computation fallback.  It is retained in the
        # logged safe set, but it cannot authorize a computation assignment.
        budget = 0 if safe_actions == [0] else len(safe_actions)
    if safety_only or not positive_admission_allowed:
        return {"budget": 0, "K_t": len(safe_actions),
                "safe_action_set": safe_actions, "assignments": [],
                "allowed_sources": [], "n_candidates": len(candidates)}, torch.tensor(0.0)
    if not candidates or budget <= 0:
        return {"budget": 0, "K_t": len(safe_actions),
                "safe_action_set": safe_actions, "assignments": [],
                "allowed_sources": [], "n_candidates": len(candidates)}, torch.tensor(0.0)
    # HQ-SPO suite baselines 1–4.
    if policy_mode in ("greedy_vb", "cbr_threshold", "myopic_dual", "ce_milp") or getattr(policy, "is_myopic_dual", False):
        from controllers.rl.hq_spo_baselines import build_baseline_pack, assignments_to_action_fields
        mode = "myopic_dual" if getattr(policy, "is_myopic_dual", False) else policy_mode
        b_cap = int(byte_budget) if byte_budget is not None else (int(policy.get_b_cap(int(getattr(layer, "current_epoch", 0)))) if hasattr(policy, "get_b_cap") else 175000)
        lam = float(getattr(policy, "lam", 0.0) or 0.0)
        ce_w = getattr(policy, "ce_weights", None)
        th_lo = float(getattr(policy, "th_lo", 0.3) or 0.3)
        th_hi = float(getattr(policy, "th_hi", 0.8) or 0.8)
        th_resel = float(getattr(policy, "th_resel", 0.5) or 0.5)
        pack, meta = build_baseline_pack(
            candidates, state, mode=mode, b_max=b_cap, unique_sources=unique_sources,
            th_lo=th_lo, th_hi=th_hi, th_resel=th_resel,
            lam=lam, ce_weights=ce_w,
        )
        action = assignments_to_action_fields(pack, layer, budget, safe_actions)
        action["n_candidates"] = int(len(candidates))
        action["policy_mode"] = mode
        action["hq_spo_baseline"] = meta
        return action, torch.tensor(0.0)
    # Structured Priority & Opportunity-Cost Controller (SPARC-V2X Redesign)
    if policy_mode == "structured_priority" or getattr(policy, "is_structured_priority_policy", False):
        from controllers.rl.hq_spo_baselines import assignments_to_action_fields
        from controllers.rl.structured_priority_policy import score_and_match
        b_cap = int(byte_budget) if byte_budget is not None else 175000
        if hasattr(policy, "act_numpy"):
            try:
                ret = policy.act_numpy(state, layer=layer, deterministic=deterministic)
            except TypeError:
                ret = policy.act_numpy(state, deterministic=deterministic)
            weights, beta, logp = ret[0], ret[1], ret[2]
            z_w = ret[3] if len(ret) > 3 else [0.0, 0.0, 0.0, 0.0]
            z_b = ret[4] if len(ret) > 4 else 0.0
            sp_obs = ret[5] if len(ret) > 5 else None
        else:
            weights = getattr(policy, "weights", [2.5, 1.0, 1.0, 0.5])
            beta = float(getattr(policy, "beta", 0.95))
            logp = torch.tensor(0.0)
            z_w = [0.0, 0.0, 0.0, 0.0]
            z_b = 0.0
            sp_obs = None
        score_match_fn = getattr(policy, "score_and_match_fn", score_and_match)
        try:
            pack, meta = score_match_fn(candidates, weights=weights, beta_throttle=beta, b_max=b_cap, unique_sources=unique_sources, layer=layer)
        except TypeError:
            pack, meta = score_match_fn(candidates, weights=weights, beta_throttle=beta, b_max=b_cap, unique_sources=unique_sources)
        action = assignments_to_action_fields(pack, layer, budget, safe_actions)
        action["n_candidates"] = int(len(candidates))
        action["policy_mode"] = "structured_priority"
        action["structured_priority_meta"] = meta
        action["weights"] = list(weights)
        action["beta_throttle"] = float(beta)
        action["z_w"] = z_w
        action["z_b"] = z_b
        action["sp_obs"] = sp_obs
        if sp_obs is not None:
            action["sp_value"] = float(sp_obs.get("value", 0.0))
            action["sp_cost_value"] = float(sp_obs.get("cost_value", 0.0))
        if not torch.is_tensor(logp):
            logp = torch.tensor(float(logp), dtype=torch.float32)
        return action, logp
    # PB-CRL / B5′: shared price-and-budget packer (λ, β_exec).
    if policy_mode in ("price_budget", "b5_prime") or getattr(policy, "is_pb_crl_b5_prime", False):
        from controllers.rl.hq_spo_baselines import assignments_to_action_fields
        from controllers.rl.pb_crl_packer import price_budget_pack, pack_meta
        from controllers.rl.pb_crl_b5_prime import state_vec_from_wilab
        b_cap = int(byte_budget) if byte_budget is not None else 175000
        svec = state_vec_from_wilab(state, layer)
        if getattr(policy, "is_pb_crl_b5_prime", False) or getattr(policy, "is_pb_crl_policy", False):
            try:
                ret = policy.act_numpy(svec, layer=layer, deterministic=deterministic)
            except TypeError:
                ret = policy.act_numpy(svec, deterministic=deterministic)
            if len(ret) == 5 and isinstance(ret[4], dict):
                lam, beta, logp, pb_z, obs_dict = ret
            elif len(ret) == 5:
                lam, beta, logp, z_l, z_b = ret
                pb_z = [float(z_l), float(z_b)]
                obs_dict = None
            else:
                lam, beta, logp = ret[0], ret[1], ret[2]
                pb_z = list(ret[3])
                obs_dict = None
        else:
            lam = float(getattr(policy, "lam", 0.0) or 0.0)
            beta = float(getattr(policy, "beta", 1.0) or 1.0)
            logp = torch.tensor(0.0)
            pb_z = [0.0, 0.0]
            obs_dict = None
        pack = price_budget_pack(
            candidates, lam=lam, beta_exec=beta, b_max=b_cap, unique_sources=unique_sources,
        )
        meta = pack_meta(pack, lam=lam, beta_exec=beta, b_max=b_cap)
        action = assignments_to_action_fields(pack, layer, budget, safe_actions)
        action["n_candidates"] = int(len(candidates))
        action["policy_mode"] = "price_budget"
        action["hq_spo_baseline"] = meta
        g_action = float(torch.sigmoid(torch.tensor(pb_z[2])) * 1.5 + 0.5) if len(pb_z) >= 3 else 1.0
        action["pb_crl"] = {"lam": lam, "beta_prop": beta, "beta_exec": beta, "g": g_action}
        action["pb_state"] = list(svec)
        action["pb_z"] = pb_z
        if obs_dict is not None:
            action["pb_full_obs"] = obs_dict
        if not torch.is_tensor(logp):
            logp = torch.tensor(float(logp), dtype=torch.float32)
        return action, logp
    # Pack-swap gate (tps4 Search-RPI): LVM vs Direct by composition rule.
    if getattr(policy, "is_pack_swap_gate", False):
        epoch = int(getattr(layer, "current_epoch", 0))
        ce = int(getattr(layer, "controller_epochs", 100) or 100)
        action_l, lp_l = sample_action(
            state, layer, policy.lvm,
            safety_only=safety_only, forced_budget=forced_budget, all_eligible=all_eligible,
            forced_byte_budget=forced_byte_budget, policy_mode="residual",
            deterministic=deterministic, unique_sources=unique_sources,
            positive_admission_allowed=positive_admission_allowed,
            max_candidates=max_candidates,
        )
        action_d, lp_d = sample_action(
            state, layer, policy.direct,
            safety_only=safety_only, forced_budget=forced_budget, all_eligible=all_eligible,
            forced_byte_budget=forced_byte_budget, policy_mode="residual",
            deterministic=deterministic, unique_sources=unique_sources,
            positive_admission_allowed=positive_admission_allowed,
            max_candidates=max_candidates,
        )
        pack_l = list(action_l.get("assignments") or [])
        pack_d = list(action_d.get("assignments") or [])
        choice, scalars, reason = policy.choose(
            pack_l, pack_d, epoch=epoch, controller_epochs=ce,
        )
        action = dict(action_d if choice == "Direct" else action_l)
        action["policy_mode"] = "pack_swap_gate"
        action["pack_swap_gate"] = {"choice": choice, "reason": reason, **scalars}
        logp_sum = lp_d if choice == "Direct" else lp_l
        return action, logp_sum
    # HQ-SPO gate: LVM / Direct / helper-spread by queue-aware one-step score.
    if getattr(policy, "is_hq_spo_gate", False):
        from controllers.rl.lvm_rpi.hq_spo_gate import build_spread_pack
        action_l, lp_l = sample_action(
            state, layer, policy.lvm,
            safety_only=safety_only, forced_budget=forced_budget, all_eligible=all_eligible,
            forced_byte_budget=forced_byte_budget, policy_mode="residual",
            deterministic=deterministic, unique_sources=unique_sources,
            positive_admission_allowed=positive_admission_allowed,
            max_candidates=max_candidates,
        )
        action_d, lp_d = sample_action(
            state, layer, policy.direct,
            safety_only=safety_only, forced_budget=forced_budget, all_eligible=all_eligible,
            forced_byte_budget=forced_byte_budget, policy_mode="residual",
            deterministic=deterministic, unique_sources=unique_sources,
            positive_admission_allowed=positive_admission_allowed,
            max_candidates=max_candidates,
        )
        pack_l = list(action_l.get("assignments") or [])
        pack_d = list(action_d.get("assignments") or [])
        env = int(forced_byte_budget) if forced_byte_budget is not None else int(getattr(policy, "envelope", 175000))
        pack_s = build_spread_pack(candidates, env, unique_sources=unique_sources)
        choice, scalars, reason = policy.choose(pack_l, pack_d, pack_s, candidates)
        if choice == "Direct":
            action = dict(action_d)
            logp_sum = lp_d
        elif choice == "Spread":
            # Rebuild action shell from LVM, replace assignments with spread pack.
            action = dict(action_l)
            packet_counts = {}
            payload_bytes = {}
            for a in pack_s:
                sid = int(a["source_id"])
                packet_counts[sid] = packet_counts.get(sid, 0) + int(a["packet_count"])
                payload_bytes[sid] = payload_bytes.get(sid, 0) + int(a["payload_bytes"])
            action["assignments"] = pack_s
            action["allowed_sources"] = [int(a["source_id"]) for a in pack_s]
            action["source_packet_ids"] = list(packet_counts.keys())
            action["source_packet_counts"] = list(packet_counts.values())
            action["source_payload_bytes"] = list(payload_bytes.values())
            action["computation_packets_offered"] = int(sum(packet_counts.values()))
            action["computation_bytes_offered"] = int(sum(payload_bytes.values()))
            logp_sum = lp_l
        else:
            action = dict(action_l)
            logp_sum = lp_l
        action["policy_mode"] = "hq_spo_gate"
        action["hq_spo_gate"] = {"choice": choice, "reason": reason, **scalars}
        return action, logp_sum
    # Pressure-gated Direct/POPL/LVM: choose pack from frozen generators by causal regime.
    if getattr(policy, "is_pressure_gate", False):
        choice, scalars, reason = policy.choose(candidates)
        gate_meta = {"choice": choice, "reason": reason, **scalars}
        if choice == "POPL":
            epoch = int(getattr(layer, "current_epoch", 0))
            byte_cap = int(byte_budget) if byte_budget is not None else 175000
            pack, logp_sum, _, _, _ = policy.popl.sample_pack(
                state,
                candidates,
                epoch=epoch,
                byte_cap=byte_cap,
                controller_epochs=int(getattr(layer, "controller_epochs", 100) or 100),
                deterministic=deterministic,
            )
            assignments = list(pack)
            packet_counts = {}
            payload_bytes = {}
            for a in assignments:
                sid = int(a["source_id"])
                packet_counts[sid] = packet_counts.get(sid, 0) + int(a["packet_count"])
                payload_bytes[sid] = payload_bytes.get(sid, 0) + int(a["payload_bytes"])
            action = {
                "budget": budget,
                "K_t": len(safe_actions),
                "safe_action_set": safe_actions,
                "assignments": assignments,
                "policy_mode": "pressure_gate",
                "allowed_sources": [int(a["source_id"]) for a in assignments],
                "source_packet_ids": list(packet_counts.keys()),
                "source_packet_counts": list(packet_counts.values()),
                "source_payload_bytes": list(payload_bytes.values()),
                "computation_packet_bytes": int(layer.computation_packet_bytes),
                "computation_packets_offered": int(sum(packet_counts.values())),
                "computation_bytes_offered": int(sum(payload_bytes.values())),
                "n_candidates": int(len(candidates)),
                "sequential_context": False,
                "popl_pack": True,
                "pressure_gate": gate_meta,
            }
            return action, logp_sum
        residual_pol = policy.policy_for(choice)
        action, logp_sum = sample_action(
            state, layer, residual_pol,
            safety_only=safety_only, forced_budget=forced_budget, all_eligible=all_eligible,
            forced_byte_budget=forced_byte_budget, policy_mode="residual",
            deterministic=deterministic, unique_sources=unique_sources,
            positive_admission_allowed=positive_admission_allowed,
            max_candidates=max_candidates,
        )
        action = dict(action)
        action["policy_mode"] = "pressure_gate"
        action["pressure_gate"] = gate_meta
        return action, logp_sum
    # POPL sequential pack policy: construct the full assignment pack at every epoch.
    # Lean action payload (no per-candidate audit dumps) — scheduling fields unchanged.
    if getattr(policy, "is_popl_pack_policy", False) and policy_mode in ("residual", "popl"):
        epoch = int(getattr(layer, "current_epoch", 0))
        byte_cap = int(byte_budget) if byte_budget is not None else 175000
        pack, logp_sum, _, _, _ = policy.sample_pack(
            state,
            candidates,
            epoch=epoch,
            byte_cap=byte_cap,
            controller_epochs=int(getattr(layer, "controller_epochs", 100) or 100),
            deterministic=deterministic,
        )
        assignments = list(pack)
        packet_counts = {}
        payload_bytes = {}
        for a in assignments:
            sid = int(a["source_id"])
            packet_counts[sid] = packet_counts.get(sid, 0) + int(a["packet_count"])
            payload_bytes[sid] = payload_bytes.get(sid, 0) + int(a["payload_bytes"])
        action = {
            "budget": budget,
            "K_t": len(safe_actions),
            "safe_action_set": safe_actions,
            "assignments": assignments,
            "policy_mode": "popl",
            "allowed_sources": [int(a["source_id"]) for a in assignments],
            "source_packet_ids": list(packet_counts.keys()),
            "source_packet_counts": list(packet_counts.values()),
            "source_payload_bytes": list(payload_bytes.values()),
            "computation_packet_bytes": int(layer.computation_packet_bytes),
            "computation_packets_offered": int(sum(packet_counts.values())),
            "computation_bytes_offered": int(sum(payload_bytes.values())),
            "n_candidates": int(len(candidates)),
            "sequential_context": False,
            "popl_pack": True,
        }
        return action, logp_sum
    x = torch.tensor([c["features"] for c in candidates], dtype=torch.float32)
    analytic = torch.tensor([c["analytic_score"] for c in candidates], dtype=torch.float32)
    use_ctx = bool(getattr(policy, "uses_sequential_context", False)) and policy_mode not in (
        "anchor", "value_aware", "shield_greedy", "minfinish", "edf", "itora_inspired", "random",
        "cbr_rule", "rh_mpc",
    )
    selected_indices: list[int] = []
    assignments = []
    if use_ctx:
        from controllers.rl.context_direct_policy import sequential_select
        feat0 = candidates[0].get("features") or []
        epoch_frac = float(feat0[8]) if len(feat0) > 8 else 0.0
        cbr = float(state.get("cbr", 0.0) or 0.0)
        selected_indices, total_lp = sequential_select(
            policy,
            candidates,
            cbr=cbr,
            byte_budget=byte_budget,
            epoch_frac=epoch_frac,
            max_picks=min(budget, len(candidates)),
            unique_sources=unique_sources,
            deterministic=deterministic,
        )
        for ix in selected_indices:
            c = candidates[ix]
            assignments.append({
                "task_id": c["task_id"], "helper_id": c["helper_id"],
                "source_id": c["source_id"],
                "payload_bytes": c["payload_bytes"],
                "packet_count": c["packet_count"],
            })
        logp_sum = total_lp
    else:
        if policy_mode in ("anchor", "value_aware", "shield_greedy"):
            logits = analytic
        elif policy_mode == "minfinish":
            logits = -torch.tensor([float(c.get("finish_ms", 1e9)) for c in candidates], dtype=torch.float32)
        elif policy_mode == "edf":
            logits = -torch.tensor([float(c.get("deadline_ms", 1e9)) for c in candidates], dtype=torch.float32)
        elif policy_mode == "itora_inspired":
            # VLRO (Value-Latency-Resource Offloading); code key kept for artifact compatibility.
            # See protocol/baseline_nomenclature.md and b5_itora_mapping_audit.md
            scores = []
            for c in candidates:
                finish = max(float(c.get("finish_ms", 1e9)), 1e-6)
                payload = max(float(c.get("payload_bytes", 1.0)), 1.0)
                value = float(c.get("task_value", c.get("analytic_score", 0.0) * finish))
                # value / (finish × payload) — value, latency, and Mode-2 byte resource
                scores.append(value / (finish * (payload / 1e4)))
            logits = torch.tensor(scores, dtype=torch.float32)
        elif policy_mode == "random":
            logits = torch.rand(len(candidates))
        elif hasattr(policy, "score_logits"):
            logits = policy.score_logits(x, analytic)
        else:
            logits = analytic + policy.residual(x)
        remaining = list(range(len(candidates))); used = set(); used_sources = set(); logp = []
        used_bytes = 0
        for _ in range(min(budget, len(remaining))):
            choices = [i for i in remaining
                       if candidates[i]["task_id"] not in used
                       and (not unique_sources or candidates[i]["source_id"] not in used_sources)
                       and (byte_budget is None or used_bytes + int(candidates[i]["payload_bytes"]) <= byte_budget)]
            if not choices: break
            dist = torch.distributions.Categorical(logits=logits[choices])
            if deterministic:
                local = torch.argmax(logits[choices])
            else:
                local = dist.sample()
            ix = choices[int(local)]
            logp.append(dist.log_prob(local)); remaining.remove(ix)
            c = candidates[ix]; used.add(c["task_id"]); used_sources.add(c["source_id"])
            used_bytes += int(c["payload_bytes"])
            selected_indices.append(ix)
            assignments.append({"task_id": c["task_id"], "helper_id": c["helper_id"],
                                "source_id": c["source_id"],
                                "payload_bytes": c["payload_bytes"],
                                "packet_count": c["packet_count"]})
        logp_sum = torch.stack(logp).sum() if logp else torch.tensor(0.0)
    sources = {c["task_id"]: c["source_id"] for c in candidates}
    packet_counts = {}
    payload_bytes = {}
    for a in assignments:
        sid = int(a["source_id"])
        packet_counts[sid] = packet_counts.get(sid, 0) + int(a["packet_count"])
        payload_bytes[sid] = payload_bytes.get(sid, 0) + int(a["payload_bytes"])
    anchor_first = max(candidates, key=lambda c: float(c.get("analytic_score", 0.0))) if candidates else None
    action = {"budget": budget, "K_t": len(safe_actions),
              "safe_action_set": safe_actions, "assignments": assignments,
              "policy_mode": policy_mode,
              "anchor_first": None if anchor_first is None else [anchor_first["task_id"], anchor_first["helper_id"]],
              "residual_changed_anchor": bool(assignments and anchor_first is not None and [assignments[0]["task_id"], assignments[0]["helper_id"]] != [anchor_first["task_id"], anchor_first["helper_id"]]),
              "allowed_sources": [sources[a["task_id"]] for a in assignments],
              "source_packet_ids": list(packet_counts.keys()),
              "source_packet_counts": list(packet_counts.values()),
              "source_payload_bytes": list(payload_bytes.values()),
              "computation_packet_bytes": int(layer.computation_packet_bytes),
              "computation_packets_offered": int(sum(packet_counts.values())),
              "computation_bytes_offered": int(sum(payload_bytes.values()))}
    action["selected_candidate_indices"] = [int(i) for i in selected_indices]
    action["candidate_features"] = x.detach().cpu().tolist()
    action["candidate_analytic_scores"] = analytic.detach().cpu().tolist()
    action["candidate_task_ids"] = [int(c["task_id"]) for c in candidates]
    action["candidate_source_ids"] = [int(c["source_id"]) for c in candidates]
    action["candidate_payload_bytes"] = [int(c.get("payload_bytes", 0)) for c in candidates]
    action["candidate_helper_ids"] = [int(c.get("helper_id", -1)) for c in candidates]
    action["candidate_finish_ms"] = [float(c.get("finish_ms", 0.0)) for c in candidates]
    action["candidate_deadline_ms"] = [float(c.get("deadline_ms", 0.0)) for c in candidates]
    action["candidate_task_values"] = [float(c.get("task_value", 0.0)) for c in candidates]
    action["candidate_cycles"] = [
        float((c.get("features") or [0, 0])[1]) * 1e8 for c in candidates
    ]
    action["n_candidates"] = int(len(candidates))
    action["sequential_context"] = bool(use_ctx)
    return action, logp_sum


def run_episode(root, bridge, episode, density, policy, timeout, external_matlab=False, safety_only=False, generation_ms=10.0, allocation_ms=10.0, sim_seed=0, controller_epochs=100, forced_budget=None, tasks_per_source=3, all_eligible=False, forced_byte_budget=None, safety_payload_bytes=1000, simulation_time=20.0, policy_mode="residual", branch_epoch=None, deterministic=False, replay_mode="off", replay_file="", return_trace=False, task_seed=None, governor=None, value_envelope_bytes=250000, spread_task_arrivals=False, unique_sources=False, deadline_multiplier=1.0, max_candidates=None, helper_swap=None, force_pack=None, helper_cpu_lo=2e8, helper_cpu_hi=1e9, cluster_share=None, output_root=None, arrival_mode="legacy"):
    # helper_swap (optional diagnostic): {"epoch", "task_id", "from_helper", "to_helper"}
    # force_pack (optional diagnostic): {"epoch", "assignments": [...]} — replace proposal then re-apply governor
    # Applied after the governor on the executed assignment list only. Does not
    # change the policy, admission envelope, or which tasks are admitted.
    # Use the extended trace for the post-warm-up WiLab horizon.  The legacy
    # 10 s traces are retained for provenance and are never overwritten.
    trace = root / "traces" / "mode2_external_validation" / f"synthetic_highway_rho_{density}_20s.txt"
    out_root = Path(output_root) if output_root is not None else Path(os.environ.get("WILAB_OUTPUT_ROOT", str(root / "results" / "additional_validation_20261007" / "wilab_closed_loop")))
    out = out_root / f"episode_{episode:04d}"
    out.mkdir(parents=True, exist_ok=True)
    # Materialize a run-local configuration so the simulator's scalar rho
    # agrees with the requested trace density.  The shared template contains
    # a legacy rho=35.5 default; leaving it unchanged made density campaigns
    # silently use mixed configuration/trace semantics.
    cfg_template = root / "vendor" / "WiLabV2Xsim" / "ConfigFiles" / "test_trace.cfg"
    cfg = out / "test_trace_runtime.cfg"
    cfg_text = cfg_template.read_text(encoding="utf-8")
    cfg_text = re.sub(r"(?m)^(\s*\[rho\]\s+)[^\r\n]+$", rf"\g<1>{float(density):g}", cfg_text)
    cfg.write_text(cfg_text, encoding="utf-8")
    layer = TaskLayer(density, (90000 + episode) if task_seed is None else int(task_seed),
                      tasks_per_source=tasks_per_source,
                      spread_arrivals=spread_task_arrivals,
                      controller_epochs=controller_epochs,
                      deadline_multiplier=deadline_multiplier,
                      helper_cpu_lo=helper_cpu_lo,
                      helper_cpu_hi=helper_cpu_hi,
                      cluster_share=cluster_share,
                      arrival_mode=arrival_mode)
    offered_value = sum(float(t.value) for t in layer.tasks)
    bridge.mkdir(parents=True, exist_ok=True)
    for p in bridge.glob("*.json"): p.unlink()
    matlab = (root / "experiments" / "wilab_closed_loop" / "run_wilab_closed_loop_episode.m").as_posix()
    hook_dir = (root / "experiments" / "wilab_closed_loop").as_posix()
    cmd = f"cd('{root.as_posix()}'); addpath('{hook_dir}','-begin'); run_wilab_closed_loop_episode('{cfg.as_posix()}','{trace.as_posix()}','{out.as_posix()}','{bridge.as_posix()}',{episode},0,{generation_ms/1000.0},{allocation_ms/1000.0},{sim_seed},{controller_epochs},{int(safety_payload_bytes)},{float(simulation_time)},'{replay_mode}','{replay_file}')"
    log_path = out / "matlab.log"
    log_handle = log_path.open("w", encoding="utf-8")
    # MATLAB is launched from its own installation directory.  Starting it
    # with the worktree as cwd can trigger MATLAB's "File system
    # inconsistency" startup failure before user code runs.
    matlab_bin = Path("C:/Program Files/MATLAB/R2026a/bin")
    proc = None if external_matlab else subprocess.Popen(
        [str(matlab_bin / "matlab.exe"), "-batch", cmd],
        cwd=str(matlab_bin), stdout=log_handle, stderr=subprocess.STDOUT)
    logp=[]; total=0.0; audit=[]; trace=[]
    prev_n_reassign = 0.0
    try:
        epoch = 0
        while True:
            if controller_epochs is not None and epoch >= int(controller_epochs):
                break
            state_path = bridge / f"state_{episode:06d}_{epoch:06d}.json"
            deadline = time.monotonic() + min(timeout, 600.0)
            while not state_path.exists():
                if proc is not None and proc.poll() is not None:
                    if proc.returncode != 0:
                        log_handle.flush(); log_handle.close()
                        raise RuntimeError(f"MATLAB exited before state {epoch} (code {proc.returncode}); see {log_path}:\n{log_path.read_text(encoding='utf-8', errors='replace')[-4000:]}")
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for {state_path}")
                time.sleep(0.01)
            if not state_path.exists():
                break
            while True:
                try:
                    state = json.loads(state_path.read_text(encoding='utf-8'))
                    break
                except (OSError, json.JSONDecodeError):
                    time.sleep(0.001)

            n_reassign_cur = float(state.get("n_reassign", 0.0) or 0.0)
            if epoch == 0:
                prev_n_reassign = n_reassign_cur
                delta_reassign = 0.0
            else:
                delta_reassign = max(0.0, n_reassign_cur - prev_n_reassign)
                prev_n_reassign = n_reassign_cur
            dt_step = float(allocation_ms) / 1000.0
            n_veh = max(1, int(state.get("density", density) or density))
            resel_rate = (delta_reassign / max(1e-6, dt_step)) / n_veh
            state["resel_t"] = resel_rate
            state["reselection_rate"] = resel_rate

            active_mode = "anchor" if branch_epoch is not None and epoch < branch_epoch else policy_mode
            before_advance = layer.completed_value()
            layer.set_epoch(epoch)
            advance_reward = layer.completed_value() - before_advance
            admission_allowed = (governor is None or governor.admission_allowed(state))
            gate_blocked = bool(governor is not None and not admission_allowed)
            t_infer0 = time.perf_counter()
            action, lp = sample_action(
                state, layer, policy, safety_only=safety_only,
                forced_budget=forced_budget, all_eligible=all_eligible,
                forced_byte_budget=forced_byte_budget, policy_mode=active_mode,
                deterministic=deterministic, unique_sources=unique_sources,
                positive_admission_allowed=admission_allowed,
                max_candidates=max_candidates)
            inference_ms = (time.perf_counter() - t_infer0) * 1000.0
            action["inference_ms"] = float(inference_ms)
            action["score_select_ms"] = float(inference_ms)  # full sample_action wall; split TBD
            if "n_candidates" not in action:
                action["n_candidates"] = 0
            # Phase-2b: freeze proposed-pack likelihood identity BEFORE governor.
            # PPO ratio / old_logp must refer to A_proposed; A_exec may differ.
            proposed_assignments = list(action.get("assignments", []))
            proposed_indices = [int(i) for i in (action.get("selected_candidate_indices") or [])]
            if not torch.is_tensor(lp):
                lp = torch.tensor(float(lp), dtype=torch.float32)
            else:
                lp = lp.reshape(()).to(dtype=torch.float32)
            action["proposed_assignments"] = proposed_assignments
            action["proposed_selected_candidate_indices"] = list(proposed_indices)
            # Bridge JSON cannot carry torch Tensors; keep float on the wire and
            # re-wrap to tensor when building the PPO trace (below).
            action["proposed_logp"] = float(lp.detach().cpu())
            action["pack_logp_is_proposed"] = True
            # Keep selected_candidate_indices as the proposed indices for PPO even
            # if executed assignments later diverge under the governor.
            action["selected_candidate_indices"] = list(proposed_indices)
            t_gov0 = time.perf_counter()
            if governor is not None and not safety_only:
                candidates_now = layer.candidates(float(state.get("cbr", 0.0)), 0.0)
                executed, intervened, reason = governor.apply(proposed_assignments, candidates_now, state)
                action["assignments"] = executed
                action["executed_assignments"] = list(executed)
                action["governor_intervened"] = bool(intervened) or gate_blocked
                action["governor_reason"] = ("invalid_radio_state" if gate_blocked and reason in ("none", None) else reason)
                if gate_blocked and not proposed_assignments:
                    action["governor_reason"] = action.get("governor_reason") or "admission_blocked"
                action["governor_max_bytes"] = governor.max_bytes
                action["computation_bytes_offered"] = int(sum(int(a.get("payload_bytes", 0)) for a in executed))
                action["computation_packets_offered"] = int(sum(int(a.get("packet_count", 0)) for a in executed))
                packet_counts = {}
                payload_bytes = {}
                for a in executed:
                    sid = int(a.get("source_id", -1))
                    packet_counts[sid] = packet_counts.get(sid, 0) + int(a.get("packet_count", 0))
                    payload_bytes[sid] = payload_bytes.get(sid, 0) + int(a.get("payload_bytes", 0))
                action["allowed_sources"] = [int(a["source_id"]) for a in executed]
                action["source_packet_ids"] = list(packet_counts.keys())
                action["source_packet_counts"] = list(packet_counts.values())
                action["source_payload_bytes"] = list(payload_bytes.values())
                # Never rewrite proposed logp / proposed indices after intervention.
                action["selected_candidate_indices"] = list(proposed_indices)
            elif gate_blocked:
                action["governor_intervened"] = True
                action["governor_reason"] = "admission_blocked"
                action["executed_assignments"] = []
                action["assignments"] = []
                action["allowed_sources"] = []
                action["source_packet_ids"] = []
                action["source_packet_counts"] = []
                action["source_payload_bytes"] = []
                action["computation_bytes_offered"] = 0
                action["computation_packets_offered"] = 0
            else:
                action["executed_assignments"] = list(action.get("assignments") or [])
            action["governor_ms"] = (time.perf_counter() - t_gov0) * 1000.0
            action["admission_gate_blocked"] = gate_blocked
            action["helper_swap_applied"] = False
            action["force_pack_applied"] = False
            if force_pack is not None and int(epoch) == int(force_pack.get("epoch", -1)):
                forced = list(force_pack.get("assignments") or [])
                # Diagnostic override: treat forced pack as the proposal for this epoch.
                action["proposed_assignments"] = forced
                if governor is not None and not safety_only:
                    candidates_now = layer.candidates(float(state.get("cbr", 0.0)), 0.0)
                    executed, intervened, reason = governor.apply(forced, candidates_now, state)
                    action["assignments"] = executed
                    action["governor_intervened"] = bool(intervened) or gate_blocked
                    action["governor_reason"] = reason
                    action["computation_bytes_offered"] = int(sum(int(a.get("payload_bytes", 0)) for a in executed))
                    action["computation_packets_offered"] = int(sum(int(a.get("packet_count", 0)) for a in executed))
                else:
                    action["assignments"] = forced
                # Rebuild allowed_sources / packet maps from forced executed pack
                packet_counts = {}
                payload_bytes = {}
                for a in action.get("assignments", []):
                    sid = int(a.get("source_id", -1))
                    packet_counts[sid] = packet_counts.get(sid, 0) + int(a.get("packet_count", 0))
                    payload_bytes[sid] = payload_bytes.get(sid, 0) + int(a.get("payload_bytes", 0))
                action["allowed_sources"] = [int(a["source_id"]) for a in action.get("assignments", [])]
                action["source_packet_ids"] = list(packet_counts.keys())
                action["source_packet_counts"] = list(packet_counts.values())
                action["source_payload_bytes"] = list(payload_bytes.values())
                action["force_pack_applied"] = True
                action["force_pack_spec"] = {
                    "epoch": int(force_pack["epoch"]),
                    "n_forced": len(forced),
                    "n_executed": len(action.get("assignments") or []),
                }
            if helper_swap is not None and int(epoch) == int(helper_swap.get("epoch", -1)):
                tid = int(helper_swap["task_id"])
                from_h = int(helper_swap["from_helper"])
                to_h = int(helper_swap["to_helper"])
                swapped = False
                for a in action.get("assignments", []):
                    if int(a.get("task_id", -1)) == tid and int(a.get("helper_id", -1)) == from_h:
                        a["helper_id"] = to_h
                        swapped = True
                        break
                action["helper_swap_applied"] = bool(swapped)
                action["helper_swap_spec"] = {
                    "epoch": int(helper_swap["epoch"]),
                    "task_id": tid,
                    "from_helper": from_h,
                    "to_helper": to_h,
                    "applied": bool(swapped),
                }
                # Recompute helper accounting only; radio source/byte totals unchanged.
                helper_packet_counts_tmp = {}
                helper_payload_bytes_tmp = {}
                for assn in action.get("assignments", []):
                    hid = str(assn.get("helper_id", "unknown"))
                    helper_packet_counts_tmp[hid] = helper_packet_counts_tmp.get(hid, 0) + int(assn.get("packet_count", 0))
                    helper_payload_bytes_tmp[hid] = helper_payload_bytes_tmp.get(hid, 0) + int(assn.get("payload_bytes", 0))
                action["_helper_packet_counts_post_swap"] = helper_packet_counts_tmp
                action["_helper_payload_bytes_post_swap"] = helper_payload_bytes_tmp
            obs_vec = state_vector(state, epoch, controller_epochs, value_envelope_bytes if forced_byte_budget is None else forced_byte_budget)
            value = policy.value(obs_vec) if hasattr(policy, 'value') else torch.tensor(0.0)
            cost_value = policy.cost_value(obs_vec) if hasattr(policy, 'cost_value') else torch.tensor(0.0)
            bridge_action = {k: v for k, v in action.items() if k not in ("sp_obs", "pb_full_obs")}
            atomic_json_write(bridge / f"action_{episode:06d}_{epoch:06d}.json", bridge_action)
            result = wait_json(bridge / f"result_{episode:06d}_{epoch:06d}.json", timeout)
            before = layer.completed_value()
            delivered_ids = result.get("delivered_task_ids", [])
            # MATLAB jsonencode emits a scalar rather than a one-element
            # array when exactly one source delivered.  Normalize both forms
            # at the bridge boundary so the task layer always receives an
            # iterable of source/task IDs.
            if isinstance(delivered_ids, (int, float)):
                delivered_ids = [int(delivered_ids)]
            elif delivered_ids is None:
                delivered_ids = []
            delivered_sources = {int(x) for x in delivered_ids}
            # WiLab reports the transmitting source vehicle; the Python task
            # layer accounts completion by task ID.  Map only the admitted
            # assignments whose source actually delivered in this interval.
            delivered_tasks = [a["task_id"] for a in action["assignments"]
                               if int(a.get("source_id", -1)) in delivered_sources]
            # Older action files carry source_id only in the candidate table;
            # reconstruct it from the deterministic task list when absent.
            if not delivered_tasks and delivered_sources:
                source_by_task = {c["task_id"]: c["source_id"]
                                  for c in layer.candidates(float(state.get("cbr", 0.0)), 0.0)}
                delivered_tasks = [a["task_id"] for a in action["assignments"]
                                   if source_by_task.get(a["task_id"]) in delivered_sources]
            layer.apply_radio_outcomes(action["assignments"], delivered_tasks, 10.0)
            reward = advance_reward + layer.completed_value() - before
            helper_packet_counts = {}
            helper_payload_bytes = {}
            for assn in action.get("assignments", []):
                hid = str(assn.get("helper_id", "unknown"))
                helper_packet_counts[hid] = helper_packet_counts.get(hid, 0) + int(assn.get("packet_count", 0))
                helper_payload_bytes[hid] = helper_payload_bytes.get(hid, 0) + int(assn.get("payload_bytes", 0))
            epoch_rec = {
                "prr": state.get("window_prr", state.get("prr")),
                "window_prr": state.get("window_prr", state.get("prr")),
                "cumulative_prr": state.get("cumulative_prr", state.get("prr")),
                "p95_delay_ms": state.get("p95_delay_ms"),
                "cbr": state.get("cbr"),
                "resel_t": state.get("resel_t", 0.0),
                "reselection_rate": state.get("reselection_rate", 0.0),
                "n_reassign_cumulative": state.get("n_reassign", 0),
                "cbr_valid": state.get("cbr_valid", False),
                "delay_valid": state.get("delay_valid", False),
                "safe_action_set": action.get("safe_action_set", []),
                "K_t": action.get("K_t", 0),
                "num_selected": len(action.get("assignments", [])),
                "n_candidates": action.get("n_candidates", 0),
                "inference_ms": action.get("inference_ms"),
                "score_select_ms": action.get("score_select_ms"),
                "governor_ms": action.get("governor_ms"),
                "computation_packets_offered": action.get("computation_packets_offered", 0),
                "computation_bytes_offered": action.get("computation_bytes_offered", 0),
                "computation_packet_bytes": action.get("computation_packet_bytes", layer.computation_packet_bytes),
                "computation_packets_rx_delta": result.get("rx_packet_delta", result.get("delta_packets", 0)),
                "safety_rx_cumulative": result.get("safety_rx_cumulative", None),
                "safety_error_cumulative": result.get("safety_error_cumulative", None),
                "safety_prr_cumulative": result.get("safety_prr_cumulative", None),
                "safety_p95_delay_cumulative_ms": result.get("safety_p95_delay_cumulative_ms", None),
                "safety_delay_hist_cumulative": result.get("safety_delay_hist_cumulative", None),
                "delay_resolution_ms": result.get("delay_resolution_ms", 1.0),
                "prr_vs_distance": result.get("prr_vs_distance", None),
                "wilab_enqueued_packets": result.get("wilab_enqueued_packets", 0),
                "wilab_enqueued_bytes": result.get("wilab_enqueued_bytes", 0),
                "wilab_scheduled_attempts": result.get("wilab_scheduled_attempts", 0),
                "wilab_pending_packets": result.get("wilab_pending_packets", 0),
                "radio_events": result.get("radio_events", []),
                "selected_assignments": action.get("assignments", []),
                "proposed_assignments": action.get("proposed_assignments", action.get("assignments", [])),
                "governor_intervened": action.get("governor_intervened", False),
                "governor_reason": action.get("governor_reason", "disabled"),
                "governor_max_bytes": action.get("governor_max_bytes", None),
                "helper_packet_counts": helper_packet_counts,
                "helper_payload_bytes": helper_payload_bytes,
                "helper_swap_applied": bool(action.get("helper_swap_applied", False)),
                "helper_swap_spec": action.get("helper_swap_spec"),
                "pack_swap_gate": action.get("pack_swap_gate"),
                "hq_spo_gate": action.get("hq_spo_gate"),
                "hq_spo_baseline": action.get("hq_spo_baseline"),
                "structured_priority_meta": action.get("structured_priority_meta"),
                "weights": action.get("weights"),
                "policy_mode": action.get("policy_mode"),
                "reward": reward,
                "delivered_source_ids": delivered_ids,
            }
            enrich_audit_epoch(
                epoch_rec,
                delivered_task_ids=delivered_tasks,
                gate_blocked=bool(action.get("admission_gate_blocked")),
            )
            audit.append(epoch_rec)
            if hasattr(policy, "note_window"):
                try:
                    policy.note_window(epoch_rec)
                except Exception:
                    pass
            # PPO identity: joint logπ of proposed pack (not executed-after-governor).
            proposed_lp = action.get("proposed_logp", lp)
            if not torch.is_tensor(proposed_lp):
                proposed_lp = torch.tensor(float(proposed_lp), dtype=torch.float32)
            logp.append((proposed_lp, reward))
            trace.append({
                "logp": proposed_lp,
                "value": action.get("sp_value", value),
                "cost_value": action.get("sp_cost_value", cost_value),
                "reward": float(reward),
                "normalized_reward": float(reward) / max(offered_value, 1e-9),
                "offered_value": offered_value,
                "observation": obs_vec.detach().cpu().tolist(),
                "sp_obs": action.get("sp_obs"),
                "z_w": action.get("z_w"),
                "z_b": action.get("z_b"),
                "pb_state": action.get("pb_state"),
                "pb_z": action.get("pb_z"),
                "pb_crl": action.get("pb_crl"),
                "pb_full_obs": action.get("pb_full_obs"),
                "assignments": action.get("executed_assignments", action.get("assignments", [])),
                "proposed_assignments": action.get("proposed_assignments", []),
                "executed_assignments": action.get("executed_assignments", action.get("assignments", [])),
                "selected_candidate_indices": list(
                    action.get("proposed_selected_candidate_indices")
                    or action.get("selected_candidate_indices")
                    or []
                ),
                "proposed_selected_candidate_indices": list(
                    action.get("proposed_selected_candidate_indices")
                    or action.get("selected_candidate_indices")
                    or []
                ),
                "pack_logp_is_proposed": True,
                "candidate_features": action.get("candidate_features", []),
                "candidate_analytic_scores": action.get("candidate_analytic_scores", []),
                "candidate_task_ids": action.get("candidate_task_ids", []),
                "candidate_source_ids": action.get("candidate_source_ids", []),
                "candidate_payload_bytes": action.get("candidate_payload_bytes", []),
                "candidate_helper_ids": action.get("candidate_helper_ids", []),
                "candidate_finish_ms": action.get("candidate_finish_ms", []),
                "candidate_deadline_ms": action.get("candidate_deadline_ms", []),
                "candidate_task_values": action.get("candidate_task_values", []),
                "candidate_cycles": action.get("candidate_cycles", []),
                "bytes": int(action.get("computation_bytes_offered", 0)),
                "residual_changed_anchor": bool(action.get("residual_changed_anchor", False)),
                "governor_intervened": bool(action.get("governor_intervened", False)),
                "governor_reason": action.get("governor_reason", "disabled"),
                "cbr_valid": bool(state.get("cbr_valid", False)),
                "delay_valid": bool(state.get("delay_valid", False)),
                "prr": state.get("prr"),
                "p95_delay_ms": state.get("p95_delay_ms"),
            })
            total += reward
            epoch += 1
        # Terminal drain (spread arrivals): finalize queued deadline-valid work that
        # becomes due by the controller horizon so G_0 / V_done match γ=1 accounting.
        if spread_task_arrivals:
            before_drain = layer.completed_value()
            layer.set_epoch(int(controller_epochs))
            drain_reward = float(layer.completed_value() - before_drain)
            if drain_reward != 0.0:
                total += drain_reward
                if audit:
                    audit[-1]["reward"] = float(audit[-1].get("reward") or 0.0) + drain_reward
                    audit[-1]["drain_reward"] = drain_reward
                if trace:
                    offered = float(trace[-1].get("offered_value") or 1.0)
                    trace[-1]["reward"] = float(trace[-1].get("reward") or 0.0) + drain_reward
                    trace[-1]["normalized_reward"] = float(trace[-1]["reward"]) / max(offered, 1e-9)
                    trace[-1]["drain_reward"] = drain_reward
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=60.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        log_handle.close()
        if proc is not None and proc.returncode != 0 and (controller_epochs is None or epoch < int(controller_epochs)):
            raise RuntimeError(f"MATLAB episode failed ({proc.returncode}); see {log_path}:\n{log_path.read_text(encoding='utf-8', errors='replace')[-4000:]}")
    (out / "bridge_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    # Always persist machine-readable performance metrics (episode/epoch/task/decision).
    metrics_meta = {
        "episode_id": episode,
        "density": density,
        "sim_seed": sim_seed,
        "task_seed": (90000 + episode) if task_seed is None else int(task_seed),
        "policy_mode": policy_mode,
        "envelope_bytes": forced_byte_budget if forced_byte_budget is not None else value_envelope_bytes,
        "tasks_per_source": tasks_per_source,
        "deadline_multiplier": deadline_multiplier,
        "max_candidates": max_candidates,
        "governor_enabled": governor is not None,
    }
    try:
        write_episode_metrics(
            out,
            audit=audit,
            trace=trace,
            layer=layer,
            meta=metrics_meta,
            total_return=float(total),
        )
        (out / "task_layer_state.json").write_text(
            json.dumps(layer.state(), indent=2) + "\n", encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001 — metrics must not fail the episode
        (out / "metrics_export_error.txt").write_text(str(exc), encoding="utf-8")
    if logp:
        loss = -torch.stack([lp * float(r) for lp, r in logp]).sum()
        if return_trace: return total, loss, layer.completed_value(), trace
        return total, loss, layer.completed_value()
    if return_trace: return total, torch.tensor(0.0), layer.completed_value(), trace
    return total, torch.tensor(0.0), layer.completed_value()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--external-matlab", action="store_true")
    ap.add_argument("--start-episode", type=int, default=0)
    ap.add_argument("--resume-checkpoint", type=str, default=None)
    ap.add_argument("--output-json", type=str, default=None)
    ap.add_argument("--safety-only", action="store_true",
                    help="run WiLab safety traffic only; never admit computation or update the policy")
    ap.add_argument("--density", type=int, default=None,
                    help="single density for an isolated calibration run")
    ap.add_argument("--generation-ms", type=float, default=10.0)
    ap.add_argument("--allocation-ms", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--controller-epochs", type=int, default=100)
    ap.add_argument("--forced-budget", type=int, default=None,
                    help="calibration mode: force computation budget a, bypassing the learned shield")
    ap.add_argument("--load-multiplier", type=float, default=1.0,
                    help="offered computation workload multiplier; 3 tasks/source at 1.0")
    ap.add_argument("--all-eligible", action="store_true",
                    help="calibration mode: admit all eligible candidates; nonbinding budget")
    ap.add_argument("--forced-byte-budget", type=int, default=None,
                    help="calibration mode: cap cumulative admitted task payload bytes")
    ap.add_argument("--safety-payload-bytes", type=int, default=1000,
                    help="WiLab safety-message payload size in bytes")
    ap.add_argument("--simulation-time", type=float, default=20.0,
                    help="WiLab simulation duration in seconds; defaults to the frozen 20 s")
    ap.add_argument("--disable-safety-governor", action="store_true",
                    help="disable the execution-time governor (calibration/debug only)")
    ap.add_argument("--governor-max-bytes", type=int, default=175000,
                    help="hard execution-time computation envelope in bytes")
    ap.add_argument("--risk-calibration", type=str, default=None,
                    help="certified next-window risk table; absent means fail-safe zero computation")
    ap.add_argument("--horizon-safety-envelope", action="store_true",
                    help="use the frozen long-horizon empirical byte envelope; do not require short-window risk artifact")
    args = ap.parse_args(); print("runner start", flush=True); root = Path(__file__).resolve().parents[2]
    # Use a local temp directory for the live IPC channel. This avoids file
    # synchronization/permission virtualization in managed worktree mounts;
    # episode logs/results remain in the repository.
    bridge = Path(os.environ.get("TEMP", str(root / "results"))) / "wilab_closed_loop_bridge"
    checkpoint = Path(args.resume_checkpoint) if args.resume_checkpoint else root / "checkpoints" / "residual_task_policy_cf.pt"
    policy = ResidualTaskPolicy(); print("policy created", flush=True); policy.load_state_dict(torch.load(checkpoint, map_location="cpu")["policy"], strict=False); policy.train(); print(f"checkpoint loaded: {checkpoint}", flush=True)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4); rows=[]
    n = 1 if args.smoke else args.episodes
    start = 0 if args.smoke else args.start_episode
    for ep in range(start, start + n):
        print(f"episode {ep} start", flush=True)
        density = args.density if args.density is not None else (50,100,150,200)[ep % 4]
        tasks_per_source = max(1, int(math.ceil(3.0 * args.load_multiplier)))
        use_governor = (not args.disable_safety_governor and not args.safety_only and
                        args.forced_budget is None and not args.all_eligible and
                        args.forced_byte_budget is None)
        governor = SafetyGovernor(args.governor_max_bytes,
                                   risk_calibration=args.risk_calibration,
                                   horizon_envelope_only=args.horizon_safety_envelope) if use_governor else None
        total, loss, completed = run_episode(root, bridge, ep, density, policy, 600.0, args.external_matlab, args.safety_only, args.generation_ms, args.allocation_ms, args.seed, args.controller_epochs, args.forced_budget, tasks_per_source, args.all_eligible, args.forced_byte_budget, args.safety_payload_bytes, args.simulation_time, governor=governor)
        if loss.requires_grad and not args.safety_only and args.forced_budget is None and not args.all_eligible and args.forced_byte_budget is None:
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        rows.append({"episode":ep,"density":density,"return":total,"completed_value":completed,"loss":float(loss.detach())})
        print(rows[-1], flush=True)
    default_name = "wilab_safety_only.json" if args.safety_only else ("wilab_closed_loop_smoke.json" if args.smoke else "wilab_closed_loop_training_100ep.json")
    out = Path(args.output_json) if args.output_json else root / "results" / "additional_validation_20261007" / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not args.smoke and not args.safety_only and args.forced_budget is None and not args.all_eligible and args.forced_byte_budget is None:
        checkpoint_name = root / "checkpoints" / ("residual_task_policy_wilab_150ep.pt" if start >= 100 else "residual_task_policy_wilab_100ep.pt")
        torch.save({"policy":policy.state_dict(),"algorithm":"WiLab closed-loop residual RL","episodes":start+n}, checkpoint_name)


if __name__ == "__main__": main()
