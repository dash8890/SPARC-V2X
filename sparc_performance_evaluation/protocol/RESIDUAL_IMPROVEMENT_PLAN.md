# Residual improvement experiments (post-lock evidence)

**Hard rule:** These runs must **not** mutate the manuscript evidence package.
They start only after `results/aggregates/campaign_aggregate_DONE` exists, and they
write exclusively under isolated paths below.

## Isolation (do not violate)

| Allowed | Forbidden |
|---|---|
| `results/wilab_improve_*` | `results/wilab_residual_ppo_96ep_va_relative` |
| `checkpoints/wilab_improve_*.pt` | `checkpoints/wilab_residual_ppo_96ep_va_relative.pt` |
| `sparc_performance_evaluation/results/raw/improve_*` | `final_test_nominal_d35`, `final_test_direct_d35`, A1/A3/A5/A6/B4 DONE trees |
| New seed bases ≥ 88000 | Reusing train/val/final-test sealed seeds for *training* of improve runs |

Frozen SPARC / Direct / VA final-test rows remain the lock baseline. Improve FTs
are labeled `improve_*` and compared offline.

## Problems under test

1. **VA soft-lock** — additive VA logits ⇒ VA clone on some sealed seeds.
2. **Weak / peaky convergence** — SPARC val ΔVA spikes then decays; brittle `best_delta_va` peak.

## Length policy (convergence / combined)

Do **not** commit to 192 episodes up front.

1. Train **96** (same as lock package) → sealed FT  
2. If **not** improved → continue **+48** (to 144) → FT again  
3. If still not → **stop** and wait for manual gate  
   `sparc_performance_evaluation/results/raw/improve_CONTINUE_LAST_48`  
   then optional final **+48** (to 192)

“Improved” = mean FT \(V\) > frozen Direct **and** residual ≠ VA on ≥4/5 seeds.

## Experiment matrix

### Escape-VA (I*) — fixed 96 only

1. **I1** high-β (`--residual-gain 3`)  
2. **I2** residual_only  
3. **I3** changed-only  
4. **I4** high-β + changed-only  

### Convergence (C*) / Combined (X*) — staged 96→+48→(?)+48

5. **C1** slower LR/entropy decay (`lr/ent final_frac 0.5`)  
6. **C2** more rollouts + entropy  
7. **C3** EMA ΔVA selection  
8. **C4** mean-V selection  
9. **X1** high-β + EMA + slow decay  
10. **X2** residual_only + EMA + slow decay  

Trainer: `--continue-from` + `--extra-episodes` for staged continues (defaults unchanged for evidence trains).

## Runner

```text
python -u sparc_performance_evaluation/scripts/residual_improvement_queue.py
```

Blocks until campaign aggregate DONE. To authorize the last +48 after ep144 fails:

```text
# create empty gate file
New-Item sparc_performance_evaluation/results/raw/improve_CONTINUE_LAST_48
```
