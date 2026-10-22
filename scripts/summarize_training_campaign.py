import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
hist_file = ROOT / "sparc_performance_evaluation" / "results" / "raw" / "sp_train" / "sp_training_history.json"
hist = json.loads(hist_file.read_text(encoding="utf-8"))

print(f"Total rounds recorded: {len(hist)}")
print(f"| Round | Train V | Val V | Delta B4 | P_viol (%) | Informative Steps | Distinct Matchings | Adv_R | Adv_C | Value Loss | Cost Loss | Grad Norm | KL | w_bias | beta |")
print(f"|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in hist:
    w_str = "[" + ", ".join(f"{x:.3f}" for x in r["weights_bias"]) + "]"
    print(f"| {r['round']} | {r['train_mean_v']:.2f} | {r['val_mean_v']:.2f} | {r['delta_b4']:+.2f} | {r['val_mean_p_viol']*100:.2f}% | {r.get('informative_steps', 0)} | {r.get('distinct_matchings', 0)} | {r['mean_adv_r']:+.2f} | {r['mean_adv_c']:+.2f} | {r['value_loss']:.2f} | {r['cost_loss']:.2f} | {r.get('grad_norm',0):.2f} | {r.get('approx_kl',0):.4f} | {w_str} | {r['beta_bias']:.3f} |")

print("\n--- Per-Seed Validation Outcomes Across Rounds ---")
for r in hist:
    row_strs = [f"Seed {v['seed']}: V={v['V_done']:.2f}, Pviol={v['P_viol']*100:.2f}%" for v in r.get("val_rows", [])]
    print(f"Round {r['round']}: " + " | ".join(row_strs))
