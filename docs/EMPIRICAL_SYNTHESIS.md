# Empirical Synthesis: Sidelink Physics, Coupling, and Lessons Learned

## 1. Physical Environment & Coupling Characterization

A core requirement for evaluating safety-constrained task offloading is operating in a **physically coupled regime** where offload computation traffic directly competes with safety beacons for radio resources.

### Sidelink Mode 2 Channel Dynamics
- **Beacon Traffic:** Periodic safety beacons (BSMs/CAMs) transmitted at 10 Hz (every 100 ms) via Semi-Persistent Scheduling (SPS).
- **Computation Traffic:** Aperiodic offload packet bursts admitted by the offloading controller.
- **Evaluation Window:** 20 ms epochs (50 epochs = 1.0 s post-warmup evaluation horizon).

### Coupling Metrics ($N = 30$ vs. $N = 28$)
Across extensive closed-loop WiLab simulations on screening seeds `97000`–`97019`, coupling between computation offloading and radio channel congestion is strong and unambiguous:

| Metric | Cell $N=30$ (`cs=0.70, tps=5`) | Cell $N=28$ (`cs=0.70, tps=5`) |
| :--- | :--- | :--- |
| **No-Offload CBR** | $0.0638$ | $0.0644$ |
| **B1 CBR (Offload Everything)** | $0.0858$ (**$+34.5\%$**) | $0.0886$ (**$+37.7\%$**) |
| **B1 Bytes Offered** | $38,107\text{ B}$ | $38,419\text{ B}$ |
| **P95 Sidelink Delay** | $18.6\text{ ms} \to 37.6\text{ ms}$ ($+19.0\text{ ms}$) | $18.5\text{ ms} \to 37.4\text{ ms}$ ($+18.9\text{ ms}$) |
| **B1 Mean PRR Drop** | $-2.4\text{ pp}$ ($0.947 \to 0.923$) | $-3.1\text{ pp}$ ($0.963 \to 0.931$) |
| **B1 Infeasibility Rate ($Z > 2$)** | $20 / 20$ ($100\%$) | $20 / 20$ ($100\%$) |

---

## 2. Resolving the Delay Hypothesis

Prior interim reports suspected that the background violation floor was caused by an overly sensitive P95 delay threshold ($> 50\text{ ms}$).

### Empirical Audit Across 120 Episodes:
- When isolating delay violations ($p_{95} > 50\text{ ms}$), exceedances were **0 / 20 for No-Offload and 0 / 20 for B1** in all cells. Delay never crossed 50 ms.
- However, delay responded strongly to offloading ($\approx +19\text{ ms}$, doubling from $18.5\text{ ms}$ to $37.5\text{ ms}$).
- **Conclusion:** Delay is a valid, sensitive diagnostic of channel loading, but the background violation floor is entirely driven by Packet Reception Ratio (PRR < 0.95).

---

## 3. SPS Reservation Physics & Event-Based Scoring

When analyzing per-epoch traces at the sub-window level, a critical characteristic of 3GPP Mode 2 SPS was discovered:

### The Reservation Lottery Problem:
1. In Mode 2, vehicles select resource reservations with a reselection counter ($C_{\text{resel}} \approx 5\text{--}15$ periods).
2. If two vehicles choose the same subchannel/time slot on initialization, they experience **recurrent collisions every 100 ms** (every 5 epochs) until one reselects.
3. In a 50-epoch (1.0 s) simulation, a single initial collision produces periodic drops at epochs 2–3, 7–8, 12–13, 17–18, etc.
4. If scoring counts raw 20 ms windows, one initial collision creates $Z \approx 10\text{--}20$ violating windows.
5. Even when merging consecutive violating windows into **events**, collisions recurring across separate 100 ms beacon periods cannot be merged. Thus, any seed with an initial collision lottery results in $Z_{\text{event}} \approx 10 > K=2$.

### Clean Seeds vs. Colliding Seeds:
- At $N = 28$, exactly **6 of 20 seeds ($30\%$)** are collision-free under No-Offload ($Z = 0, \text{PRR} \ge 0.988$).
- On those 6 clean seeds, **B1 causes severe, attributable degradation on 6 of 6 seeds ($100\%$)**, dropping PRR by up to 10.6 points and causing $Z \in [5, 11]$.
- On the remaining 14 seeds, the background lottery itself triggers recurrent beacon losses even with zero offload traffic.

---

## 4. Why Prior RL Underperformed (Diagnosing U = 5.75 vs. Baselines ~100+)

In earlier evaluation tables, PB-CRL achieved only $U = 5.75$ and B5 achieved $U = 0$, while B1/B3 earned $118$. Why did this happen?

1. **Overly Conservative Safety Shield / External Clamping:**
   The policy was evaluated with a static safety governor that clamped allowable bytes $B_{\text{eff}} \to 0$ whenever channel metrics approached nominal thresholds. The RL policy never had the opportunity to explore and transmit.
2. **Unlearned Dual Prices ($\lambda \approx 0$ or Diverged):**
   In early training checkpoints, the Lagrangian dual price $\lambda$ was either frozen at 0 (ignoring safety) or diverged to large values, inducing "learned helplessness" where the policy concluded any offload incurs fatal penalty.
3. **Lack of End-to-End Headroom Discovery:**
   The RL agent must be trained in the coupled environment with an uninhibited action space so it actively discovers the maximum safe burst size and learns value-dense task selection.
