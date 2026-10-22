# Stage 1 — Evaluation Framework (Revised)

**Status:** Locked for Stage-1 development and validation reporting.  
Supersedes the earlier rule that \(G<15\%\) is an automatic Stage-1 research failure.

---

## Primary metric

\[
G
=
\frac{V_{\mathrm{SPARC}}-V_{\mathrm{best\ matched}}}
{V_{\mathrm{best\ matched}}}
\times 100\%.
\]

- \(V_{\mathrm{best\ matched}}\): strongest method among SPARC’s matched competitors on the locked validation comparison (currently **LVM** unless another matched method is higher).
- Engineering benchmark vs Stage-0 LVM: \(1.15\times 46.8173\approx\mathbf{53.84}\) completed value.
- These percentage bands are **interpretive guidelines**, not new hard acceptance thresholds beyond the mandatories below.

| Result | Classification | Interpretation |
|---|---|---|
| \(G\ge 15\%\) | Target achieved | Strong result; still requires safety + statistical verification |
| \(5\%\le G<15\%\) | Promising | Potentially meaningful; weigh consistency, mechanism, robustness, complexity |
| \(0\%<G<5\%\) | Marginal | Positive; needs particularly strong supporting evidence |
| \(G\le 0\%\) | No demonstrated value advantage | Do not claim scheduling superiority |

---

## Mandatory (not relaxed)

1. **Safety:** \(P_{\mathrm{viol}}\le 2\%\), with uncertainty.
2. **Fairness:** SPARC, VAS, LAS, VLRO, LVM under identical matched admission (\(B_{\max}=175\) kB, common governor).
3. **Statistical credibility:** Paired validation results and uncertainty — not averages alone.
4. **Mechanism:** Evidence that context-conditioned scheduling meaningfully affects decisions.
5. **Reproducibility:** Locked train/val seeds; unchanged baselines; **no sealed final-test** during development.
6. **D7 pre-train gate:** Required before expensive training. Passing D7 **authorizes training**; it does **not** establish performance success.

---

## Dual decisions

### Decision A — Engineering target

Did Stage 1 meet \(G\ge 15\%\) over the strongest matched competitor (with safety held)?

Answer: **yes / no** (numerical).

### Decision B — Scientific meaning

Is the result a scientifically meaningful improvement worth developing further?

Answer considers: effect-size band, paired uncertainty, safety, decision quality / mechanism, robustness, and computational overhead.

| Pattern | Guidance |
|---|---|
| A yes, B yes | Strong case to request Stage 2 (final-test / manuscript) separately |
| A no, B yes (e.g. consistent 8–10% over LVM + mechanism) | Valuable; evidence-based retain/revise — **not** automatic reject |
| A yes, B no (e.g. 16% from one odd seed) | Do not treat as convincing success |
| A no, B no | Do not claim scheduling superiority; no final-test; postmortem |

---

## Hard stops (automatic negative for advancement)

- \(P_{\mathrm{viol}}>2\%\)
- Gain from unequal admission
- Trained policy still fails meaningful state-channel responsiveness
- Claims that ignore the strongest matched comparator (LVM unless superseded)

Missing \(G\ge15\%\) alone is **not** a hard stop; it forces Decision B scrutiny instead of automatic failure.
