# Phase C0 — LVM-RPI Consequentiality Protocol

**Status:** AUTHORIZED (user 2026-10-19)  
**Formulation:** `SPARC_FORMULATION_LOCK.md` (D1) — do not reopen.

## Question

How much realized deadline-valid value can a feasible one-step deviation recover over LVM, under frozen edit families, on **fresh** seeds?

## Fixed mechanism

- \(A^B=\mathrm{LVM}(s)\); \(\mathcal C=\{A^{\mathrm{LVM}},A^{\mathrm{VLRO}},A^{\mathrm{Direct}}\}\cup\) edit families below.
- Complete packs; same governor/feasibility; cloned pre-decision state; common RNG where supported.
- \(H=1\): execute candidate at \(t\), then **LVM for the full residual episode**.
- \(\Delta(A)=G(A\rightarrow\pi_{\mathrm{LVM}})-G(A^{\mathrm{LVM}}\rightarrow\pi_{\mathrm{LVM}})\).
- \(\Delta^{\mathrm{safe}}=-\infty\) if branch violates frozen safety (\(P_{\mathrm{viol}}>0.02\)).

## Edit families (frozen)

LVM, VLRO, Direct, DROP-1, REBIND-1, SUBSTITUTE-1, ADD-1, DROP+ADD, REBIND+ADD.

No learned candidate generation in C0.

## Seeds

`72200+i` / `182200+i`. Forbidden: 8200x, 9300x, 72000–72095, 72100–72147.

## No numeric pass % 

Characterize \(P(\Delta^\star>0)\), \(E[\Delta^\star\mid >0]\), \(M_e\), edit-family decomposition. Do not sum one-step gains as deployable headroom.

## Outputs

See authorization list (1)–(12) → `SPARC_LVM_RPI_C0_REPORT.md` + `results/raw/lvm_rpi_c0/`.
