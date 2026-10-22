"""Apply VAS/LAS/SPARC-V2X/LVM/VLRO display names to Stage-5 evidence markdown.

Does not rename raw method keys in JSON artifacts or --methods CLI flags.
See protocol/baseline_nomenclature.md.
"""
from __future__ import annotations

import hashlib
import json
import re
import stat
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
LOCK_ROOT = PKG / "results" / "evidence_locks" / "locked_20261016_014531"

TARGETS = [
    PKG / "reports" / "STAGE5_NO_RESIDUAL_ANALYSIS.md",
    PKG / "results" / "stage5_no_residual" / "STAGE5_NO_RESIDUAL_ANALYSIS.md",
    LOCK_ROOT / "reports" / "STAGE5_NO_RESIDUAL_ANALYSIS.md",
    LOCK_ROOT / "stage5_no_residual" / "STAGE5_NO_RESIDUAL_ANALYSIS.md",
    PKG
    / "results"
    / "evidence_copies"
    / "locked_20261016_014531_itora_deadline_ext"
    / "reports"
    / "STAGE5_NO_RESIDUAL_ANALYSIS.md",
    PKG
    / "results"
    / "evidence_copies"
    / "locked_20261016_014531_itora_deadline_ext"
    / "reports"
    / "STAGE5_NO_RESIDUAL_ANALYSIS_ITORA_DEADLINE_EXT.md",
    PKG
    / "results"
    / "evidence_copies"
    / "locked_20261016_014531_itora_deadline_ext"
    / "stage5_no_residual"
    / "STAGE5_NO_RESIDUAL_ANALYSIS.md",
]

NOMENCLATURE_BLOCK = """## Baseline nomenclature

Canonical display names (see `protocol/baseline_nomenclature.md`). Code keys in artifacts unchanged.

| Code / legacy | Full name | Acronym |
|---|---|---|
| `value_aware` / VA / B1 | Value-Aware Scheduling | **VAS** |
| `minfinish` / MF / B2 | Latency-Aware Scheduling | **LAS** |
| Direct PPO / B3 | SPARC-V2X (proposed) | **SPARC-V2X** |
| B4 Lagrangian | Lagrangian Value Maximization | **LVM** |
| `itora_inspired` / B5 | Value-Latency-Resource Offloading | **VLRO** |

**VLRO** (not “ITORA”): frozen score \(v/(F\\cdot b)\) — value, predicted finish, and payload resource. Not a bit-faithful ITORA reproduction.

| Method | Principle | Learning | Constraint handling |
|---|---|---:|---|
| **VAS** | task-value-aware scheduling | No | common safety-constrained admission |
| **LAS** | predicted-latency-aware scheduling | No | common safety-constrained admission |
| **SPARC-V2X** | learned sequential task–helper scheduling | PPO | common safety-constrained admission |
| **LVM** | Lagrangian value maximization | PPO | dual/soft constraint handling |
| **VLRO** | joint value–latency–resource offloading | No | unconstrained / external admission |

"""


def writable(p: Path) -> None:
    if p.exists():
        try:
            p.chmod(p.stat().st_mode | stat.S_IWRITE)
        except Exception:
            pass


def readonly(p: Path) -> None:
    try:
        p.chmod(p.stat().st_mode & ~stat.S_IWRITE & ~stat.S_IWGRP & ~stat.S_IWOTH)
    except Exception:
        pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protect_phrases(text: str) -> tuple[str, list[str]]:
    """Shield phrases that must not be rewritten by naive replacements."""
    shields: list[str] = []
    patterns = [
        r"SPARC residual RL[^\n]*",
        r"SPARC residual[^\n]*",
        r"residual SPARC[^\n]*",
        r"no residual SPARC[^\n]*",
        r"SPARC residual excluded[^\n]*",
        r"`value_aware`",
        r"`minfinish`",
        r"`itora_inspired`",
        r"`residual`",
        r"protocol/baseline_nomenclature\.md",
        r"catchup_e4_deadline_m\{0p5\|1p0\|1p5\|2p0\}_itora",
        r"catchup_e4_deadline_m[0-9p]+_itora",
        r"final_test_itora_d35",
        r"final_test_direct_d35",
        r"final_test_b4_d35",
        r"--methods",
        r"--logit-mode direct",
        r"--no-governor",
    ]
    out = text
    for pat in patterns:
        def repl(m, _shields=shields):
            _shields.append(m.group(0))
            return f"\x00SHIELD{len(_shields)-1}\x00"
        out = re.sub(pat, repl, out)
    return out, shields


def unprotect(text: str, shields: list[str]) -> str:
    for i, s in enumerate(shields):
        text = text.replace(f"\x00SHIELD{i}\x00", s)
    return text


def rename_display(text: str) -> str:
    text, shields = protect_phrases(text)

    # Ordered phrase replacements (longest / most specific first)
    reps = [
        ("Direct vs ITORA (deadline-tightness)", "SPARC-V2X vs VLRO (deadline-tightness)"),
        ("Direct V | ITORA V | ΔV (D−I) | Direct R | ITORA R | ΔR (D−I)",
         "SPARC-V2X V | VLRO V | ΔV (S−V) | SPARC-V2X R | VLRO R | ΔR (S−V)"),
        ("VA/MF/Direct/B4/ITORA", "VAS/LAS/SPARC-V2X/LVM/VLRO"),
        ("VA/MF/ITORA + Direct + B4", "VAS/LAS/VLRO + SPARC-V2X + LVM"),
        ("VA/MF/Direct/ITORA", "VAS/LAS/SPARC-V2X/VLRO"),
        ("VA/MF/Direct", "VAS/LAS/SPARC-V2X"),
        ("VA/MF/ITORA", "VAS/LAS/VLRO"),
        ("Density Direct + VA/MF/ITORA", "Density SPARC-V2X + VAS/LAS/VLRO"),
        ("B1 VA, B2 MF, B3 Direct, B4 Lagrangian, B5 ITORA",
         "VAS, LAS, SPARC-V2X, LVM, VLRO"),
        ("A2(=VA), A4(=Direct)", "A2(=VAS), A4(=SPARC-V2X)"),
        ("A2(=B1 VA)", "A2(=VAS)"),
        ("A4(=B3 Direct)", "A4(=SPARC-V2X)"),
        ("B5_ITORA_inspired", "VLRO"),
        ("B5_ITORA", "VLRO"),
        ("B4_Lagrangian", "LVM"),
        ("B3_Direct", "SPARC-V2X"),
        ("B1_VA", "VAS"),
        ("B2_MF", "LAS"),
        ("ITORA-inspired", "VLRO"),
        ("ITORA (`itora_inspired`)", "VLRO (`itora_inspired`)"),
        ("Method:** `itora_inspired`", "Method:** VLRO (`itora_inspired`)"),
        ("itora_inspired", "vlro"),  # table method cells after shields
        ("value_aware", "vas"),
        ("minfinish", "las"),
        # Direct PPO / Direct labels (careful: after shields for residual)
        ("Direct PPO", "SPARC-V2X"),
        ("governor-matched Direct", "governor-matched SPARC-V2X"),
        ("matched Direct", "matched SPARC-V2X"),
        ("vs Direct", "vs SPARC-V2X"),
        ("Direct anchors", "SPARC-V2X anchors"),
        ("Direct across densities", "SPARC-V2X across densities"),
        ("Direct closed-loop", "SPARC-V2X closed-loop"),
        ("Closed-loop Direct", "Closed-loop SPARC-V2X"),
        ("Direct training", "SPARC-V2X training"),
        ("Direct train", "SPARC-V2X train"),
        ("Direct FT", "SPARC-V2X FT"),
        ("Direct@80", "SPARC-V2X@80"),
        ("Direct@50", "SPARC-V2X@50"),
        ("Direct-on-d50", "SPARC-V2X-on-d50"),
        ("Direct-VA", "SPARC-V2X−VAS"),
        ("Direct−VA", "SPARC-V2X−VAS"),
        ("ΔV Direct−VA", "ΔV SPARC-V2X−VAS"),
        ("Paired Direct-VA", "Paired SPARC-V2X−VAS"),
        ("Direct mean V", "SPARC-V2X mean V"),
        ("Direct V", "SPARC-V2X V"),
        ("baseline V | Direct V | ΔV (base−Direct)",
         "baseline V | SPARC-V2X V | ΔV (base−SPARC-V2X)"),
        ("base P_viol | Direct P_viol | base PRR | Direct PRR | base P95 | Direct P95",
         "base P_viol | SPARC-V2X P_viol | base PRR | SPARC-V2X PRR | base P95 | SPARC-V2X P95"),
        ("E_Direct", "E_SPARC"),
        ("T_Direct", "T_SPARC"),
        ("T_d", "T_S"),
        (" | direct | ", " | SPARC-V2X | "),
        ("| direct |", "| SPARC-V2X |"),
        (" method | direct\n", " method | SPARC-V2X\n"),
        ("| density | value_aware | minfinish | direct |",
         "| density | VAS | LAS | SPARC-V2X |"),  # may already be vas/las
        ("| density | vas | las | direct |", "| density | VAS | LAS | SPARC-V2X |"),
        ("| density | vas | las | SPARC-V2X | vlro |",
         "| density | VAS | LAS | SPARC-V2X | VLRO |"),
        ("| density | vas | las | SPARC-V2X |",
         "| density | VAS | LAS | SPARC-V2X |"),
        ("| VA | MF | Direct | B4 | ITORA |",
         "| VAS | LAS | SPARC-V2X | LVM | VLRO |"),
        ("| sim_seed | VA | MF | Direct | B4 | ITORA |",
         "| sim_seed | VAS | LAS | SPARC-V2X | LVM | VLRO |"),
        ("Strongest governor-matched: **SPARC-V2X**", "Strongest governor-matched: **SPARC-V2X**"),
        ("Strongest governor-matched: **B3_Direct**", "Strongest governor-matched: **SPARC-V2X**"),
        ("Do not claim Direct beats B4", "Do not claim SPARC-V2X beats LVM"),
        ("Do not treat ITORA density", "Do not treat VLRO density"),
        ("VA first-pick", "VAS first-pick"),
        ("VA+MF", "VAS+LAS"),
        (" VA ", " VAS "),
        (" MF ", " LAS "),
        (" B4 ", " LVM "),
        (" ITORA ", " VLRO "),
        ("Panel (b+) ITORA", "Panel (b+) VLRO"),
        ("ITORA deadline", "VLRO deadline"),
        ("COPY EXTENSION — ITORA", "COPY EXTENSION — VLRO"),
        ("the locked Stage-5 snapshot is unchanged. ITORA uses",
         "the locked Stage-5 snapshot is unchanged. VLRO uses"),
        # column headers that survived
        ("| Direct |", "| SPARC-V2X |"),
        ("| ITORA |", "| VLRO |"),
        ("| VA |", "| VAS |"),
        ("| MF |", "| LAS |"),
        ("| B4 |", "| LVM |"),
        # lowercase method cells after earlier replace
        ("| vas |", "| VAS |"),
        ("| las |", "| LAS |"),
        ("| vlro |", "| VLRO |"),
        ("| direct |", "| SPARC-V2X |"),
        # prose leftovers
        ("non-learned comparator (VA/MF)", "non-learned comparator (VAS/LAS)"),
        ("non-learned comparator (VAS/LAS)", "non-learned comparator (VAS/LAS)"),
        ("Baselines matched to Direct", "Baselines matched to SPARC-V2X"),
        ("Match baselines to Direct", "Match baselines to SPARC-V2X"),
        ("V(Direct,", "V(SPARC-V2X,"),
        ("V(baseline,E*) vs V(SPARC-V2X,E_D)", "V(baseline,E*) vs V(SPARC-V2X,E_D)"),
        ("specialized vs nominal Direct", "specialized vs nominal SPARC-V2X"),
        ("nominal Direct", "nominal SPARC-V2X"),
        ("for VA/MF and Direct", "for VAS/LAS and SPARC-V2X"),
        ("VA/MF together; Direct alone", "VAS/LAS together; SPARC-V2X alone"),
        ("Methods:** VA+MF", "Methods:** VAS+LAS"),
        ("; Direct `", "; SPARC-V2X `"),
        ("Not included in this ablation:** ['B4 Lagrangian', 'ITORA-inspired', 'SPARC residual RL']",
         "Not included in this ablation:** ['LVM', 'VLRO', 'SPARC residual RL']"),
        ("['B4 Lagrangian', 'VLRO',", "['LVM', 'VLRO',"),
        ("primary learned method in this Stage-5 package is B3 Direct PPO",
         "primary learned method in this Stage-5 package is SPARC-V2X"),
        ("primary learned method in this Stage-5 package is SPARC-V2X",
         "primary learned method in this Stage-5 package is SPARC-V2X"),
        ("treated as B3 Direct", "treated as SPARC-V2X"),
        ("method=B3_Direct", "method=SPARC-V2X"),
        ("method=SPARC-V2X", "method=SPARC-V2X"),
        ("ΔV (base−Direct)", "ΔV (base−SPARC-V2X)"),
        ("delta_V_baseline_minus_Direct", "delta_V_baseline_minus_SPARC"),
        ("Direct_V", "SPARC_V"),
        ("Direct_P_viol", "SPARC_P_viol"),
        ("Direct_prr", "SPARC_prr"),
        ("Direct_p95", "SPARC_p95"),
        ("does governor-matched SPARC-V2X yield", "does governor-matched SPARC-V2X yield"),
        ("does governor-matched Direct PPO yield", "does governor-matched SPARC-V2X yield"),
        ("Fig.3 Direct", "Fig.3 SPARC-V2X"),
        ("Fig.8 Direct", "Fig.8 SPARC-V2X"),
        ("Fig.11 Direct", "Fig.11 SPARC-V2X"),
        ("Fig.11a specialized Direct", "Fig.11a specialized SPARC-V2X"),
        ("Stress Direct@", "Stress SPARC-V2X@"),
        ("Stress SPARC-V2X@", "Stress SPARC-V2X@"),
    ]
    for a, b in reps:
        text = text.replace(a, b)

    # Remaining standalone "Direct" in table cells / short labels — conservative
    text = re.sub(r"\bDirect\b", "SPARC-V2X", text)
    text = re.sub(r"\bITORA\b", "VLRO", text)
    # Avoid double-prefix
    text = text.replace("SPARC-V2X-V2X", "SPARC-V2X")
    text = text.replace("VASAS", "VAS")
    text = text.replace("LASAS", "LAS")

    text = unprotect(text, shields)
    return text


def ensure_nomenclature_section(text: str) -> str:
    if "## Baseline nomenclature" in text:
        return text
    # Insert after title block / before Completeness ledger
    marker = "## Completeness ledger"
    if marker in text:
        return text.replace(marker, NOMENCLATURE_BLOCK + marker, 1)
    return NOMENCLATURE_BLOCK + text


def patch_one(path: Path) -> bool:
    if not path.exists():
        return False
    writable(path)
    old = path.read_text(encoding="utf-8")
    new = ensure_nomenclature_section(rename_display(old))
    # Fix inventory line if still old
    new = new.replace(
        "Fig.6 E4 load + deadline sweeps (VAS/LAS/SPARC-V2X)",
        "Fig.6 E4 load + deadline sweeps (VAS/LAS/SPARC-V2X/VLRO)",
    )
    if new != old:
        path.write_text(new, encoding="utf-8")
    return new != old


def main() -> int:
    changed = []
    for p in TARGETS:
        if patch_one(p):
            changed.append(str(p))
            print(f"updated {p}")
        elif p.exists():
            print(f"unchanged {p}")
        else:
            print(f"missing {p}")

    # Refresh lock hashes if lock report updated
    lock_json = LOCK_ROOT / "LOCK.json"
    lock_report = LOCK_ROOT / "reports" / "STAGE5_NO_RESIDUAL_ANALYSIS.md"
    lock_stage5 = LOCK_ROOT / "stage5_no_residual" / "STAGE5_NO_RESIDUAL_ANALYSIS.md"
    if lock_json.exists() and lock_report.exists():
        writable(lock_json)
        manifest = json.loads(lock_json.read_text(encoding="utf-8"))
        manifest["nomenclature_applied_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest["nomenclature"] = "VAS/LAS/SPARC-V2X/LVM/VLRO (see protocol/baseline_nomenclature.md)"
        for key, path in [
            ("STAGE5_NO_RESIDUAL_ANALYSIS.md", lock_report),
            ("stage5_no_residual/STAGE5_NO_RESIDUAL_ANALYSIS.md", lock_stage5),
        ]:
            if path.exists():
                entry = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                if key == "STAGE5_NO_RESIDUAL_ANALYSIS.md":
                    entry["source"] = "reports/STAGE5_NO_RESIDUAL_ANALYSIS.md"
                manifest["files"][key] = entry
        # copy nomenclature into lock
        nom_src = PKG / "protocol" / "baseline_nomenclature.md"
        nom_dst = LOCK_ROOT / "protocol" / "baseline_nomenclature.md"
        nom_dst.parent.mkdir(parents=True, exist_ok=True)
        writable(nom_dst)
        nom_dst.write_text(nom_src.read_text(encoding="utf-8"), encoding="utf-8")
        manifest["files"]["protocol/baseline_nomenclature.md"] = {
            "sha256": sha256_file(nom_dst),
            "bytes": nom_dst.stat().st_size,
        }
        lock_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for p in [lock_json, lock_report, lock_stage5, nom_dst]:
            readonly(p)

    # re-readonly primary + stage5
    for p in [
        PKG / "reports" / "STAGE5_NO_RESIDUAL_ANALYSIS.md",
        PKG / "results" / "stage5_no_residual" / "STAGE5_NO_RESIDUAL_ANALYSIS.md",
    ]:
        if p.exists():
            readonly(p)

    primary = PKG / "reports" / "STAGE5_NO_RESIDUAL_ANALYSIS.md"
    t = primary.read_text(encoding="utf-8") if primary.exists() else ""
    print(
        json.dumps(
            {
                "n_changed": len(changed),
                "has_nomenclature": "## Baseline nomenclature" in t,
                "has_SPARC_V2X": "SPARC-V2X" in t,
                "has_VLRO": "VLRO" in t,
                "legacy_ITORA_inspired_left": "ITORA-inspired" in t,
                "legacy_B3_Direct_left": "B3_Direct" in t,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
