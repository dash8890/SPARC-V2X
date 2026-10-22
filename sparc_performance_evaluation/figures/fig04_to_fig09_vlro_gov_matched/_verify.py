import json
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image

raw = Path(__file__).resolve().parents[2] / "results" / "raw"
out = Path(__file__).resolve().parent
bundle = out.parent / "fig04_to_fig09_bundle"


def dens_R(d: int) -> float:
    rows = json.loads((raw / f"vlro_gov_matched_density_d{d}" / "rows.json").read_text(encoding="utf-8-sig"))
    return mean(float(r["R_value"]) for r in rows if r["method"] == "itora_inspired")


print("RAW gov-matched VLRO R:")
for d in [35, 40, 50, 100]:
    print(f"  dens {d}: {dens_R(d):.6f}")

a = np.asarray(Image.open(out / "fig04a_R_vs_density.png").convert("RGB"), dtype=np.float64)
b_img = Image.open(bundle / "fig04a_R_vs_density.png").convert("RGB").resize((a.shape[1], a.shape[0]))
b = np.asarray(b_img, dtype=np.float64)
print("fig04a MSE vs bundle", float(((a - b) ** 2).mean()))

meta = json.loads((out / "vlro_gov_matched_plot_data.json").read_text())
print("plot_data dens50", meta["vlro_density"]["50"]["R_value"])
print("plot_data dens100", meta["vlro_density"]["100"]["R_value"])

STAGE5 = Path(__file__).resolve().parents[2] / "results" / "stage5_no_residual" / "figures"
d4 = json.loads((STAGE5 / "fig04_data.json").read_text())
pa = d4["panel_a_R_value_vs_density"]
DENS = [20, 25, 30, 35, 40, 45, 50]
print("Would plot VLRO:", [round(dens_R(d), 4) for d in DENS])
print("Old sealed VLRO:", [round(pa[str(d)]["itora_inspired"], 4) for d in DENS])
