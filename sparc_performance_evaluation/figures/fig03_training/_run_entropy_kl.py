import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

nb = json.loads(
    Path("sparc_performance_evaluation/figures/fig03_training/fig03_learning_behavior.ipynb").read_text(
        encoding="utf-8"
    )
)
for idx in (3, 4):
    code = "".join(nb["cells"][idx]["source"]).replace("plt.show()", "plt.close()")
    exec(code, {"__name__": "__main__"})
