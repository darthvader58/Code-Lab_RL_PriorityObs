import json
from pathlib import Path
p = Path(__file__).resolve().parents[1] / "geos5data_0.0625deg_30mn_trackwise.ipynb"
nb = json.loads(p.read_text())
for idx in [2, 13]:
    src = "".join(nb["cells"][idx].get("source", []))
    src = src.replace("%pip install tatc pyarrow", "# Dependencies are already installed; no package download is needed.")
    src = src.replace("%pip install seaborn cartopy", "# Dependencies are already installed; no package download is needed.")
    nb["cells"][idx]["source"] = src.splitlines(True)
p.write_text(json.dumps(nb, indent=1) + "\n")
print("removed notebook package-install directives")

