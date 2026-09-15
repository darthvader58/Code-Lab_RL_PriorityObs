import json
from pathlib import Path
p = Path(__file__).resolve().parents[1] / "geos5data_0.0625deg_30mn_trackwise.ipynb"
nb = json.loads(p.read_text())
src = "".join(nb["cells"][13].get("source", []))
src = src.replace("# Offline render: coastlines omitted to avoid downloading Natural Earth assets.",
                  "ax.coastlines(linewidth=0.6)")
nb["cells"][13]["source"] = src.splitlines(True)
p.write_text(json.dumps(nb, indent=1) + "\n")
print("restored cached local coastlines")

