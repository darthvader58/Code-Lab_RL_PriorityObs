from pathlib import Path
p = Path(__file__).resolve().parents[1] / "geos5data_0.0625deg_30mn_trackwise.ipynb"
s = p.read_text()
if s.endswith("}\\n"):
    s = s[:-3] + "}\n"
elif s.endswith("}}\\n"):
    s = s[:-4] + "}\n"
elif s.endswith("}}\n"):
    s = s[:-3] + "}\n"
p.write_text(s)
print("fixed notebook JSON terminator")
