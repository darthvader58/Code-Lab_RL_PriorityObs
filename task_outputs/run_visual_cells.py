#!/usr/bin/env python3
"""Execute only completed-dataset analysis/visual cells and store outputs."""
import copy
import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "geos5data_0.0625deg_30mn_trackwise.ipynb"
nb = nbformat.read(NB_PATH, as_version=4)

selected = [8, 9, 10, 11, 12, 13, 14, 15]
setup = nbformat.v4.new_code_cell(
    """import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
DECISION = timedelta(minutes=1)
MSI = ["tautot", "tauhgh", "taulow", "cldtot", "cldhgh", "cldmid",
       "cldlow", "cldtmp", "lwtup"]
get_ipython().run_line_magic('matplotlib', 'inline')
"""
)
run_nb = nbformat.v4.new_notebook(metadata=copy.deepcopy(nb.metadata))
run_nb.cells = [setup] + [copy.deepcopy(nb.cells[i]) for i in selected]
for c in run_nb.cells:
    if c.cell_type == "code":
        c.outputs = []
        c.execution_count = None

client = NotebookClient(run_nb, timeout=900, startup_timeout=180,
                        kernel_name="trackwise-py",
                        resources={"metadata": {"path": str(ROOT)}})
client.execute()

# Transfer executed outputs/counts back to the corresponding notebook cells.
for source_idx, executed_cell in zip(selected, run_nb.cells[1:]):
    if nb.cells[source_idx].cell_type == "code":
        nb.cells[source_idx].outputs = executed_cell.outputs
        nb.cells[source_idx].execution_count = executed_cell.execution_count

nb.metadata["visual_execution"] = {
    "cells": selected,
    "mode": "completed parquet only",
    "network": "disabled for this run",
}
nbformat.write(nb, NB_PATH)
print(f"executed and stored outputs for cells {selected}")
