"""Ensure a V2 notebook reloads path-safety fixes in long nested checkouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
path = args.root.resolve() / "notebooks" / "03_multi_agent_gpt5_V2_optimized.ipynb"
notebook = nbformat.read(path, as_version=4)
cell = next(
    item for item in notebook.cells
    if item.cell_type == "code" and "ensure_fixed_split_v2" in item.source
)
old = "from evaluation.mcq_hybrid_v2 import load_evaluation_dataset, ensure_fixed_split_v2"
new = """import importlib
import evaluation.mcq_hybrid_v2 as _mcq_v2
importlib.reload(_mcq_v2)
load_evaluation_dataset = _mcq_v2.load_evaluation_dataset
ensure_fixed_split_v2 = _mcq_v2.ensure_fixed_split_v2"""
if old in cell.source:
    cell.source = cell.source.replace(old, new)
elif "importlib.reload(_mcq_v2)" not in cell.source:
    raise RuntimeError("Could not locate the V2 loader import")
cell.outputs = []
cell.execution_count = None

run_cell = next(
    item for item in notebook.cells
    if item.cell_type == "code" and "run_multi_agent_v2" in item.source
)
old_run = "from evaluation.mcq_hybrid_v2 import MultiAgentConfig, run_multi_agent_v2"
new_run = """import importlib
import evaluation.mcq_hybrid_v2 as _mcq_v2
importlib.reload(_mcq_v2)
MultiAgentConfig = _mcq_v2.MultiAgentConfig
run_multi_agent_v2 = _mcq_v2.run_multi_agent_v2"""
if old_run in run_cell.source:
    run_cell.source = run_cell.source.replace(old_run, new_run)
elif "run_multi_agent_v2 = _mcq_v2.run_multi_agent_v2" not in run_cell.source:
    raise RuntimeError("Could not locate the V2 run import")
run_cell.outputs = []
run_cell.execution_count = None
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f"Updated: {path}")
