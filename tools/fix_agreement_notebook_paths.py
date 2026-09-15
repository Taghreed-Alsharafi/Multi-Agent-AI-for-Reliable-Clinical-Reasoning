"""Make the agreement notebook use valid cross-checkout V2 predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
path = args.root.resolve() / "notebooks" / "Agreement_and_MultiAgent_Evaluation.ipynb"
notebook = nbformat.read(path, as_version=4)
cell = next(item for item in notebook.cells if item.cell_type == "code" and "MULTI_PATH =" in item.source)
start_marker = (
    'RESULTS_DIR = ROOT / "results" / "mcq_v2"'
    if 'RESULTS_DIR = ROOT / "results" / "mcq_v2"' in cell.source
    else 'canonical_name = "Multi-Agent-AI-for-Reliable-Clinical-Reasoning"'
)
start = cell.source.index(start_marker)
end = cell.source.index("DISAGREEMENT_CONFIG =")
replacement = '''canonical_name = "Multi-Agent-AI-for-Reliable-Clinical-Reasoning"
RESULT_CANDIDATES = [
    (ROOT / "results" / "mcq_v2").resolve(),
    (ROOT.parent / "results" / "mcq_v2").resolve(),
    (ROOT.parent / "Multi-Agent-AI-for-Reliable-Clin" / canonical_name / "results" / "mcq_v2").resolve(),
    (ROOT.parent.parent / canonical_name / "results" / "mcq_v2").resolve(),
]

def find_result(filename):
    matches = [directory / filename for directory in RESULT_CANDIDATES if (directory / filename).exists()]
    if not matches:
        raise FileNotFoundError(f"Missing active result: {filename}. Run Notebooks 02 and 03 first.")
    return matches[0]

MULTI_PATH = find_result(f"multi_dynamic_one_gpt5_v2_{RUN_SPLIT}_predictions.csv")
SINGLE_PATH = find_result(f"single_gpt5_{RUN_SPLIT}_predictions.csv")

OUTPUT_ROOT = (ROOT.parent.parent / canonical_name / "results" / "mcq_v2").resolve()
if not (OUTPUT_ROOT.parent.parent / "evaluation" / "agreement_analysis.py").exists():
    OUTPUT_ROOT = (ROOT / "results" / "mcq_v2").resolve()
OUTPUT_DIR = OUTPUT_ROOT / "agreement_analysis" / RUN_SPLIT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
'''
cell.source = cell.source[:start] + replacement + cell.source[end:]
for item in notebook.cells:
    if item.cell_type == "code":
        item.outputs = []
        item.execution_count = None
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f"Updated: {path}")
