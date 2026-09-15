"""Make Notebook 04 discover one complete, compatible V2 result directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
path = args.root.resolve() / "notebooks" / "04_compare_single_multi_mcq.ipynb"
notebook = nbformat.read(path, as_version=4)
config = next(cell for cell in notebook.cells if cell.cell_type == "code" and "STRICT_CASE_MEMBERSHIP" in cell.source)
old = 'OUTPUT_DIR = ROOT / "results" / "mcq_v2"'
new = '''
canonical_name = "Multi-Agent-AI-for-Reliable-Clinical-Reasoning"
result_candidates = [
    ROOT / "results" / "mcq_v2",
    ROOT.parent / "results" / "mcq_v2",
    ROOT.parent.parent / canonical_name / "results" / "mcq_v2",
]
OUTPUT_DIR = (ROOT.parent.parent / canonical_name / "results" / "mcq_v2").resolve()
if not (OUTPUT_DIR.parent.parent / "evaluation" / "mcq_v2_analysis.py").exists():
    OUTPUT_DIR = (ROOT / "results" / "mcq_v2").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Comparison tables output:", OUTPUT_DIR)
'''.strip()
if old in config.source:
    config.source = config.source.replace(old, new)
elif "Shared comparison results:" not in config.source:
    if "Comparison tables output:" not in config.source:
        raise RuntimeError("Could not locate comparison OUTPUT_DIR assignment")

# Always override older generated discovery blocks with a short publication
# output directory. Prediction files are resolved independently below.
if "# Canonical publication output" not in config.source:
    config.source += '''

# Canonical publication output
_canonical_output = (ROOT.parent.parent / canonical_name / "results" / "mcq_v2").resolve()
if (_canonical_output.parent.parent / "evaluation" / "mcq_v2_analysis.py").exists():
    OUTPUT_DIR = _canonical_output
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Publication output directory:", OUTPUT_DIR)
'''

loader = next(cell for cell in notebook.cells if cell.cell_type == "code" and "prediction_paths =" in cell.source)
loader.source = '''
prediction_names = {
    "single_gpt4o_mini_v2": f"single_gpt4o_mini_v2_{RUN_SPLIT}_predictions.csv",
    "single_gpt5": f"single_gpt5_{RUN_SPLIT}_predictions.csv",
    "multi_dynamic_one_gpt5_v2": f"multi_dynamic_one_gpt5_v2_{RUN_SPLIT}_predictions.csv",
}

def find_prediction(name):
    matches = [(directory / name).resolve() for directory in result_candidates if (directory / name).exists()]
    if not matches:
        raise FileNotFoundError(f"Run Notebooks 02 and 03 first. Missing: {name}")
    return matches[0]

prediction_paths = {name: find_prediction(filename) for name, filename in prediction_names.items()}
print("Prediction files:")
for name, path in prediction_paths.items():
    print(f"  {name}: {path}")

systems = {name: pd.read_csv(path, keep_default_na=False) for name, path in prediction_paths.items()}

from evaluation.mcq_v2_analysis import validate_comparison_inputs
compatibility = validate_comparison_inputs(systems)
print("Compatibility:", compatibility)

id_sets = {name: set(frame.fixed_case_id) for name, frame in systems.items()}
common_ids = set.intersection(*id_sets.values())
print("Common paired cases:", len(common_ids))
if STRICT_CASE_MEMBERSHIP:
    assert all(ids == common_ids for ids in id_sets.values()), "System case membership differs"
'''.strip()

# Dataset and QC files may live in the opened nested repository; only results
# need redirection. The V2 split function handles shared Windows-safe manifests.
for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f"Updated: {path}")
