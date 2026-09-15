"""Repair project-root, .env, and Windows TLS setup in the legacy notebook."""

from __future__ import annotations

from pathlib import Path
import argparse
import re
import textwrap

import nbformat


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
ROOT = parser.parse_args().root.resolve()
NOTEBOOK = ROOT / "notebooks" / "03_multi_agent_development_mcq.ipynb"

replacement = textwrap.dedent(
    r'''
    from pathlib import Path
    import os, sys
    import pandas as pd
    from IPython.display import display
    _start = Path.cwd().resolve()
    ROOT = next(
        (candidate for candidate in [_start, *_start.parents]
         if (candidate / "evaluation" / "legacy_notebook_setup.py").exists()),
        None,
    )
    if ROOT is None:
        raise FileNotFoundError("Could not locate the repository root")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import importlib
    import evaluation.legacy_notebook_setup as _legacy_setup
    importlib.reload(_legacy_setup)
    prepare_legacy_multi_cases = _legacy_setup.prepare_legacy_multi_cases
    from evaluation.mcq_hybrid import ensure_fixed_split, run_multi_agent, summarize_system

    setup = prepare_legacy_multi_cases(
        CSV_PATH, RUN_SPLIT, DEVELOPMENT_FRACTION, SPLIT_SEED, MAX_CASES
    )
    globals().update(setup)
    print("OpenAI API key loaded from:", env_path)

    print("Fixed split manifest:", manifest_path)
    print("Total / development / test:", manifest["n_total"], manifest["n_development"], manifest["n_test"])
    print("Running split/cases:", RUN_SPLIT, len(df))
    print("Models:")
    print("  Router/Specialists/Critic/Auditor:", ROUTER_MODEL)
    print("  EXACTLY ONE strong Judge call:", JUDGE_MODEL)
    if df.reference_letter.isna().any():
        raise RuntimeError("Some reference answers could not be extracted. Fix these before evaluation.")
    display(df[["fixed_case_id", "instruction", "reference_letter"]].head())
    '''
).strip()

notebook = nbformat.read(NOTEBOOK, as_version=4)
config_cell = next(
    (cell for cell in notebook.cells if cell.cell_type == "code" and "MAX_TOKENS_JUDGE" in cell.source and "RUN_SPLIT" in cell.source),
    None,
)
if config_cell is None:
    raise RuntimeError("Could not find the legacy configuration cell")
config_cell.source = re.sub(
    r"^MAX_CASES\s*=.*$",
    "MAX_CASES = None             # all 60 development cases; test remains locked",
    config_cell.source,
    flags=re.MULTILINE,
)
config_cell.source = re.sub(
    r"^MAX_TOKENS_JUDGE\s*=.*$",
    "MAX_TOKENS_JUDGE = 4000",
    config_cell.source,
    flags=re.MULTILINE,
)
config_cell.source = re.sub(
    r"^RESUME\s*=.*$",
    "RESUME = False              # fresh run after changing token/config settings",
    config_cell.source,
    flags=re.MULTILINE,
)
config_cell.outputs = []
config_cell.execution_count = None
target = next(
    (
        cell for cell in notebook.cells
        if cell.cell_type == "code" and (
            "Set OPENAI_API_KEY before running this notebook" in cell.source
            or "OpenAI API key loaded from:" in cell.source
        )
    ),
    None,
)
if target is None:
    raise RuntimeError("Could not find the legacy setup cell to repair")
target.source = replacement
target.outputs = []
target.execution_count = None

run_cell = next(
    (cell for cell in notebook.cells if cell.cell_type == "code" and "results = await run_multi_agent(" in cell.source),
    None,
)
if run_cell is None:
    raise RuntimeError("Could not find the legacy model-run cell")
run_body = run_cell.source
marker = "# Self-healing prerequisite setup"
if marker in run_body:
    run_body = run_body.split("# End prerequisite setup", 1)[1].lstrip()
prerequisite = textwrap.dedent(
    r'''
    # Refresh prerequisites every run so stale kernels cannot retain an unsafe path.
    if True:
        from pathlib import Path
        import sys
        _start = Path.cwd().resolve()
        _root = next(
            (candidate for candidate in [_start, *_start.parents]
             if (candidate / "evaluation" / "legacy_notebook_setup.py").exists()),
            None,
        )
        if _root is None:
            raise FileNotFoundError("Could not locate the repository root")
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        import importlib
        import evaluation.legacy_notebook_setup as _legacy_setup
        importlib.reload(_legacy_setup)
        prepare_legacy_multi_cases = _legacy_setup.prepare_legacy_multi_cases
        from evaluation.mcq_hybrid import run_multi_agent
        globals().update(prepare_legacy_multi_cases(
            CSV_PATH, RUN_SPLIT, DEVELOPMENT_FRACTION, SPLIT_SEED, MAX_CASES, _start
        ))
        _prediction_name = f"{RUN_NAME}_{RUN_SPLIT}_predictions.csv"
        if len(str(OUTPUT_DIR / _prediction_name)) >= 248:
            OUTPUT_DIR = Path(env_path).parent / "results" / "mcq_comparison"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if len(str(OUTPUT_DIR / _prediction_name)) >= 248:
            raise RuntimeError(f"Output path is still too long for Windows: {OUTPUT_DIR}")
        print("Prerequisites refreshed; cases:", len(df), "output:", OUTPUT_DIR)
    # End prerequisite setup
    '''
).strip()
run_cell.source = prerequisite + "\n\n" + run_body
run_cell.outputs = []
run_cell.execution_count = None
nbformat.validate(notebook)
nbformat.write(notebook, NOTEBOOK)
print(f"Updated: {NOTEBOOK}")
