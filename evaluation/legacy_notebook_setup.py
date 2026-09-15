"""Idempotent setup for directly maintained legacy MCQ notebooks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "evaluation" / "mcq_hybrid.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root from the current Jupyter directory")


def prepare_legacy_multi_cases(
    csv_path: str | Path,
    run_split: str,
    development_fraction: float,
    split_seed: int,
    max_cases: int | None,
    start: str | Path | None = None,
) -> dict[str, Any]:
    """Load environment, TLS trust, protected split, and selected cases."""
    root = find_project_root(start)
    env_candidates = [
        root / ".env",
        root.parent / ".env",
        root.parent.parent / "Multi-Agent-AI-for-Reliable-Clinical-Reasoning" / ".env",
    ]
    env_path = next((path.resolve() for path in env_candidates if path.exists()), root / ".env")
    load_dotenv(env_path, override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY was not found in {env_path}")

    if os.name == "nt":
        try:
            import truststore
        except ImportError as exc:
            raise RuntimeError("Install Windows TLS support with: python -m pip install truststore") from exc
        truststore.inject_into_ssl()

    from evaluation.mcq_hybrid import ensure_fixed_split

    configured = Path(csv_path)
    candidates = [
        configured if configured.is_absolute() else root / configured,
        configured if configured.is_absolute() else root / "notebooks" / configured,
    ]
    resolved_csv = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
    if resolved_csv is None:
        raise FileNotFoundError("Dataset not found. Checked: " + ", ".join(str(path.resolve()) for path in candidates))

    splits, manifest, manifest_path = ensure_fixed_split(
        resolved_csv, root, development_fraction, split_seed
    )
    if run_split not in splits:
        raise ValueError(f"Unknown split {run_split!r}; expected one of {sorted(splits)}")
    cases = splits[run_split].copy()
    if max_cases is not None:
        cases = cases.head(max_cases).copy()
    if cases.reference_letter.isna().any():
        raise RuntimeError("Some reference answers could not be extracted")
    canonical_root = env_path.parent if (env_path.parent / "evaluation" / "mcq_hybrid.py").exists() else root
    output_dir = canonical_root / "results" / "mcq_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "ROOT": root,
        "env_path": env_path,
        "OUTPUT_DIR": output_dir,
        "csv_path": resolved_csv,
        "splits": splits,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "df": cases,
    }
