#!/usr/bin/env python
"""Create a complete V2 repository ZIP without local secrets or caches."""

from __future__ import annotations

from pathlib import Path
import re
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent / "Multi-Agent-AI-for-Reliable-Clinical-Reasoning-V2-optimized.zip"
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", ".ipynb_checkpoints"}
EXCLUDED_NAMES = {".env"}
SECRET_PATTERN = re.compile(r"sk-(?:proj|ant)-[A-Za-z0-9_-]{32,}")


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.name in EXCLUDED_NAMES or path.suffix in {".pyc", ".pyo", ".zip"}:
        return False
    return path.is_file()


def assert_no_embedded_secret(paths: list[Path]) -> None:
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if SECRET_PATTERN.search(text):
            raise RuntimeError(f"Secret-like token found outside .env: {path}")


def main() -> None:
    paths = [path for path in ROOT.rglob("*") if included(path)]
    assert_no_embedded_secret(paths)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, Path(ROOT.name) / path.relative_to(ROOT))

    with zipfile.ZipFile(OUTPUT) as archive:
        names = archive.namelist()
        if any(Path(name).name == ".env" for name in names):
            raise RuntimeError("Packaging invariant failed: .env was included")
    print("Created:", OUTPUT)
    print("Files:", len(paths))
    print("Bytes:", OUTPUT.stat().st_size)


if __name__ == "__main__":
    main()
