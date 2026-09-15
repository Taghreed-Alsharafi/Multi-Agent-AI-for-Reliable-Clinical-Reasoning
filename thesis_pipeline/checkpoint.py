"""Atomic, per-case checkpoint helpers used by resumable runners."""
from __future__ import annotations
import json
from pathlib import Path

def load_completed(path: str|Path) -> dict[str,dict]:
    p=Path(path)
    if not p.exists(): return {}
    out={}
    # Tolerate a truncated / null-padded final line left by an abruptly killed
    # writer: skip any line that is not valid JSON rather than aborting the resume.
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line=line.replace("\x00","").strip()
        if not line: continue
        try:
            row=json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "case_id" in row:
            out[str(row["case_id"])] = row
    return out

def append_checkpoint(path: str|Path, row: dict) -> None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a",encoding="utf-8") as h: h.write(json.dumps(row,default=str)+"\n")

def should_resume(path: str|Path, case_id: str) -> bool:
    return str(case_id) in load_completed(path)
