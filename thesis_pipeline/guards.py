from __future__ import annotations
import pandas as pd

def assert_identical_final_inputs(frames: dict[str,pd.DataFrame]) -> None:
    if not frames: raise ValueError("No final systems supplied")
    first=next(iter(frames.values()))
    ids=set(first.case_id.astype(str)); gold=dict(zip(first.case_id.astype(str),first.gold.astype(str)))
    for name,frame in frames.items():
        if set(frame.case_id.astype(str)) != ids: raise ValueError(f"Final systems do not share identical test case IDs: {name}")
        if dict(zip(frame.case_id.astype(str),frame.gold.astype(str))) != gold: raise ValueError(f"Final systems do not share identical gold labels: {name}")
