"""Legacy helper retained for compatibility.

The finalized MCQ notebooks are maintained directly:
  02_single_agent_mcq.ipynb
  03_multi_agent_gpt5_judge_mcq.ipynb
  04_compare_single_multi_mcq.ipynb

This script intentionally does not regenerate or overwrite them.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name in [
    "02_single_agent_mcq.ipynb",
    "03_multi_agent_gpt5_judge_mcq.ipynb",
    "04_compare_single_multi_mcq.ipynb",
]:
    path = ROOT / "notebooks" / name
    if not path.exists():
        raise FileNotFoundError(path)
    print("Ready:", path)
