#!/usr/bin/env python
"""Download and format MedQA-USMLE (4 options) as a second clinical-MCQ dataset
for a generalizability test. Distinct source (US USMLE) from final_dataset_1,
with clean built-in gold letters (answer_idx) -- no extraction needed.

Output: data/medqa_usmle_1200_with_gold.csv
Columns: instruction (stem + lettered options), input, gold_letter,
         gold_option_text, source.
"""
from __future__ import annotations
import json, sys, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
DATASET = "GBaker/MedQA-USMLE-4-options"
OUT = ROOT / "data" / "medqa_usmle_1200_with_gold.csv"


def fetch(offset, length):
    url = (f"https://datasets-server.huggingface.co/rows?dataset={DATASET}"
           f"&config=default&split=test&offset={offset}&length={length}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)["rows"]


def main():
    rows = []
    got = 0
    while got < N:
        batch = fetch(got, min(100, N - got))
        if not batch:
            break
        for item in batch:
            r = item["row"]
            opts = r.get("options", {})
            letters = sorted(opts.keys())
            stem = str(r["question"]).strip()
            body = "\n".join(f"{L}. {opts[L]}" for L in letters)
            instruction = f"{stem}\n{body}"
            gold = str(r.get("answer_idx", "")).strip().upper()
            if gold not in letters:
                continue
            rows.append({"instruction": instruction, "input": "",
                         "gold_letter": gold, "gold_option_text": opts.get(gold, ""),
                         "source": "MedQA-USMLE-4opt"})
        got += len(batch)
        print(f"fetched {got} rows ...", flush=True)
    df = pd.DataFrame(rows).drop_duplicates(subset=["instruction"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT} with {len(df)} rows")
    print("gold distribution:", df["gold_letter"].value_counts().to_dict())
    print("sample instruction:\n", df.iloc[0]["instruction"][:300])


if __name__ == "__main__":
    main()
