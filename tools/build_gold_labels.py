#!/usr/bin/env python
"""Build a reliable gold-answer column for the chain-of-thought MCQ dataset.

`data/final_dataset_1.csv` stores, for each question, a free-text `output`
explanation that concludes with the correct option *in prose* (by option text,
not by letter). There is no clean embedded gold letter, so we recover it by
*extraction* (reading which option the explanation endorses), never by solving
the question ourselves.

Method (per row):
  1. deterministic parser  -- explicit "answer is X" / conclusion text match;
  2. LLM pass 1 (cheap)    -- extract the letter the explanation concludes;
  3. LLM pass 2 (stronger) -- ONLY when pass 1 and the deterministic parser
                              disagree, or pass 1 is empty; used as a tiebreak.

Output: data/final_dataset_1_with_gold.csv (all rows, original columns kept) plus
a build summary. The run is checkpointed and resumable and never prints the key.

Usage:
    python tools/build_gold_labels.py                # all rows
    python tools/build_gold_labels.py --limit 20     # smoke test
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from evaluation.mcq_hybrid import parse_options
from thesis_pipeline.data import _derive_gold_letter

CHEAP_MODEL = os.environ.get("GOLD_CHEAP_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
STRONG_MODEL = os.environ.get("GOLD_STRONG_MODEL", "gpt-4.1")
CONCURRENCY = int(os.environ.get("GOLD_CONCURRENCY", "8"))

SYSTEM_PROMPT = (
    "You extract an answer; you do NOT solve the question. You are given a medical "
    "multiple-choice question, its lettered options, and an explanation that was "
    "already written for it. Report which SINGLE option letter the EXPLANATION "
    "concludes is correct, based only on what the explanation argues for -- not your "
    "own medical opinion. Prefer an explicit concluding statement (e.g. 'the answer "
    "is ...'). If the explanation is genuinely ambiguous or endorses none of the "
    "listed options, return null. Return only JSON: {\"letter\": \"A\"} (or null)."
)


def _client():
    import httpx
    from openai import AsyncOpenAI
    verify = os.environ.get("VERIFY_SSL", "true").strip().lower() not in {"false", "0", "no", "off"}
    http_client = None if verify else httpx.AsyncClient(verify=False, timeout=60)
    return AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"],
                       **({"http_client": http_client} if http_client else {}))


def options_for(instruction: str) -> dict[str, str]:
    try:
        o = parse_options(instruction) or {}
    except Exception:
        o = {}
    if not o:
        o = dict(re.findall(r"(?m)^[ \t]*([A-H])[.)/:][ \t]*([^\r\n]+)", instruction))
    return {str(k).upper(): str(v).strip() for k, v in o.items()}


async def extract_letter(client, model, instruction, explanation, allowed) -> str:
    stem = instruction.split("\n")[0]
    if not stem.strip():
        stem = instruction[:400]
    block = "\n".join(f"{k}. {options_for(instruction)[k]}" for k in allowed)
    user = (f"## Question stem\n{stem[:600]}\n\n## Options\n{block}\n\n"
            f"## Explanation (verbatim)\n{explanation[-3800:]}")
    resp = await client.chat.completions.create(
        model=model, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}])
    try:
        letter = json.loads(resp.choices[0].message.content or "{}").get("letter")
    except Exception:
        letter = None
    letter = str(letter).strip().upper() if letter else ""
    return letter if letter in allowed else ""


async def process_row(client, sem, idx, row):
    async with sem:
        instruction = str(row["instruction"]); explanation = str(row["output"])
        opts = options_for(instruction); allowed = sorted(opts.keys())
        if not allowed:
            return {"row": idx, "gold_letter": "", "gold_option_text": "", "n_options": 0,
                    "deterministic": "", "llm_pass1": "", "llm_pass2": "",
                    "method": "no_options", "confidence": "excluded"}
        deterministic = _derive_gold_letter(instruction, explanation)
        pass1 = await extract_letter(client, CHEAP_MODEL, instruction, explanation, allowed)
        pass2 = ""
        if pass1 and deterministic and pass1 == deterministic:
            gold, method, conf = pass1, "llm+deterministic_agree", "high"
        else:
            # Ambiguous: bring in the stronger model as a tiebreak/confirmation.
            pass2 = await extract_letter(client, STRONG_MODEL, instruction, explanation, allowed)
            if pass2 and pass2 == pass1:
                gold, method, conf = pass2, "llm_two_model_agree", "high"
            elif pass2 and pass2 == deterministic:
                gold, method, conf = pass2, "strong_llm+deterministic_agree", "high"
            elif pass2:
                gold, method, conf = pass2, "strong_llm_resolved", "medium"
            elif pass1:
                gold, method, conf = pass1, "cheap_llm_only", "medium"
            elif deterministic:
                gold, method, conf = deterministic, "deterministic_only", "low"
            else:
                gold, method, conf = "", "unresolved", "excluded"
        return {"row": idx, "gold_letter": gold,
                "gold_option_text": opts.get(gold, ""), "n_options": len(allowed),
                "deterministic": deterministic, "llm_pass1": pass1, "llm_pass2": pass2,
                "method": method, "confidence": conf}


def load_checkpoint(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line); out[int(r["row"])] = r
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data" / "final_dataset_1.csv"))
    ap.add_argument("--output", default=str(ROOT / "data" / "final_dataset_1_with_gold.csv"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input).fillna("")
    if args.limit:
        df = df.head(args.limit)
    ck_dir = ROOT / "results" / "gold_build"; ck_dir.mkdir(parents=True, exist_ok=True)
    ck_path = ck_dir / "gold_checkpoint.jsonl"
    done = load_checkpoint(ck_path)
    print(f"Rows: {len(df)} | already done: {len(done)} | cheap={CHEAP_MODEL} strong={STRONG_MODEL}")

    client = _client(); sem = asyncio.Semaphore(CONCURRENCY)
    todo = [(i, df.iloc[i]) for i in range(len(df)) if i not in done]
    with ck_path.open("a", encoding="utf-8") as ck:
        for chunk_start in range(0, len(todo), 50):
            chunk = todo[chunk_start:chunk_start + 50]
            results = await asyncio.gather(*[process_row(client, sem, i, r) for i, r in chunk])
            for res in results:
                done[res["row"]] = res
                ck.write(json.dumps(res) + "\n")
            ck.flush()
            print(f"  processed {min(chunk_start + 50, len(todo))}/{len(todo)} remaining rows")

    # Assemble output in original row order.
    records = [done[i] for i in range(len(df))]
    gold = pd.DataFrame(records)
    out = df.reset_index(drop=True).copy()
    out["gold_letter"] = gold["gold_letter"].values
    out["gold_option_text"] = gold["gold_option_text"].values
    out["gold_extraction_method"] = gold["method"].values
    out["gold_confidence"] = gold["confidence"].values
    out["n_options"] = gold["n_options"].values
    out.to_csv(args.output, index=False)

    n = len(out); labeled = int((out["gold_letter"].astype(str).str.fullmatch(r"[A-H]")).sum())
    summary = {
        "total_rows": n, "labeled_rows": labeled, "excluded_rows": n - labeled,
        "by_confidence": out["gold_confidence"].value_counts().to_dict(),
        "by_method": out["gold_extraction_method"].value_counts().to_dict(),
        "label_distribution": out.loc[out.gold_letter != "", "gold_letter"].value_counts().to_dict(),
        "cheap_model": CHEAP_MODEL, "strong_model": STRONG_MODEL, "output": args.output,
    }
    (ck_dir / "gold_build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== GOLD BUILD SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
