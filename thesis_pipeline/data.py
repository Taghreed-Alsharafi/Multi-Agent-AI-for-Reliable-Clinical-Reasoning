from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
import pandas as pd
import re
from difflib import SequenceMatcher

def _hash(value: str) -> str: return hashlib.sha256(value.encode()).hexdigest()

def _derive_gold_letter(question: str, answer: str) -> str:
    """Derive a single-letter label from a verbose chain-of-thought answer.

    Two cases are handled:
      * an explicit ``final answer`` marker -> match the text right after it;
      * no marker -> match the *concluding* sentence(s) of the reasoning.

    The earlier version only matched the first line of the last 1200 characters,
    which is mid-reasoning rather than the conclusion, so verbose answers without
    a literal "final answer" were either mis-mapped or left as the full paragraph.
    Matching is deliberately conservative: when no option scores above the
    threshold the function returns "" so the row is excluded rather than assigned
    a fabricated gold label.
    """
    answer = answer or ""
    try:
        from evaluation.mcq_hybrid import parse_options
        parsed=parse_options(question)
        options={str(k).upper():str(v) for k,v in parsed.items()}
    except Exception:
        options=dict(re.findall(r"(?m)^[ \t]*([A-H])[.)/:][ \t]*([^\r\n]+)", question))
    marker=answer.lower().rfind("final answer") if answer else -1
    if marker >= 0:
        tail=answer[marker+len("final answer"):].strip(" :#-\n\r")
        nonempty_lines=[line.strip() for line in tail.splitlines() if line.strip()]
        answer_line=nonempty_lines[0] if nonempty_lines else tail.strip()
    else:
        tail=answer[-700:]
        nonempty_lines=[line.strip() for line in tail.splitlines() if line.strip()]
        answer_line=" ".join(nonempty_lines[-2:]) if nonempty_lines else tail.strip()
    if not options or not tail: return ""
    answer_line=re.sub(r"[*_`]", "", answer_line)
    scores=[]
    for letter,text in options.items():
        a=re.sub(r"[^a-z0-9 ]"," ",text.lower()); b=re.sub(r"[^a-z0-9 ]"," ",answer_line.lower()); full=re.sub(r"[^a-z0-9 ]"," ",tail[-1000:].lower())
        at=set(a.split()); bt=set(b.split()); token_score=len(at & bt)/len(at) if at else 0.0
        substring=1.0 if a.strip() and a.strip() in b else (0.6 if a.strip() and a.strip() in full else 0.0)
        score=max(SequenceMatcher(None,a,b).ratio(), token_score, substring)
        scores.append((score,letter))
    scores.sort(reverse=True)
    # Require a clear winner: the top option must beat the runner-up, guarding
    # against near-ties where the conclusion mentions several options.
    if len(scores) >= 2 and scores[0][0] - scores[1][0] < 0.10 and scores[0][0] < 0.9:
        return ""
    return scores[0][1].upper() if scores and scores[0][0] >= .34 else ""

@dataclass
class DatasetBundle:
    frame: pd.DataFrame; dataset_name: str; dataset_path: str; dataset_fingerprint: str; label_fingerprint: str; splits: dict[str,pd.DataFrame]; validation: dict
    def cases(self, split: str) -> pd.DataFrame:
        if split not in self.splits: raise ValueError(f"Unknown split {split!r}")
        return self.splits[split].copy()

def initialize_dataset(spec: dict, root: str|Path=".", seed: int=42) -> DatasetBundle:
    path=Path(spec["path"]); path=path if path.is_absolute() else Path(root)/path
    if not path.exists(): raise FileNotFoundError(path)
    cols=spec.get("columns",{}); id_col=cols.get("id"); q_col=cols.get("instruction","instruction"); input_col=cols.get("input","input"); gold_col=cols.get("gold","gold")
    raw=pd.read_json(path,lines=path.suffix.lower()==".jsonl") if path.suffix.lower() in {".json",".jsonl"} else pd.read_csv(path)
    missing=[c for c in (id_col,q_col,input_col,gold_col) if c and c not in raw.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    f=raw.copy(); f["case_id"]=f[id_col].astype(str).str.strip() if id_col else ""
    blank=f.case_id.isin({"","nan","None"}); f.loc[blank,"case_id"]=[_hash(f"{q}\x1f{i}") for q,i in zip(f.loc[blank,q_col],f.loc[blank,input_col])]
    if f.case_id.duplicated().any(): raise ValueError("Case IDs must be unique")
    if f[[q_col,input_col]].duplicated().any(): raise ValueError("Duplicate cases detected")
    f["instruction"]=f[q_col].fillna("").astype(str); f["input"]=f[input_col].fillna("").astype(str)
    # Preserve the original reference text so derivation always sees the full answer.
    f["gold_source"]=f[gold_col].astype(str)
    f["gold"]=f[gold_col].astype(str).str.strip()
    # Any gold that is not already a clean single option letter (A-H) is treated as
    # a verbose reference answer and run through derivation. This covers blank/NaN
    # golds, chain-of-thought answers with a "final answer" marker, AND those
    # without one -- the previous code only re-derived the marker case, leaving
    # ~half the rows with an entire reasoning paragraph stored as the "label".
    is_clean_letter=f["gold"].str.fullmatch(r"[A-Ha-h]")
    needs_derivation=~is_clean_letter
    if needs_derivation.any():
        f.loc[needs_derivation,"gold"]=[_derive_gold_letter(q,a) for q,a in zip(f.loc[needs_derivation,"instruction"],f.loc[needs_derivation,"gold_source"])]
    f["gold"]=f["gold"].astype(str).str.strip().str.upper()
    # A row is scorable only when its gold is exactly one supplied option letter.
    # A leftover paragraph, blank, or an unresolved derivation is excluded and
    # reported -- never silently scored as an automatic wrong answer.
    unparseable=~f["gold"].str.fullmatch(r"[A-H]")
    f["eligible_for_evaluation"]=(~unparseable).astype(bool)
    eligible=f[f.eligible_for_evaluation].copy()
    fingerprint=_hash("\n".join(sorted(f.case_id.astype(str)))); labels=_hash("\n".join(f"{i}:{g}" for i,g in sorted(zip(eligible.case_id,eligible.gold.astype(str)))))
    split_col=cols.get("split","split")
    if split_col in f: f["split"]=f[split_col].astype(str).str.lower().replace({"dev":"development","validation":"development"})
    else:
        f["split"]=f.case_id.map(lambda x:"development" if int(_hash(f"{x}:{seed}")[:8],16)/0xffffffff<.2 else "test")
        if len(eligible) > 1 and "development" not in set(f.loc[eligible.index, "split"]): f.loc[eligible.index[0], "split"] = "development"
        if len(eligible) > 1 and "test" not in set(f.loc[eligible.index, "split"]): f.loc[eligible.index[-1], "split"] = "test"
    if not {"development","test"}.issubset(set(f.split)) and len(eligible) > 1: raise ValueError("Dataset must contain development and test splits")
    validation={"file_exists":True,"required_columns":True,"unique_case_ids":True,"missing_or_unparseable_gold_answers":int(unparseable.sum()),"eligible_cases":int(len(eligible)),"excluded_cases":int(len(f)-len(eligible)),"duplicate_cases":0,"dataset_fingerprint":fingerprint,"label_fingerprint":labels,"class_distribution":eligible.gold.value_counts().to_dict()}
    return DatasetBundle(f,spec.get("name",path.stem),str(path),fingerprint,labels,{s:f[(f.split==s)&f.eligible_for_evaluation].copy() for s in ("development","test")},validation)
