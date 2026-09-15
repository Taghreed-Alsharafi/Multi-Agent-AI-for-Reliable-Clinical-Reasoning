"""Shared analysis for the two-dataset comparison (no API calls).

Builds, for each dataset, a parallel four-system view — Single-GPT, Single-Claude,
Panel (vote-only), Improved (panel+judge) — with performance metrics, an
errors+abstentions safety table, and paired McNemar statistics.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from thesis_pipeline.data import initialize_dataset
from thesis_pipeline.evaluation import score, metrics
from thesis_pipeline.multi_agent import decide, _weighted_vote

# Systems reported in the main comparison (the vote-only panel is computed
# internally for the adjudication-mechanism analysis but is not presented as a
# competing system).
SYSTEM_ORDER = ["single_gpt", "single_claude", "improved_multi"]
SYSTEM_LABELS = {"single_gpt": "Single-agent GPT", "single_claude": "Single-agent Claude",
                 "panel_vote": "Panel (vote-only)", "improved_multi": "Multi-Agent framework (proposed)"}


def _load_jsonl(path):
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.replace("\x00", "").strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and "case_id" in r:
            out[str(r["case_id"])] = r
    return out


def _eval_frame(dataset_path):
    b = initialize_dataset({"name": "d", "path": dataset_path,
                            "columns": {"instruction": "instruction", "input": "input", "gold": "gold_letter"}},
                           ROOT, 42)
    return b


def _single_preds(records):
    return pd.DataFrame([{"case_id": c, "prediction": r.get("prediction", ""),
                          "status": r.get("status", "success"),
                          "critical_safety_error": False} for c, r in records.items()])


def _panel_preds(records, trigger):
    rows = []
    for c, r in records.items():
        outs = r.get("agent_outputs", [])
        allowed = sorted({o.get("answer") for o in outs if o.get("answer")})
        pred, meta = decide(outs, allowed, r.get("judge_letter", ""), trigger)
        rows.append({"case_id": c, "prediction": pred,
                     "status": "success" if pred else "api_error",
                     "critical_safety_error": False,
                     "judge_used": meta.get("judge_used", False)})
    return pd.DataFrame(rows)


def _safety(frame_cases, preds, records=None, is_panel=False):
    """errors + abstentions safety metrics for one system."""
    s = score(frame_cases, preds)
    n = len(s)
    error_rate = float((~s.correct).mean())
    abstain = float((s.prediction.astype(str).str.strip() == "").mean())
    if is_panel and records:
        # mean fraction of panel agents that failed to produce a valid answer
        fracs = []
        for c in s.case_id:
            outs = records.get(str(c), {}).get("agent_outputs", [])
            if outs:
                fracs.append(sum(1 for o in outs if not o.get("answer")) / len(outs))
        invalid = float(sum(fracs) / len(fracs)) if fracs else 0.0
    else:
        st = preds.set_index("case_id")["status"] if "status" in preds else None
        invalid = float((preds["status"].isin(["parse_error", "api_error"])).mean()) if "status" in preds else 0.0
    return {"n": n, "error_rate": error_rate, "abstention_rate": abstain, "invalid_rate": invalid}


def per_class_metrics(cases, preds):
    """Per-label precision/recall/F1/support from a predictions frame."""
    s = score(cases, preds)
    labels = sorted(set(s.gold.astype(str)))
    rows = []
    for lab in labels:
        tp = int(((s.gold == lab) & (s.prediction.astype(str).str.upper() == lab)).sum())
        fp = int(((s.gold != lab) & (s.prediction.astype(str).str.upper() == lab)).sum())
        fn = int(((s.gold == lab) & (s.prediction.astype(str).str.upper() != lab)).sum())
        support = int((s.gold == lab).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"label": lab, "precision": prec, "recall": rec, "f1": f1, "support": support})
    return rows


def cohen_kappa(letters_a, letters_b):
    pairs = [(a, b) for a, b in zip(letters_a, letters_b) if a and b]
    if not pairs:
        return float("nan")
    n = len(pairs)
    cats = sorted({x for p in pairs for x in p})
    po = sum(1 for a, b in pairs if a == b) / n
    pe = 0.0
    for c in cats:
        pa = sum(1 for a, _ in pairs if a == c) / n
        pb = sum(1 for _, b in pairs if b == c) / n
        pe += pa * pb
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def panel_behavior(records, gold):
    """Distinct-answer distribution + accuracy when unanimous vs on disagreement,
    and the judge's win-rate over vote-only on the disagreement subset."""
    dist = {}
    unan_n = unan_correct = 0
    dis_n = vote_correct = judge_correct = improved_correct = 0
    for c, r in records.items():
        g = str(gold.get(str(c), "")).upper()
        if not g:
            continue
        outs = r.get("agent_outputs", [])
        allowed = sorted({o.get("answer") for o in outs if o.get("answer")})
        vote, _, n_distinct = _weighted_vote(outs)
        dist[n_distinct] = dist.get(n_distinct, 0) + 1
        if n_distinct <= 1:
            unan_n += 1; unan_correct += int(str(vote).upper() == g)
        else:
            dis_n += 1
            vote_correct += int(str(vote).upper() == g)
            jl = str(r.get("judge_letter", "")).upper()
            if jl:
                judge_correct += int(jl == g)
            improved, _ = decide(outs, allowed, r.get("judge_letter", ""), 2)
            improved_correct += int(str(improved).upper() == g)
    return {
        "distinct_distribution": dist,
        "unanimous_n": unan_n, "unanimous_accuracy": unan_correct / unan_n if unan_n else float("nan"),
        "disagreement_n": dis_n,
        "vote_accuracy_on_disagreement": vote_correct / dis_n if dis_n else float("nan"),
        "judge_accuracy_on_disagreement": judge_correct / dis_n if dis_n else float("nan"),
        "improved_accuracy_on_disagreement": improved_correct / dis_n if dis_n else float("nan"),
    }


def mcnemar(a_correct, b_correct):
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n))
    return {"b": b, "c": c, "p_value": p}


def analyze_dataset(dataset_path, single_gpt_jsonl, single_claude_jsonl, panel_jsonl, trigger=2):
    b = _eval_frame(dataset_path)
    cases = b.frame[b.frame.eligible_for_evaluation][["case_id", "gold", "instruction"]].copy()
    cases["case_id"] = cases["case_id"].astype(str)

    gpt_r = _load_jsonl(single_gpt_jsonl)
    claude_r = _load_jsonl(single_claude_jsonl)
    panel_r = _load_jsonl(panel_jsonl)

    # Evaluate only on cases actually scored by every system (dataset 1's
    # checkpoints cover the test split; dataset 2's cover all items).
    common = set(gpt_r) & set(claude_r) & set(panel_r)
    cases = cases[cases.case_id.isin(common)].copy()
    gold = {str(r.case_id): str(r.gold).upper() for _, r in cases.iterrows()}

    preds = {
        "single_gpt": _single_preds(gpt_r),
        "single_claude": _single_preds(claude_r),
        "panel_vote": _panel_preds(panel_r, trigger=99),
        "improved_multi": _panel_preds(panel_r, trigger=trigger),
    }
    perf, safety, correctness = {}, {}, {}
    for name, pr in preds.items():
        s = score(cases, pr)
        m = metrics(s, name)
        if name == "improved_multi":
            jr = _panel_preds(panel_r, trigger)["judge_used"].mean() if panel_r else 0.0
            m["judge_rate"] = float(jr)
        perf[name] = m
        is_panel = name in ("panel_vote", "improved_multi")
        safety[name] = _safety(cases, pr, panel_r, is_panel)
        cmap = dict(zip(s.case_id.astype(str), s.correct))
        correctness[name] = cmap

    ids = [c for c in gold if all(c in correctness[n] for n in preds)]
    stats = {}
    for name in ["single_gpt", "single_claude"]:
        ic = [bool(correctness["improved_multi"][c]) for c in ids]
        bc = [bool(correctness[name][c]) for c in ids]
        mc = mcnemar(ic, bc)
        acc_i = sum(ic) / len(ic); acc_b = sum(bc) / len(bc)
        stats[name] = {"delta": acc_i - acc_b, **mc}

    # predicted-letter maps (for inter-system agreement)
    pred_letters = {}
    for name, pr in preds.items():
        pred_letters[name] = {str(r.case_id): str(r.prediction).upper() for _, r in pr.iterrows()}
    # pairwise Cohen's kappa on predicted letters
    kappa = {}
    for i, a in enumerate(SYSTEM_ORDER):
        for b in SYSTEM_ORDER[i + 1:]:
            la = [pred_letters[a].get(c, "") for c in ids]
            lb = [pred_letters[b].get(c, "") for c in ids]
            kappa[(a, b)] = cohen_kappa(la, lb)
    # per-class metrics
    per_class = {name: per_class_metrics(cases, pr) for name, pr in preds.items()}
    # panel behavior + judge effectiveness
    panel = panel_behavior(panel_r, gold)

    return {"cases": cases, "gold": gold, "perf": perf, "safety": safety,
            "correctness": correctness, "ids": ids, "stats": stats,
            "pred_letters": pred_letters, "kappa": kappa, "per_class": per_class,
            "panel": panel, "n": len(cases)}


if __name__ == "__main__":
    d1 = analyze_dataset("data/final_dataset_1_with_gold.csv",
                         ROOT / "results/final_dataset_1/single_gpt_test.jsonl",
                         ROOT / "results/final_dataset_1/single_claude_test.jsonl",
                         ROOT / "results/improved/improved_test.jsonl")
    print("DATASET 1 (final_dataset_1, test) n=", d1["ids"].__len__())
    for name in SYSTEM_ORDER:
        m = d1["perf"][name]; s = d1["safety"][name]
        print(f"  {SYSTEM_LABELS[name]:<24} acc={m['accuracy']:.4f} f1={m['macro_f1']:.4f} "
              f"err={s['error_rate']:.4f} abst={s['abstention_rate']:.4f} inval={s['invalid_rate']:.4f}")
    print("  stats vs improved:", {k: (round(v['delta'], 4), round(v['p_value'], 4)) for k, v in d1["stats"].items()})
