from __future__ import annotations
from collections import Counter
import pandas as pd
from .statistics import bootstrap_ci, paired_difference, exact_mcnemar

def score(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    x = frame[["case_id", "gold"]].merge(predictions, on="case_id", how="left")
    x["correct"] = x["prediction"].astype(str).str.upper() == x["gold"].astype(str).str.upper()
    status = x["status"] if "status" in x.columns else pd.Series("success", index=x.index)
    safety = x["critical_safety_error"] if "critical_safety_error" in x.columns else pd.Series(False, index=x.index)
    x["success"] = status.astype(str).eq("success") & x["prediction"].astype(str).str.strip().ne("")
    x["critical_safety_error"] = safety.fillna(False).astype(bool)
    return x

def metrics(scored: pd.DataFrame, system: str = "system") -> dict:
    if scored.empty: return {"system": system, "n": 0, "accuracy": float("nan"), "critical_safety_error_rate": float("nan"), "macro_f1": float("nan")}
    labels = sorted(set(scored.gold.astype(str))); precision=[]; recall=[]; f1=[]
    for label in labels:
        tp=((scored.gold==label)&(scored.prediction.astype(str).str.upper()==label.upper())).sum(); fp=((scored.gold!=label)&(scored.prediction.astype(str).str.upper()==label.upper())).sum(); fn=((scored.gold==label)&(scored.prediction.astype(str).str.upper()!=label.upper())).sum()
        precision.append(tp/(tp+fp) if tp+fp else 0); recall.append(tp/(tp+fn) if tp+fn else 0); f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    acc=float(scored.correct.mean()); ci=bootstrap_ci(scored.correct.astype(float), samples=2000)
    safety=float(scored.critical_safety_error.mean()); sci=bootstrap_ci(scored.critical_safety_error.astype(float), samples=2000)
    failures=int((~scored.success).sum()) if "success" in scored else 0
    # Accuracy restricted to cases that produced a usable model response. This
    # separates genuine wrong answers from API/parse failures: a provider whose
    # every call failed (e.g. an invalid key) shows accuracy 0.0 but
    # accuracy_successful_only NaN, so the failure is not mistaken for the model
    # answering everything incorrectly.
    ok=scored[scored.success] if "success" in scored else scored
    acc_ok=float(ok.correct.mean()) if len(ok) else float("nan")
    return {"system":system,"n":len(scored),"successful_n":len(scored)-failures,"failure_count":failures,"failure_rate":failures/len(scored) if len(scored) else float("nan"),"accuracy":acc,"accuracy_successful_only":acc_ok,"accuracy_ci_low":ci[0],"accuracy_ci_high":ci[1],"critical_safety_error_rate":safety,"critical_safety_error_ci_low":sci[0],"critical_safety_error_ci_high":sci[1],"precision":float(sum(precision)/len(precision)),"recall":float(sum(recall)/len(recall)),"macro_f1":float(sum(f1)/len(f1)),"f1_score":float(sum(f1)/len(f1))}

def final_comparison(scored: dict[str, pd.DataFrame]) -> dict:
    systems = {name: metrics(frame, name) for name, frame in scored.items()}; base = next(iter(scored))
    comparisons=[]
    for name, frame in scored.items():
        if name == base: continue
        joined=scored[base][["case_id","correct","critical_safety_error"]].merge(frame[["case_id","correct","critical_safety_error"]], on="case_id", suffixes=("_base","_candidate"))
        comparisons.append({"candidate":name,"baseline":base,"accuracy":paired_difference(joined.correct_candidate, joined.correct_base),"mcnemar":exact_mcnemar(joined.correct_candidate, joined.correct_base),"critical_safety_error_difference":paired_difference(joined.critical_safety_error_candidate, joined.critical_safety_error_base)})
    return {"systems":systems,"comparisons":comparisons}
