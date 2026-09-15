from __future__ import annotations
import math, random, re, statistics
from collections import Counter, defaultdict
from typing import Iterable
from .schema import EvaluationCase, PredictionRecord

def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

def token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize(prediction).split(), normalize(reference).split()
    if not pred or not ref: return float(pred == ref)
    overlap=sum((Counter(pred)&Counter(ref)).values())
    if not overlap: return 0.0
    precision,recall=overlap/len(pred),overlap/len(ref)
    return 2*precision*recall/(precision+recall)

def quote_is_grounded(quote: str, documents: list[str]) -> bool:
    candidate=normalize(quote)
    return bool(candidate) and candidate in normalize(" ".join(documents))

def evidence_recall(quotes: list[str], references: list[str]) -> float:
    if not references: return 1.0
    quoted=normalize(" ".join(quotes))
    return sum(normalize(item) in quoted for item in references)/len(references)

def _case_metrics(case: EvaluationCase, pred: PredictionRecord) -> dict[str,float]:
    grounded=[quote_is_grounded(q,case.documents) for q in pred.evidence_quotes]
    precision=sum(grounded)/len(grounded) if grounded else float(not case.reference_evidence)
    safety_correct = (float(pred.is_safe == case.reference_is_safe)
                      if case.reference_is_safe is not None else math.nan)
    return {"exact_match":float(normalize(pred.answer)==normalize(case.reference_answer)),
        "token_f1":token_f1(pred.answer,case.reference_answer),
        "evidence_precision":precision,"hallucinated_evidence_rate":1-precision,
        "evidence_recall":evidence_recall(pred.evidence_quotes,case.reference_evidence),
        "abstention_accuracy":float(pred.insufficient_information==case.should_abstain),
        "safety_accuracy":safety_correct,
        "unsafe_output_rate":float(not pred.is_safe),
        "latency_seconds":pred.latency_seconds,"error_rate":float(pred.status=="error")}

def bootstrap_ci(values:list[float],seed:int=2026,samples:int=2000,
                 confidence_level:float=.95)->tuple[float,float]:
    values=[v for v in values if not math.isnan(v)]
    if not values:return math.nan,math.nan
    rng=random.Random(seed)
    means=sorted(statistics.fmean(rng.choices(values,k=len(values))) for _ in range(samples))
    alpha=(1-confidence_level)/2
    return means[int(alpha*samples)],means[min(int((1-alpha)*samples),samples-1)]

def evaluate_predictions(cases:Iterable[EvaluationCase],predictions:Iterable[PredictionRecord],
                         bootstrap_samples:int=2000,confidence_level:float=.95):
    case_map={c.case_id:c for c in cases}; detailed=[]; by_system=defaultdict(list)
    for pred in predictions:
        if pred.case_id not in case_map: raise ValueError(f"Unknown case {pred.case_id}")
        scores=_case_metrics(case_map[pred.case_id],pred)
        detailed.append({"case_id":pred.case_id,"system":pred.system,**scores});by_system[pred.system].append(scores)
    summary=[]
    for system,rows in sorted(by_system.items()):
        for metric in rows[0]:
            values=[r[metric] for r in rows if not math.isnan(r[metric])]
            if not values: continue
            low,high=bootstrap_ci(values,samples=bootstrap_samples,confidence_level=confidence_level)
            summary.append({"system":system,"metric":metric,"n":len(values),
                "mean":statistics.fmean(values),"ci_low":low,"ci_high":high})
    return detailed,summary

def paired_bootstrap_difference(detailed,system_a,system_b,metric,seed=2026,samples=5000):
    indexed={(str(r["case_id"]),str(r["system"])):float(r[metric]) for r in detailed}
    ids=sorted({i for i,_ in indexed if (i,system_a) in indexed and (i,system_b) in indexed})
    if not ids: raise ValueError("No paired cases found")
    diffs=[indexed[(i,system_a)]-indexed[(i,system_b)] for i in ids]
    diffs=[value for value in diffs if not math.isnan(value)]
    if not diffs: raise ValueError("No non-missing paired values found")
    rng=random.Random(seed)
    draws=sorted(statistics.fmean(rng.choices(diffs,k=len(diffs))) for _ in range(samples))
    return {"mean_difference":statistics.fmean(diffs),"ci_low":draws[int(.025*samples)],
        "ci_high":draws[int(.975*samples)],"probability_a_better":sum(x>0 for x in draws)/samples,
        "n_pairs":len(diffs)}

def robustness_by_variant(cases, predictions):
    groups={c.case_id:c.variant_group for c in cases if c.variant_group}; answers=defaultdict(list)
    for p in predictions:
        if groups.get(p.case_id): answers[(p.system,groups[p.case_id])].append(normalize(p.answer))
    return [{"system":s,"variant_group":g,"consistency":max(Counter(a).values())/len(a),"n":len(a)}
            for (s,g),a in sorted(answers.items())]

def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0

def classification_metrics(cases, predictions, positive_label=None):
    """Accuracy, per-class, macro/weighted F1, and binary safety-style rates."""
    references={c.case_id:c.reference_label for c in cases if c.reference_label is not None}
    systems=defaultdict(list)
    for pred in predictions:
        if pred.case_id in references:
            systems[pred.system].append((references[pred.case_id],pred.predicted_label))
    reports=[]
    for system,pairs in sorted(systems.items()):
        labels=sorted({str(y) for pair in pairs for y in pair if y is not None})
        per_class=[]
        for label in labels:
            tp=sum(y==label and p==label for y,p in pairs)
            fp=sum(y!=label and p==label for y,p in pairs)
            fn=sum(y==label and p!=label for y,p in pairs)
            tn=sum(y!=label and p!=label for y,p in pairs)
            precision=_safe_div(tp,tp+fp);recall=_safe_div(tp,tp+fn)
            f1=_safe_div(2*precision*recall,precision+recall);support=sum(y==label for y,_ in pairs)
            per_class.append({"label":label,"precision":precision,"recall":recall,
                "f1":f1,"support":support,"specificity":_safe_div(tn,tn+fp)})
        total=len(pairs); correct=sum(y==p for y,p in pairs)
        macro_f1=statistics.fmean(x["f1"] for x in per_class) if per_class else 0
        weighted_f1=_safe_div(sum(x["f1"]*x["support"] for x in per_class),total)
        report={"system":system,"n":total,"accuracy":_safe_div(correct,total),
            "macro_f1":macro_f1,"weighted_f1":weighted_f1,"per_class":per_class,
            "confusion_matrix":{"labels":labels,"values":[[sum(y==a and p==b for y,p in pairs)
                for b in labels] for a in labels]}}
        if positive_label:
            positive=next((x for x in per_class if x["label"]==positive_label),None)
            if positive:
                report.update({"sensitivity":positive["recall"],"specificity":positive["specificity"],
                    "positive_predictive_value":positive["precision"]})
        reports.append(report)
    return reports
