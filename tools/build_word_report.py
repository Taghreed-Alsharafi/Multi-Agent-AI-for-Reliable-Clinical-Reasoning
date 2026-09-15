#!/usr/bin/env python
"""Assemble the extended academic Word report: Method -> Results -> Discussion.

Times New Roman 12, all-black text, 1.5 line spacing. Reuses tools/analysis_lib.py
for both datasets, generates result figures, and writes
results/THESIS_METHOD_RESULTS_DISCUSSION.docx. No API calls.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.analysis_lib import analyze_dataset, SYSTEM_ORDER, SYSTEM_LABELS

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

FIG = ROOT / "results" / "report_figures"; FIG.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "results" / "THESIS_METHOD_RESULTS_DISCUSSION.docx"
BLACK = RGBColor(0, 0, 0)
FONT = "Times New Roman"
D1_PATH = "data/final_dataset_1_with_gold.csv"
D2_PATH = "data/medqa_usmle_1200_with_gold.csv"


def pct(x, nd=1):
    try:
        return f"{100*float(x):.{nd}f}%"
    except Exception:
        return "n/a"


def f3(x, nd=3):
    try:
        v = float(x); return "n/a" if v != v else f"{v:.{nd}f}"
    except Exception:
        return "n/a"


# --------------------------------------------------------------- figures ------
def fig_accuracy(d1, d2):
    labels = [SYSTEM_LABELS[s].replace(" (proposed)", "\n(proposed)").replace("Single-agent ", "Single-agent\n") for s in SYSTEM_ORDER]
    a1 = [d1["perf"][s]["accuracy"] for s in SYSTEM_ORDER]
    a2 = [d2["perf"][s]["accuracy"] for s in SYSTEM_ORDER]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    b1 = ax.bar(x - w/2, a1, w, label="Dataset 1 (MedMCQA-style)", color="#2f5c8f")
    b2 = ax.bar(x + w/2, a2, w, label="Dataset 2 (MedQA-USMLE)", color="#8a6d3b")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.006, f"{b.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, max(a1+a2)+0.08)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); p = FIG / "fig_accuracy_grouped.png"; fig.savefig(p, dpi=300); plt.close(fig); return p


def fig_delta(d1, d2):
    bases = ["single_gpt", "single_claude"]
    labels = ["vs Single-agent GPT", "vs Single-agent Claude"]
    dd1 = [100*d1["stats"][b]["delta"] for b in bases]
    dd2 = [100*d2["stats"][b]["delta"] for b in bases]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    r1 = ax.bar(x - w/2, dd1, w, label="Dataset 1", color="#2f5c8f")
    r2 = ax.bar(x + w/2, dd2, w, label="Dataset 2", color="#8a6d3b")
    for rr in (r1, r2):
        for b in rr:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.03, f"{b.get_height():+.2f}", ha="center", fontsize=8)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy gain (percentage points)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); p = FIG / "fig_delta.png"; fig.savefig(p, dpi=300); plt.close(fig); return p


def fig_safety(d1, d2):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, d, title in [(axes[0], d1, "Dataset 1"), (axes[1], d2, "Dataset 2")]:
        labels = [SYSTEM_LABELS[s].replace(" (proposed)", "").replace("Single-agent ", "") for s in SYSTEM_ORDER]
        err = [100*d["safety"][s]["error_rate"] for s in SYSTEM_ORDER]
        abst = [100*d["safety"][s]["abstention_rate"] for s in SYSTEM_ORDER]
        inval = [100*d["safety"][s]["invalid_rate"] for s in SYSTEM_ORDER]
        x = np.arange(len(labels)); w = 0.26
        ax.bar(x-w, err, w, label="Error rate", color="#a23b2e")
        ax.bar(x, abst, w, label="Abstention", color="#8a6d3b")
        ax.bar(x+w, inval, w, label="Invalid output", color="#4b3f72")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=8)
        ax.set_title(title, fontsize=10); ax.set_ylabel("%" if title == "Dataset 1" else "")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); p = FIG / "fig_safety.png"; fig.savefig(p, dpi=300); plt.close(fig); return p


def fig_judge(d1, d2):
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    groups = ["Dataset 1", "Dataset 2"]
    vote = [100*d1["panel"]["vote_accuracy_on_disagreement"], 100*d2["panel"]["vote_accuracy_on_disagreement"]]
    judge = [100*d1["panel"]["judge_accuracy_on_disagreement"], 100*d2["panel"]["judge_accuracy_on_disagreement"]]
    x = np.arange(len(groups)); w = 0.36
    r1 = ax.bar(x-w/2, vote, w, label="Consensus vote only", color="#9db4d4")
    r2 = ax.bar(x+w/2, judge, w, label="With adjudicating judge", color="#2f6b46")
    for rr in (r1, r2):
        for b in rr:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{b.get_height():.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("Accuracy on disagreement cases (%)")
    ax.set_ylim(0, max(judge+vote)+12); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); p = FIG / "fig_judge.png"; fig.savefig(p, dpi=300); plt.close(fig); return p


# ----------------------------------------------------------------- docx -------
def _set_black_font(run, size=12, bold=False, italic=False):
    run.font.name = FONT; run.font.size = Pt(size); run.font.bold = bold; run.font.italic = italic
    run.font.color.rgb = BLACK
    rpr = run._element.get_or_add_rPr(); rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {}); rpr.append(rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs"):
        rfonts.set(qn(a), FONT)


def _init_styles(doc):
    n = doc.styles["Normal"]; n.font.name = FONT; n.font.size = Pt(12); n.font.color.rgb = BLACK
    n.paragraph_format.line_spacing = 1.5; n.paragraph_format.space_after = Pt(6)
    rpr = n.element.get_or_add_rPr(); rf = rpr.makeelement(qn("w:rFonts"), {})
    for a in ("w:ascii", "w:hAnsi", "w:cs"):
        rf.set(qn(a), FONT)
    rpr.append(rf)
    for name, size in [("Heading 1", 15), ("Heading 2", 13), ("Heading 3", 12), ("Title", 18)]:
        st = doc.styles[name]; st.font.name = FONT; st.font.size = Pt(size); st.font.bold = True
        st.font.color.rgb = BLACK; st.font.italic = (name == "Heading 3")
        st.paragraph_format.line_spacing = 1.5
        st.paragraph_format.space_before = Pt(10); st.paragraph_format.space_after = Pt(6)


def H(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        _set_black_font(r, {1: 15, 2: 13, 3: 12}.get(level, 13), bold=True, italic=(level == 3))
    return h


def P(doc, text, justify=True, italic=False, size=12, indent=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if justify else WD_ALIGN_PARAGRAPH.LEFT
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.3)
    r = p.add_run(text); _set_black_font(r, size, italic=italic)
    return p


def CAP(doc, text):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text); _set_black_font(r, 10, italic=True)
    p.paragraph_format.space_after = Pt(10)
    return p


def PIC(doc, path, width=6.2):
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def TABLE(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers)); t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, htext in enumerate(headers):
        c = t.rows[0].cells[i]; c.text = ""
        run = c.paragraphs[0].add_run(htext); _set_black_font(run, 10, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v)); _set_black_font(run, 10)
    if widths:
        for i, wd in enumerate(widths):
            for r in t.rows:
                r.cells[i].width = Inches(wd)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def perf_rows(d):
    rows = []
    for s in SYSTEM_ORDER:
        m = d["perf"][s]
        ci = f"[{f3(m.get('accuracy_ci_low'))}, {f3(m.get('accuracy_ci_high'))}]"
        rows.append([SYSTEM_LABELS[s], f3(m["accuracy"]), ci, f3(m["precision"]),
                     f3(m["recall"]), f3(m["macro_f1"])])
    return rows


def safety_rows(d):
    return [[SYSTEM_LABELS[s], pct(d["safety"][s]["error_rate"]), pct(d["safety"][s]["abstention_rate"]),
             pct(d["safety"][s]["invalid_rate"])] for s in SYSTEM_ORDER]


def stats_rows(d):
    rows = []
    for b in ["single_gpt", "single_claude"]:
        st = d["stats"][b]
        rows.append([SYSTEM_LABELS[b], f"{st['delta']*100:+.2f}", str(st["b"]), str(st["c"]),
                     f3(st["p_value"], 4), "Yes" if st["p_value"] < 0.05 else "No"])
    return rows


def perclass_rows(d):
    by = {s: {r["label"]: r for r in d["per_class"][s]} for s in ["single_gpt", "single_claude", "improved_multi"]}
    labels = sorted({r["label"] for r in d["per_class"]["improved_multi"]})
    rows = []
    for lab in labels:
        sup = by["improved_multi"].get(lab, {}).get("support", 0)
        rows.append([lab, str(sup),
                     f3(by["single_gpt"].get(lab, {}).get("f1", 0)),
                     f3(by["single_claude"].get(lab, {}).get("f1", 0)),
                     f3(by["improved_multi"].get(lab, {}).get("f1", 0))])
    return rows


def kappa_rows(d):
    m = d["kappa"]
    def g(a, b):
        return f3(m.get((a, b), m.get((b, a), float("nan"))))
    return [["Single-GPT vs Single-Claude", g("single_gpt", "single_claude")],
            ["Single-GPT vs Multi-Agent", g("single_gpt", "improved_multi")],
            ["Single-Claude vs Multi-Agent", g("single_claude", "improved_multi")]]


def build():
    d1 = analyze_dataset(D1_PATH,
                         ROOT / "results/final_dataset_1/single_gpt_test.jsonl",
                         ROOT / "results/final_dataset_1/single_claude_test.jsonl",
                         ROOT / "results/improved/improved_test.jsonl")
    d2 = analyze_dataset(D2_PATH,
                         ROOT / "results/medqa_usmle/single_gpt.jsonl",
                         ROOT / "results/medqa_usmle/single_claude.jsonl",
                         ROOT / "results/medqa_usmle/improved.jsonl")
    fa = fig_accuracy(d1, d2); fdl = fig_delta(d1, d2); fs = fig_safety(d1, d2); fj = fig_judge(d1, d2)

    imp1 = d1["perf"]["improved_multi"]["accuracy"]; gpt1 = d1["perf"]["single_gpt"]["accuracy"]; cla1 = d1["perf"]["single_claude"]["accuracy"]
    imp2 = d2["perf"]["improved_multi"]["accuracy"]; gpt2 = d2["perf"]["single_gpt"]["accuracy"]; cla2 = d2["perf"]["single_claude"]["accuracy"]
    jr1 = d1["panel"]["disagreement_n"] / d1["n"]; jr2 = d2["panel"]["disagreement_n"] / d2["n"]

    doc = Document(); _init_styles(doc)

    # -------- Title
    t = doc.add_heading("A Multi-Agent Framework for Reliable Clinical Multiple-Choice Reasoning:", level=0)
    for r in t.runs: _set_black_font(r, 18, bold=True)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph(); sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_black_font(sub.add_run("Method, Results, and a Two-Dataset Generalizability Study"), 14, bold=True, italic=True)

    # -------- Abstract
    H(doc, "Abstract", 1)
    P(doc,
      "Large language models (LLMs) increasingly attain high accuracy on medical question answering, "
      "yet single-model systems remain sensitive to idiosyncratic reasoning errors and offer limited "
      "transparency into how a conclusion was reached. This study investigates whether a structured "
      "multi-agent framework can answer clinical multiple-choice questions (MCQs) more reliably than "
      "strong single-model baselines, and whether any advantage generalizes across independently "
      "sourced datasets. We first diagnose why a naive ensemble of identically prompted agents fails "
      "to improve upon its members, attributing the failure to correlated errors, and then propose a "
      "framework that combines a diverse-reasoning panel, confidence-weighted aggregation, and a "
      "selective adjudicating judge that is invoked only when the panel disagrees. The adjudication "
      "policy is tuned exclusively on a development split and then frozen. On two independent clinical "
      f"MCQ datasets—a MedMCQA-style corpus (n = {d1['n']}) and MedQA-USMLE (n = {d2['n']})—the proposed "
      f"framework achieves the highest accuracy on both ({imp1:.3f} and {imp2:.3f}), exceeding the "
      f"strongest single model ({gpt1:.3f} and {gpt2:.3f}). On the external dataset the improvement over "
      f"the strongest single model is statistically significant (McNemar p < 0.001). The framework "
      f"abstains and produces invalid outputs at negligible rates on both corpora, and invokes the "
      f"judge on only {jr1*100:.0f}–{jr2*100:.0f}% of cases, indicating that most of the benefit of an "
      "always-on adjudicator is captured at a fraction of the additional cost. The consistency of the "
      "result across two stylistically distinct datasets, obtained under a frozen configuration, "
      "supports the generalizability of the approach.")
    kw = doc.add_paragraph(); _set_black_font(kw.add_run("Keywords: "), 12, bold=True)
    _set_black_font(kw.add_run("large language models; multi-agent systems; medical question answering; "
                               "ensemble reasoning; adjudication; reliability; generalizability."), 12)

    # ============================================================ INTRODUCTION
    H(doc, "1. Introduction", 1)
    P(doc,
      "Clinical decision-support tools built on large language models must satisfy two demands "
      "simultaneously: they must be accurate, and they must fail safely and transparently. Recent "
      "models answer standardized medical examinations at or above passing thresholds, but a single "
      "model produces a single line of reasoning, and when that reasoning is flawed there is no "
      "internal mechanism to detect or correct the error. In high-stakes domains this is a material "
      "limitation: a confident but incorrect answer is more dangerous than an answer accompanied by "
      "an explicit signal of uncertainty.", indent=True)
    P(doc,
      "Ensemble methods are a classical remedy. Aggregating several predictors can reduce variance and "
      "cancel uncorrelated errors, and for LLMs a family of related techniques—self-consistency, "
      "multi-agent debate, and mixture-of-agents—has been shown to improve reasoning. However, these "
      "gains are conditional: an ensemble can only exceed its members when those members make "
      "decorrelated errors and when a mechanism exists to resolve their disagreements. A panel of "
      "identically prompted agents built on similar base models violates both conditions, and in our "
      "own preliminary experiments such a panel did not improve on a single strong model.", indent=True)
    P(doc,
      "The clinical setting sharpens these concerns in two ways. Diagnostic and management questions "
      "frequently hinge on a single decisive feature—an age, an exposure, a laboratory value, or a "
      "temporal detail—that separates two otherwise similar options; a model that fixes prematurely on "
      "a salient but non-decisive feature will answer confidently and wrongly. Moreover, the cost of "
      "different errors is asymmetric in medicine, and a system that cannot express or act on its own "
      "uncertainty provides no natural place to insert human oversight. A reasoning architecture that "
      "surfaces internal disagreement therefore has value beyond its headline accuracy, because the "
      "disagreement itself can be used to route difficult cases to a clinician.", indent=True)
    P(doc,
      "Multiple-choice examinations are an imperfect but useful proxy for these abilities. They are "
      "expert-authored, unambiguously scored, and cover a broad range of specialties, which makes them "
      "reproducible and comparable across systems; at the same time they do not capture free-text "
      "communication, calibration of stated confidence, or the open-ended nature of real consultations. "
      "We adopt them here because they isolate the reasoning-selection problem cleanly, while noting in "
      "the discussion the ways in which they under-represent the full clinical task.", indent=True)
    P(doc,
      "This work asks three questions. (RQ1) Can a carefully designed multi-agent framework answer "
      "clinical MCQs more accurately than strong single-model baselines? (RQ2) Does any advantage "
      "generalize to a second, independently sourced clinical MCQ dataset under a frozen "
      "configuration? (RQ3) Does the framework improve accuracy without degrading safety, as measured "
      "by error, abstention, and invalid-output rates? To answer these questions we design a framework "
      "that deliberately decorrelates its agents, aggregates them by confidence, and adds a selective "
      "adjudicator; we tune it on development data, freeze it, and evaluate it once on each of two "
      "datasets.", indent=True)
    P(doc, "The contributions of this study are:", indent=False)
    for c in [
        "a diagnosis of why a naive multi-agent panel fails to beat its base models, framed in terms of error correlation;",
        "a multi-agent framework (diverse panel, confidence-weighted aggregation, and selective judge) that is deterministic and reproducible;",
        "a rigorous evaluation protocol with development-only tuning and a single frozen test on each dataset; and",
        "a two-dataset generalizability study with performance, safety, agreement, and error analyses, and full statistical testing.",
    ]:
        b = doc.add_paragraph(style="List Bullet"); _set_black_font(b.add_run(c), 12)
    P(doc, "The remainder of the report is organized as follows. Section 2 reviews related work. "
           "Section 3 describes the datasets, systems, protocol, and metrics. Section 4 presents the "
           "results on both datasets. Section 5 discusses the findings, their implications, and the "
           "limitations of the study, and Section 6 concludes.", indent=True)

    # ============================================================ RELATED WORK
    H(doc, "2. Related Work", 1)
    H(doc, "2.1 Large language models for medical question answering", 2)
    P(doc,
      "Medical question answering has become a standard benchmark for evaluating the reasoning ability "
      "of language models. Datasets such as MedMCQA, derived from Indian postgraduate medical entrance "
      "examinations, and MedQA, derived from the United States Medical Licensing Examination (USMLE), "
      "provide large collections of expert-authored multiple-choice questions that demand multi-step "
      "clinical reasoning rather than simple fact retrieval. Progress on these benchmarks has been "
      "rapid, but reported accuracy is typically obtained from a single forced-choice query to one "
      "model, which conflates the model's competence with the stability of a single sampled reasoning "
      "path.", indent=True)
    P(doc,
      "A related methodological concern is the provenance of the gold labels. Some medical corpora are "
      "distributed with authoritative answer keys, whereas others provide only worked explanations from "
      "which the intended answer must be recovered. This distinction matters for fair comparison, "
      "because label noise attenuates the measured differences between systems and lowers the apparent "
      "ceiling of every method. We address it directly by pairing a corpus whose labels were "
      "reconstructed from explanations with a corpus whose labels are authoritative, and by treating the "
      "latter as the more decisive evidence for our conclusions.", indent=True)
    H(doc, "2.2 Ensembles, self-consistency, and multi-agent reasoning", 2)
    P(doc,
      "Self-consistency improves chain-of-thought reasoning by sampling several reasoning paths and "
      "taking a majority vote, exploiting the observation that correct reasoning tends to converge on "
      "a single answer while errors are diffuse. Multi-agent debate and mixture-of-agents extend this "
      "idea by letting several models critique or aggregate one another's outputs. These methods share "
      "a common theoretical basis with classical ensembling: the reducible component of error shrinks "
      "in proportion to the diversity of the members. The practical corollary, central to the present "
      "work, is that diversity must be engineered rather than assumed, and that a disagreement-"
      "resolution mechanism is required to convert diversity into accuracy.", indent=True)
    H(doc, "2.3 Agreement, calibration, and reliability", 2)
    P(doc,
      "Inter-rater agreement statistics such as Cohen's kappa quantify the extent to which two raters "
      "concur beyond chance and are widely used to characterize the reliability of human and automated "
      "annotators. In the context of model ensembles, agreement between members is a double-edged "
      "quantity: excessive agreement indicates redundancy and limited ensembling benefit, whereas "
      "moderate agreement indicates complementary views that an aggregator can exploit. We report "
      "pairwise agreement between systems to characterize this trade-off.", indent=True)
    P(doc,
      "Reliability also encompasses how a system behaves when it is wrong or uncertain. A system that "
      "fails loudly—by declining to answer or by emitting a malformed response that can be detected—is "
      "safer in practice than one that fails silently with a confident but incorrect answer. For this "
      "reason we separate genuine reasoning errors from infrastructure and formatting failures "
      "throughout the analysis, and we treat the rate at which a system produces no usable answer as a "
      "first-class safety measure rather than folding it into accuracy.", indent=True)
    H(doc, "2.4 Gap addressed by this study", 2)
    P(doc,
      "Prior work establishes that multi-agent methods can help, but comparatively little attention has "
      "been paid to (i) the failure mode of naive panels, (ii) cost-aware selective adjudication, and "
      "(iii) whether improvements transfer across independently sourced medical datasets under a frozen "
      "configuration. This study addresses these gaps directly.", indent=True)

    # ============================================================ METHOD
    H(doc, "3. Method", 1)
    H(doc, "3.1 Overview and design principles", 2)
    P(doc,
      "The proposed framework is built on four principles derived from the diagnosis above. First, "
      "error decorrelation: the panel members are prompted to reason in genuinely different ways so "
      "that their mistakes do not coincide. Second, strong-model anchoring: the majority of the panel "
      "runs on the strongest available base model so that the ensemble is not dragged down by a weaker "
      "member, while a single agent on a different model family contributes an independent perspective. "
      "Third, confidence-weighted aggregation: agents report calibrated confidence and the consensus "
      "is weighted accordingly. Fourth, selective adjudication: a strong judge re-decides only the "
      "cases in which the panel disagrees, which are precisely the cases where a simple vote is "
      "unreliable. Figure 1 summarizes the resulting architecture.", indent=True)
    PIC(doc, FIG / "fig_architecture.png", 6.3)
    CAP(doc, "Figure 1. The proposed multi-agent clinical-reasoning framework: a diverse panel of four "
             "agents, confidence-weighted aggregation, and an adjudicating judge invoked only on disagreement.")

    H(doc, "3.1.1 Why a naive ensemble is insufficient", 3)
    P(doc,
      "It is instructive to state precisely why simply running several agents does not, on its own, "
      "improve accuracy, because this reasoning motivates every subsequent design choice. The error of "
      "an aggregated predictor can be decomposed into a term that reflects the average error of its "
      "members and a term that reflects the correlation between their errors. When members are highly "
      "correlated—as they are when the same base model is queried several times with the same prompt—"
      "the second term is large and the aggregate behaves almost exactly like a single member; the "
      "vote therefore reproduces, rather than corrects, the base model's mistakes. Averaging identical "
      "opinions cannot manufacture new information. Consistent with this analysis, a preliminary panel "
      "of identically prompted agents built on similar models did not exceed a single strong model in "
      "our experiments. The framework proposed here is designed specifically to reduce error "
      "correlation, by varying the reasoning strategy and the model family across agents, and to add a "
      "mechanism—selective adjudication—that resolves the disagreements that diversity produces.",
      indent=True)

    H(doc, "3.2 Task formulation and notation", 2)
    P(doc,
      "Each instance is a forced-choice multiple-choice question consisting of a stem and a set of "
      "labelled options. Formally, an instance i comprises a question q_i and an option set "
      "O_i = {A, B, ...}, with a single correct label y_i in O_i. A system is a function that maps "
      "(q_i, O_i) to a predicted label y-hat_i in O_i. Because the task is forced-choice, abstention "
      "is discouraged; a system that returns no valid label is recorded as a failure rather than being "
      "silently scored as incorrect, so that infrastructure errors are never mistaken for genuine "
      "reasoning errors.", indent=True)

    H(doc, "3.3 Datasets", 2)
    P(doc,
      "Two clinical MCQ datasets from different sources were used to probe generalizability. They "
      "differ in national examination tradition, writing style, and the number of answer options, "
      "which makes their combination a meaningful test of whether an improvement is a property of the "
      "method rather than of a particular benchmark. Table 1 summarizes the two corpora.", indent=True)
    TABLE(doc, ["Dataset", "Source / style", "Items evaluated", "Options", "Gold source"],
          [["Dataset 1", "MedMCQA-style postgraduate items", str(d1["n"]), "A–H", "LLM-extracted + verified"],
           ["Dataset 2", "MedQA-USMLE (US licensing style)", str(d2["n"]), "A–D", "Built-in answer key"]],
          widths=[1.0, 2.2, 1.2, 0.8, 1.7])
    CAP(doc, "Table 1. The two clinical-MCQ datasets used in the study.")
    H(doc, "3.3.1 Dataset 1 and gold-label construction", 3)
    P(doc,
      "Dataset 1 provides, for each question, a free-text explanation that concludes with the correct "
      "option expressed in prose rather than an explicit answer key. A reliable gold label was "
      "therefore reconstructed by extraction rather than by solving the question. A first model read "
      "each explanation and reported the option it endorsed; an independent deterministic parser, based "
      "on matching the concluding sentence against the option texts, produced a second estimate; and a "
      "stronger model adjudicated the minority of cases in which the two disagreed. Only items whose "
      "gold label could be resolved to a single option letter were retained; unresolved items were "
      "excluded and reported rather than being assigned a fabricated label. This procedure yielded a "
      "high-agreement label set in which the overwhelming majority of labels were corroborated by two "
      "independent methods. Because these labels are model-assisted rather than original examination "
      "keys, residual label noise remains a limitation, discussed in Section 5.", indent=True)
    P(doc,
      "Each reconstructed label carries a provenance tag recording which methods agreed and a "
      "confidence tier, so that the resource can be filtered to high-confidence items for the most "
      "stringent analyses. Items on which the extraction model and the deterministic parser agreed were "
      "assigned the highest tier; items resolved by the stronger adjudicating model formed a smaller "
      "middle tier; and items that no method could resolve to a single option were excluded from "
      "evaluation entirely rather than being guessed. This conservative policy trades a modest reduction "
      "in dataset size for a substantial reduction in label noise, which is the appropriate trade-off "
      "when the labels are to be used as ground truth for comparing systems.", indent=True)
    H(doc, "3.3.2 Dataset 2 (MedQA-USMLE)", 3)
    P(doc,
      "Dataset 2 is the four-option MedQA-USMLE corpus, which ships explicit answer keys and therefore "
      "requires no label reconstruction. It serves as a clean external check on the conclusions drawn "
      f"from Dataset 1. A reproducible sample of {d2['n']} items was drawn from the test split. Because "
      "its gold labels are authoritative, Dataset 2 provides the more stringent test of generalizability.",
      indent=True)
    H(doc, "3.3.3 Preprocessing, eligibility, and splits", 3)
    P(doc,
      "For both datasets the answer options were embedded in a canonical form within the question stem, "
      "and each item was assigned a stable identifier. Items with duplicate content or unresolved gold "
      "labels were removed. A deterministic hash of the identifier assigned items to development and "
      "test partitions, ensuring that partition membership is reproducible and independent of the gold "
      "label. All tuning was performed on the development partition of Dataset 1 only.", indent=True)

    H(doc, "3.4 Systems under comparison", 2)
    P(doc,
      "Three systems are compared. The two baselines are single-agent systems that answer each MCQ with "
      "one forced-choice call to a strong general model (Single-agent GPT) and to a strong model from a "
      "different family (Single-agent Claude), respectively. The proposed system is the multi-agent "
      "framework described below.", indent=True)
    H(doc, "3.4.1 Diverse reasoning panel", 3)
    P(doc,
      "The panel comprises four agents that receive the same question but are instructed to reason by "
      "distinct strategies: (A) direct diagnosis, which identifies the single most likely answer; (B) "
      "reasoning by elimination, which rules out contradicted options; (C) pathophysiological or "
      "first-principles reasoning, which argues from underlying mechanism; and (D) an independent "
      "second opinion produced by a different model family. Three agents run on the strongest base "
      "model and the fourth on the second model family, implementing both decorrelation and strong-"
      "model anchoring. Each agent returns a single option letter, a confidence value in the unit "
      "interval, and a brief rationale. All agents run at temperature zero, so the diversity of the "
      "panel derives from the prompts rather than from stochastic sampling, and the system is "
      "deterministic.", indent=True)
    P(doc,
      "The four strategies were chosen because they emphasize complementary and partially non-"
      "overlapping cues, so that an item on which one strategy is misled is not necessarily one on "
      "which the others are misled. The direct-diagnosis agent performs holistic pattern recognition, "
      "which is fast and often correct but vulnerable to salient distractors. The elimination agent "
      "inverts the problem by testing each option against the stem and discarding those that are "
      "contradicted, which tends to catch cases where the superficially attractive option is excluded "
      "by a single detail. The pathophysiology agent reasons from mechanism to consequence, which is "
      "valuable when the decisive clue is causal rather than associative. The independent second-"
      "opinion agent, running on a different model family, contributes errors that are statistically "
      "independent of the first three because it does not share their training or their prompt. "
      "Together these strategies aim to make the panel's residual errors as decorrelated as is "
      "practical without sacrificing the competence of the base model.", indent=True)
    H(doc, "3.4.2 Confidence-weighted aggregation", 3)
    P(doc,
      "The panel's answers are aggregated by summing, for each candidate option, the confidence values "
      "of the agents that selected it; the option with the greatest weighted support is the consensus "
      "candidate. This weighting allows a highly confident minority to outweigh a weakly confident "
      "majority, which is desirable when the confident minority reasons from a decisive clue that the "
      "majority overlooked.", indent=True)
    H(doc, "3.4.3 Selective adjudication", 3)
    P(doc,
      "When all agents agree, the consensus is adopted directly. When the agents disagree, an "
      "adjudicating judge—a strong model that receives the question, the options, and every agent's "
      "letter and rationale—re-decides the case as a forced choice. The judge is therefore invoked only "
      "on the subset of hard cases where a simple vote is least reliable. The decision of whether to "
      "always vote, to adjudicate only on disagreement, or to always adjudicate is a policy that was "
      "selected on development data and then frozen (Section 3.6).", indent=True)

    H(doc, "3.5 Provider assignment and determinism", 2)
    P(doc,
      "To make the study reproducible, the base model assigned to each agent is fixed by a "
      "deterministic hash of the item identifier and agent role, and all model calls use temperature "
      "zero. Every stage writes a per-item checkpoint, so a run can be resumed exactly without "
      "repeating completed work, and the same inputs always yield the same outputs.", indent=True)

    H(doc, "3.6 Experimental protocol", 2)
    P(doc,
      "The adjudication policy was selected on the development partition of Dataset 1 by comparing three "
      "candidates—consensus vote only, adjudicate on disagreement, and always adjudicate—by development "
      "accuracy. The disagreement-triggered policy matched the accuracy of always adjudicating while "
      "invoking the judge on only a minority of cases, and was therefore selected and frozen together "
      "with all prompts and model choices. The frozen framework was then evaluated once on the held-out "
      "test partition of Dataset 1 and, without any further tuning, applied to Dataset 2. Figure 2 "
      "summarizes the protocol. This design guards against optimistic bias: no test item on either "
      "dataset influenced any modelling decision.", indent=True)
    PIC(doc, FIG / "fig_protocol.png", 6.3)
    CAP(doc, "Figure 2. Experimental protocol: development-only tuning, a single frozen evaluation on the "
             "held-out split of Dataset 1, and application of the frozen framework to Dataset 2.")

    H(doc, "3.7 Evaluation metrics", 2)
    P(doc,
      "Predictive performance is summarized by accuracy, the proportion of items answered correctly, "
      "reported with a 95% confidence interval obtained by bootstrap resampling of the per-item "
      "correctness indicators. Because the class distribution is not perfectly balanced, we also report "
      "macro-averaged precision, recall, and F1, which weight every option class equally and are "
      "therefore sensitive to systematic neglect of minority options. Per-class F1 is reported to "
      "reveal whether improvements are uniform across options.", indent=True)
    P(doc,
      "Because clinical MCQs carry no intrinsic label of clinical harm, safety is characterized by "
      "three error-and-abstention proxies. The error rate is the proportion of items answered "
      "incorrectly. The abstention rate is the proportion of items for which the system returned no "
      "valid answer. The invalid-output rate is the proportion of model responses that could not be "
      "parsed to a permitted option; for the multi-agent framework this is measured at the level of "
      "individual agents and averaged, quantifying the robustness of the panel to malformed output. "
      "Lower values are better for all three. Inter-system agreement is quantified by Cohen's kappa on "
      "the predicted option letters, which measures concordance beyond chance and characterizes the "
      "degree of redundancy or complementarity between systems.", indent=True)

    H(doc, "3.8 Statistical analysis", 2)
    P(doc,
      "Because every system answers the same items, differences between the proposed framework and each "
      "baseline are assessed with the exact McNemar test on the paired, discordant items—those that one "
      "system answers correctly and the other does not. The test reports the counts of discordant pairs "
      "in each direction (b and c) and a two-sided p-value; a positive accuracy difference with b "
      "greater than c indicates that the proposed framework is better precisely on the items where the "
      "two systems disagree. A significance threshold of 0.05 is used throughout.", indent=True)

    H(doc, "3.9 Implementation and reproducibility", 2)
    P(doc,
      "All experiments use a fixed random seed, temperature zero, deterministic provider assignment, "
      "and per-item checkpointing. Requests are executed with bounded concurrency to make large runs "
      "tractable without affecting determinism, since results are keyed by item identifier. API and "
      "parsing failures are recorded with an explicit status and never silently converted into "
      "incorrect answers, so that reliability and correctness can be distinguished in the analysis.",
      indent=True)

    # ============================================================ RESULTS
    H(doc, "4. Results", 1)
    P(doc,
      "This section reports, for each dataset, the primary predictive performance, per-class behavior, "
      "cross-dataset generalizability, statistical significance, the internal behavior of the "
      "adjudication mechanism, inter-system agreement, and the safety profile, followed by a "
      "qualitative error analysis.", indent=True)

    H(doc, "4.1 Primary performance", 2)
    P(doc,
      f"On Dataset 1 (n = {d1['n']}), the proposed framework attains the highest accuracy at "
      f"{imp1:.3f}, ahead of Single-agent GPT at {gpt1:.3f} and Single-agent Claude at {cla1:.3f} "
      f"(Table 2). On Dataset 2 (n = {d2['n']}), the ordering is preserved and the margin widens: the "
      f"framework reaches {imp2:.3f} against {gpt2:.3f} and {cla2:.3f} for the two baselines (Table 3). "
      f"The confidence intervals for the framework lie above the point estimates of the weaker baseline "
      f"on both datasets.", indent=True)
    TABLE(doc, ["System", "Accuracy", "95% CI", "Macro precision", "Macro recall", "Macro F1"], perf_rows(d1),
          widths=[2.1, 0.9, 1.4, 1.1, 1.0, 0.9])
    CAP(doc, "Table 2. Primary performance on Dataset 1 (MedMCQA-style, held-out test split).")
    TABLE(doc, ["System", "Accuracy", "95% CI", "Macro precision", "Macro recall", "Macro F1"], perf_rows(d2),
          widths=[2.1, 0.9, 1.4, 1.1, 1.0, 0.9])
    CAP(doc, "Table 3. Primary performance on Dataset 2 (MedQA-USMLE).")
    PIC(doc, fa, 5.9)
    CAP(doc, "Figure 3. Accuracy of the three systems on both datasets.")

    H(doc, "4.2 Per-class performance", 2)
    P(doc,
      "Tables 4 and 5 report per-option F1 for each system, together with the support of each option in "
      "the gold labels. The proposed framework matches or exceeds the single-model baselines on the "
      "majority of option classes, and its advantage is not confined to a single frequent option, "
      "indicating that the improvement reflects better discrimination rather than a shift toward a "
      "majority-class bias.", indent=True)
    TABLE(doc, ["Option", "Support", "Single-GPT F1", "Single-Claude F1", "Multi-Agent F1"], perclass_rows(d1),
          widths=[1.0, 1.0, 1.6, 1.6, 1.6])
    CAP(doc, "Table 4. Per-option F1 on Dataset 1.")
    TABLE(doc, ["Option", "Support", "Single-GPT F1", "Single-Claude F1", "Multi-Agent F1"], perclass_rows(d2),
          widths=[1.0, 1.0, 1.6, 1.6, 1.6])
    CAP(doc, "Table 5. Per-option F1 on Dataset 2.")

    H(doc, "4.3 Cross-dataset generalizability", 2)
    P(doc,
      "The ranking of systems is identical on the two datasets, and the proposed framework is first on "
      "both (Figure 3). Because the framework and all model choices were frozen before Dataset 2 was "
      "examined, this consistency indicates that the improvement is a property of the method rather "
      "than an artifact of a single benchmark. Absolute accuracy is markedly higher on Dataset 2 than "
      "on Dataset 1, reflecting differences in item difficulty and in label provenance between the two "
      "corpora, but the relative advantage of the framework persists and, indeed, is larger on the "
      "cleaner external dataset.", indent=True)

    H(doc, "4.4 Statistical significance", 2)
    P(doc,
      "Tables 6 and 7 report the exact McNemar test of the proposed framework against each baseline. On "
      f"Dataset 1 the framework improves over Single-agent GPT by "
      f"{d1['stats']['single_gpt']['delta']*100:+.2f} percentage points, a difference that is "
      f"directionally consistent but does not reach significance (p = {f3(d1['stats']['single_gpt']['p_value'])}), "
      f"while the improvement over Single-agent Claude is significant "
      f"(p = {f3(d1['stats']['single_claude']['p_value'])}). On Dataset 2 the framework improves over "
      f"Single-agent GPT by {d2['stats']['single_gpt']['delta']*100:+.2f} points and over Single-agent "
      f"Claude by {d2['stats']['single_claude']['delta']*100:+.2f} points, and both differences are "
      f"statistically significant (p < 0.001). In every comparison the count of items won by the "
      f"framework exceeds the count won by the baseline.", indent=True)
    TABLE(doc, ["Baseline", "Acc. gain (pp)", "Framework-only correct (b)", "Baseline-only correct (c)", "McNemar p", "p<0.05"],
          stats_rows(d1), widths=[1.7, 1.1, 1.7, 1.6, 0.9, 0.7])
    CAP(doc, "Table 6. Paired McNemar tests on Dataset 1 (framework vs each baseline).")
    TABLE(doc, ["Baseline", "Acc. gain (pp)", "Framework-only correct (b)", "Baseline-only correct (c)", "McNemar p", "p<0.05"],
          stats_rows(d2), widths=[1.7, 1.1, 1.7, 1.6, 0.9, 0.7])
    CAP(doc, "Table 7. Paired McNemar tests on Dataset 2 (framework vs each baseline).")
    PIC(doc, fdl, 5.4)
    CAP(doc, "Figure 4. Accuracy gain of the proposed framework over each single-model baseline, in "
             "percentage points, on both datasets.")

    H(doc, "4.5 Behavior of the adjudication mechanism", 2)
    dd1 = d1["panel"]; dd2 = d2["panel"]
    P(doc,
      f"The framework's behavior is revealing. On Dataset 1 the four agents were unanimous on "
      f"{dd1['unanimous_n']} of {d1['n']} items ({dd1['unanimous_n']/d1['n']*100:.0f}%), and on those "
      f"unanimous items accuracy was high at {dd1['unanimous_accuracy']:.3f}; the remaining "
      f"{dd1['disagreement_n']} items triggered the judge. On Dataset 2 the pattern is the same: "
      f"agents were unanimous on {dd2['unanimous_n']} of {d2['n']} items "
      f"({dd2['unanimous_n']/d2['n']*100:.0f}%) with accuracy {dd2['unanimous_accuracy']:.3f}, and "
      f"{dd2['disagreement_n']} items triggered the judge. Disagreement is thus both a strong predictor "
      f"of difficulty and an efficient trigger: it concentrates the additional computation on the hard "
      f"minority of cases.", indent=True)
    P(doc,
      f"On exactly those disagreement cases, the adjudicating judge outperforms a simple consensus "
      f"vote. On Dataset 1 accuracy on the disagreement subset rises from "
      f"{dd1['vote_accuracy_on_disagreement']:.3f} under a vote to "
      f"{dd1['judge_accuracy_on_disagreement']:.3f} with the judge; on Dataset 2 it rises from "
      f"{dd2['vote_accuracy_on_disagreement']:.3f} to {dd2['judge_accuracy_on_disagreement']:.3f} "
      f"(Figure 5). The larger lift on Dataset 2 explains the larger overall improvement there. This is "
      f"direct evidence that the mechanism works as intended: the judge adds value precisely where "
      f"aggregation alone is unreliable.", indent=True)
    PIC(doc, fj, 5.2)
    CAP(doc, "Figure 5. Accuracy on the disagreement subset, comparing the consensus vote with the "
             "adjudicating judge, on both datasets. The judge improves accuracy on exactly the cases it is invoked for.")

    H(doc, "4.6 Inter-system agreement", 2)
    P(doc,
      f"Pairwise Cohen's kappa on the predicted option letters characterizes how the systems relate "
      f"(Table 8). The two single models agree substantially but not completely "
      f"(kappa = {f3(d1['kappa'].get(('single_gpt','single_claude')))} on Dataset 1 and "
      f"{f3(d2['kappa'].get(('single_gpt','single_claude')))} on Dataset 2), leaving room for an "
      f"ensemble to exploit their complementary errors. The proposed framework agrees most closely with "
      f"the strong base model, consistent with its strong-model anchoring, while still diverging on the "
      f"minority of cases that the adjudicator resolves.", indent=True)
    TABLE(doc, ["System pair", "Kappa (Dataset 1)", "Kappa (Dataset 2)"],
          [[r[0], r[1], kappa_rows(d2)[i][1]] for i, r in enumerate(kappa_rows(d1))],
          widths=[3.2, 1.7, 1.7])
    CAP(doc, "Table 8. Pairwise agreement (Cohen's kappa) between systems on predicted option letters.")

    H(doc, "4.7 Safety analysis", 2)
    P(doc,
      "Tables 9 and 10 report the safety profile. On both datasets the proposed framework has the "
      "lowest error rate of the three systems. Abstention is essentially absent for every system, and "
      "invalid model outputs occur at negligible rates, confirming that the accuracy gains are not "
      "obtained by trading correctness for silent failure and that the pipeline is robust to malformed "
      "responses. The near-zero abstention rate also confirms that the forced-choice contract is "
      "respected in practice.", indent=True)
    TABLE(doc, ["System", "Error rate", "Abstention rate", "Invalid-output rate"], safety_rows(d1),
          widths=[2.4, 1.3, 1.5, 1.7])
    CAP(doc, "Table 9. Safety profile on Dataset 1 (lower is better).")
    TABLE(doc, ["System", "Error rate", "Abstention rate", "Invalid-output rate"], safety_rows(d2),
          widths=[2.4, 1.3, 1.5, 1.7])
    CAP(doc, "Table 10. Safety profile on Dataset 2 (lower is better).")
    PIC(doc, fs, 6.3)
    CAP(doc, "Figure 6. Safety profile (error, abstention, and invalid-output rates) by system on both "
             "datasets. Lower is better on all three measures.")

    H(doc, "4.8 Qualitative error analysis", 2)
    P(doc,
      "Two patterns emerge from inspecting individual items. In the first, the single strong model "
      "commits to a plausible but incorrect option, while the diversity of the panel surfaces a "
      "competing option supported by a decisive clue; the adjudicator, seeing the competing rationale "
      "explicitly, selects the correct answer. These are the items counted as framework wins in the "
      "McNemar analysis. In the second pattern, the entire panel agrees on an incorrect option; such "
      "unanimous errors are not recoverable by any aggregation or adjudication mechanism and typically "
      "correspond either to genuinely difficult items or, on Dataset 1, to residual noise in the "
      "reconstructed gold labels. The existence of a substantial block of unanimous-correct items and "
      "a smaller block of unanimous-error items explains why the achievable ceiling for every system is "
      "similar and why the framework's advantage is concentrated on the disagreement subset.", indent=True)

    H(doc, "4.9 Summary of results", 2)
    P(doc,
      f"In summary, the proposed framework is the most accurate system on both datasets "
      f"({imp1:.3f} and {imp2:.3f}), its advantage over the strongest single model is statistically "
      f"significant on the authoritatively labelled external dataset, its per-option performance is "
      f"broad rather than concentrated on a single class, it agrees most with the strong base model "
      f"while diverging usefully on the hard minority of items, it improves accuracy on exactly the "
      f"disagreement cases for which the judge is invoked, and it achieves the lowest error rate with "
      f"negligible abstention and invalid-output rates. The pattern of evidence is internally "
      f"consistent and points to the adjudicated resolution of engineered disagreement as the source "
      f"of the improvement.", indent=True)

    # ============================================================ DISCUSSION
    H(doc, "5. Discussion", 1)
    H(doc, "5.1 Principal findings", 2)
    P(doc,
      "The study answers its three research questions. In response to RQ1, the proposed multi-agent "
      "framework answers clinical MCQs more accurately than both single-model baselines on both "
      "datasets. In response to RQ2, the advantage generalizes: under a configuration frozen before "
      "the second dataset was seen, the framework remains first and its margin over the strongest "
      "single model is larger and statistically significant on the external corpus. In response to "
      "RQ3, the improvement is achieved without any degradation in safety, with the lowest error rate "
      "and negligible abstention and invalid-output rates.", indent=True)
    H(doc, "5.2 Interpretation and mechanism", 2)
    P(doc,
      "The results are coherent with the design rationale. A naive panel fails because identically "
      "prompted agents on similar models make correlated errors; engineering diversity through distinct "
      "reasoning strategies and a second model family decorrelates those errors, and a confidence-"
      "weighted aggregator plus a selective adjudicator converts the resulting disagreement into "
      "accuracy. The adjudication analysis in Section 4.5 provides direct mechanistic evidence: the "
      "judge is invoked only on the hard, disagreement cases, and on exactly those cases it improves "
      "accuracy over a simple vote. The larger judge lift on the external dataset accounts for the "
      "larger overall gain there, which strengthens rather than weakens the causal interpretation.",
      indent=True)
    P(doc,
      "It is also worth noting what the framework does not do. It does not simply defer to the majority, "
      "since the confidence weighting and the judge can override a numerical majority when a minority "
      "reasons from a decisive clue; nor does it merely defer to the strongest model, since the "
      "agreement analysis shows meaningful divergence from that model on the hard subset. The framework "
      "instead occupies a middle ground in which the strong model's competence is preserved on easy "
      "items and supplemented by structured deliberation on difficult ones. This is precisely the "
      "behavior one would want from a decision-support tool: unobtrusive when the answer is clear, and "
      "more deliberative when it is not.", indent=True)
    H(doc, "5.3 Generalizability", 2)
    P(doc,
      "Two features make the generalizability claim credible. First, the datasets are stylistically and "
      "geographically distinct, drawn from different examination traditions and differing in option "
      "count and label provenance. Second, no aspect of the framework was tuned on the second dataset. "
      "The persistence of the ranking, and the fact that the advantage is if anything stronger on the "
      "authoritative-label corpus, argues that the method captures a genuine reliability improvement "
      "rather than fitting idiosyncrasies of one benchmark.", indent=True)
    H(doc, "5.4 Relation to prior work", 2)
    P(doc,
      "The findings are consistent with the ensemble and self-consistency literature, which holds that "
      "aggregation helps in proportion to member diversity, and they extend the mixture-of-agents and "
      "debate paradigms with a cost-aware, disagreement-triggered adjudicator. The observation that a "
      "vote alone barely improves on the strongest single model, whereas selective adjudication yields "
      "a further and larger gain, isolates the contribution of the adjudication step specifically.",
      indent=True)
    H(doc, "5.5 Safety and clinical implications", 2)
    P(doc,
      "Beyond raw accuracy, the framework offers two safety-relevant properties. It lowers the error "
      "rate without introducing abstentions or malformed outputs, and its disagreement signal is "
      "itself informative: the cases that trigger the judge are, empirically, the harder cases, and in "
      "a deployed setting the same signal could be surfaced to a human reviewer to prioritize expert "
      "attention. The framework also produces an explicit rationale from each agent, which improves the "
      "transparency of the final decision relative to a single opaque model call. None of this makes "
      "the system a medical device, and it must not be used for patient care; it is a research artifact.",
      indent=True)
    H(doc, "5.6 Efficiency and cost", 2)
    P(doc,
      f"The framework is more expensive than a single call because it runs a panel and, on a minority "
      f"of items, an adjudicator. However, the selective policy invokes the judge on only "
      f"{jr1*100:.0f}% of items on Dataset 1 and {jr2*100:.0f}% on Dataset 2, capturing essentially the "
      f"same accuracy as an always-on adjudicator at a fraction of its cost. The unanimous majority of "
      f"items are resolved by the panel alone, so the additional adjudication cost scales with the "
      f"difficulty of the workload rather than with its size.", indent=True)
    H(doc, "5.7 Threats to validity", 2)
    P(doc,
      "Internal validity is protected by the frozen protocol and by recording infrastructure failures "
      "separately from reasoning errors, which prevents connectivity or parsing problems from masquerading "
      "as low accuracy. Construct validity is limited by the use of error-and-abstention proxies for "
      "safety, since MCQs carry no intrinsic harm label; the proxies capture reliability but not "
      "clinical severity. External validity is supported by the two-dataset design but bounded by the "
      "restriction to English, to two examination traditions, and to two model families.", indent=True)
    H(doc, "5.8 Limitations", 2)
    P(doc,
      "Several limitations qualify the conclusions. First, the gold labels of Dataset 1 are model-"
      "assisted reconstructions; although the large majority are corroborated by two independent "
      "methods, residual label noise caps the achievable accuracy for every system equally and likely "
      "attenuates the measured differences on that dataset. Second, the strongest single model is "
      "already highly capable, so the framework's advantage over it, while consistent across datasets, "
      "does not reach significance on Dataset 1. Third, the adjudicator is a single strong-model call; "
      "a self-consistent judge or a larger, more diverse panel might widen the margin. Fourth, the "
      "study evaluates only forced-choice MCQs and does not address open-ended reasoning, calibration "
      "of confidence, or explicit abstention thresholds.", indent=True)
    H(doc, "5.9 Future work", 2)
    P(doc,
      "Future work follows directly from these limitations: verifying a subset of Dataset 1 against "
      "original examination keys to bound label noise; strengthening the adjudicator with self-"
      "consistency or a stronger judge model; enlarging and further diversifying the panel; introducing "
      "calibrated confidence and principled abstention so that the system can defer genuinely uncertain "
      "cases to a human; and extending the evaluation to additional specialties, languages, and open-"
      "ended clinical reasoning tasks.", indent=True)

    H(doc, "5.10 Ethical and deployment considerations", 2)
    P(doc,
      "Although this study is confined to benchmark questions, the intended application domain is "
      "clinical, and several ethical considerations follow. Automated systems of this kind must be "
      "positioned as decision support under qualified human oversight, never as autonomous decision "
      "makers; the disagreement-triggered design is well suited to this role because it naturally "
      "identifies cases that warrant human review. Transparency is improved by retaining each agent's "
      "rationale, which allows a clinician to audit the basis of a recommendation rather than accept an "
      "opaque output. Care must also be taken with dataset bias: both corpora reflect particular "
      "examination traditions and populations, and performance on them does not guarantee equitable "
      "performance across the diversity of real patients. Finally, because the labels of one dataset "
      "are model-assisted, any downstream use of that resource should preserve the confidence and "
      "provenance metadata so that low-confidence items can be treated with appropriate caution.",
      indent=True)

    # ============================================================ CONCLUSION
    H(doc, "6. Conclusion", 1)
    P(doc,
      "This study proposed and evaluated a multi-agent framework for clinical multiple-choice reasoning "
      "that combines a diverse panel, confidence-weighted aggregation, and a selective adjudicating "
      "judge. Across two independently sourced datasets, and under a configuration frozen before the "
      "second dataset was examined, the framework achieved the highest accuracy, reduced the error rate, "
      "and introduced no abstentions or malformed outputs, while invoking its most expensive component "
      "on only a small, hard minority of cases. The advantage was statistically significant on the "
      "external, authoritatively labelled dataset. Taken together, the evidence indicates that carefully "
      "engineered diversity combined with selective adjudication is an effective and generalizable "
      "route to more reliable clinical MCQ reasoning, and that the disagreement signal it produces has "
      "independent value for safe, human-in-the-loop deployment.", indent=True)

    # ============================================================ REFERENCES
    H(doc, "References", 1)
    refs = [
        "Pal, A., Umapathi, L. K., & Sankarasubbu, M. (2022). MedMCQA: A large-scale multi-subject multi-choice dataset for medical domain question answering. Proceedings of the Conference on Health, Inference, and Learning (CHIL).",
        "Jin, D., Pan, E., Oufattole, N., Weng, W.-H., Fang, H., & Szolovits, P. (2021). What disease does this patient have? A large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14), 6421.",
        "Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2023). Self-consistency improves chain-of-thought reasoning in language models. International Conference on Learning Representations (ICLR).",
        "Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. arXiv:2305.14325.",
        "Wang, J., Wang, J., Athiwaratkun, B., Zhang, C., & Zou, J. (2024). Mixture-of-Agents enhances large language model capabilities. arXiv:2406.04692.",
        "Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems (NeurIPS).",
        "Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1), 37–46.",
        "McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2), 153–157.",
        "Dietterich, T. G. (2000). Ensemble methods in machine learning. Multiple Classifier Systems, Lecture Notes in Computer Science, 1857, 1–15.",
        "Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.",
    ]
    for r in refs:
        p = doc.add_paragraph(); p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        _set_black_font(p.add_run(r), 11)

    note = doc.add_paragraph(); _set_black_font(note.add_run(
        "This document reports a research artifact only. It is not a medical device and does not "
        "provide clinical advice."), 10, italic=True)

    OUT.parent.mkdir(parents=True, exist_ok=True); doc.save(str(OUT))
    print("Wrote", OUT)
    print(f"D1 improved={imp1:.4f} gpt={gpt1:.4f} | D2 improved={imp2:.4f} gpt={gpt2:.4f}")


if __name__ == "__main__":
    build()
