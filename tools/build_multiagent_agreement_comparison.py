"""Generate the five-method agreement comparison notebook."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "MultiAgent_Agreement_Comparison.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells = [
    md("""
    # Multi-Agent Agreement Comparison

    **Objective:** compare Legacy Confidence Agreement, nominal Krippendorff's
    alpha, Kendall's W, Jensen-Shannon divergence (JSD), and the experimental
    **Multi-Agent Clinical Agreement (MACA) Score** using saved medical MCQ
    outputs. This notebook is offline, makes no API calls, and does not modify
    production consensus, routing, Critic, or Judge behavior.

    Agreement is not correctness. Gold answers are used only to evaluate whether
    a score predicts final correctness; no agreement algorithm receives gold.
    """),
    md("""
    ## Methodological scope

    - **Legacy:** exact current confidence consistency algorithm.
    - **Krippendorff alpha:** dataset-level nominal reliability over specialist
      letters; missing/abstentions are omitted. Per-case alpha is intentionally
      `NaN` because it is not a valid case-level statistic.
    - **Kendall W:** requires complete option rankings/scores. Historical outputs
      do not contain them, so W is unavailable rather than invented.
    - **JSD:** historical outputs lack true option distributions. The exploratory
      analysis assigns selected-answer confidence to that option and spreads the
      residual uniformly. Every row records this source.
    - **MACA:** geometric agreement across modal vote, probability similarity,
      and confidence consistency, multiplied by participation and a documented
      disagreement-severity penalty. MACA never uses gold.
    """),
    code("""
    from pathlib import Path
    import json, sys
    import numpy as np
    import pandas as pd
    from scipy import stats
    from sklearn.metrics import average_precision_score, roc_auc_score
    from IPython.display import display

    def find_root(start=Path.cwd()):
        for candidate in [start.resolve(), *start.resolve().parents]:
            if (candidate / "evaluation" / "agreement_analysis.py").exists():
                return candidate
        raise FileNotFoundError("Repository root not found")

    ROOT = find_root()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from evaluation.agreement_analysis import (
        approximate_distribution, bootstrap_alpha, bootstrap_statistic,
        build_case_analysis, compute_legacy_agreement, compute_maca_score,
        evaluate_agreement_predictor, krippendorff_alpha_nominal,
        normalize_specialist_opinions, pairwise_jsd, robustness_to_missing_agents,
    )

    RUN_SPLIT = "development"
    RANDOM_SEED = 2026
    N_BOOTSTRAP = 2000
    canonical_name = "Multi-Agent-AI-for-Reliable-Clinical-Reasoning"
    RESULT_CANDIDATES = [
        (ROOT / "results" / "mcq_v2").resolve(),
        (ROOT.parent / "results" / "mcq_v2").resolve(),
        (ROOT.parent / "Multi-Agent-AI-for-Reliable-Clin" / canonical_name / "results" / "mcq_v2").resolve(),
        (ROOT.parent.parent / canonical_name / "results" / "mcq_v2").resolve(),
    ]

    def find_result(filename):
        matches = [directory / filename for directory in RESULT_CANDIDATES if (directory / filename).exists()]
        if not matches:
            raise FileNotFoundError(f"Missing {filename}; run V2 Notebook 03 first")
        return matches[0]

    MULTI_PATH = find_result(f"multi_dynamic_one_gpt5_v2_{RUN_SPLIT}_predictions.csv")
    OUTPUT_ROOT = (ROOT.parent.parent / canonical_name / "results" / "mcq_v2").resolve()
    if not (OUTPUT_ROOT.parent.parent / "evaluation" / "agreement_analysis.py").exists():
        OUTPUT_ROOT = (ROOT / "results" / "mcq_v2").resolve()
    OUTPUT_DIR = OUTPUT_ROOT / "agreement_comparison" / RUN_SPLIT
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Input:", MULTI_PATH)
    print("Output:", OUTPUT_DIR)
    """),
    md("""
    ## Load and validate specialist outputs

    Technical failures are excluded, duplicate case IDs are rejected, answer
    letters and confidence ranges are validated, and all exclusions are saved.
    """),
    code("""
    multi = pd.read_csv(MULTI_PATH, keep_default_na=False)
    required = {"fixed_case_id", "reference_letter", "predicted_letter", "correct", "specialist_opinions", "difficulty", "critic_used"}
    missing = required - set(multi)
    assert not missing, f"Missing columns: {sorted(missing)}"
    assert not multi.fixed_case_id.duplicated().any(), "Duplicate case IDs"
    assert set(multi.split) == {RUN_SPLIT}, "Unexpected split"
    api_error = multi.api_error.astype(str).str.lower().isin(["true", "1"])
    exclusions = multi.loc[api_error, ["fixed_case_id"]].assign(reason="api_error")
    multi = multi.loc[~api_error].copy()
    assert multi.reference_letter.str.fullmatch(r"[A-H]").all()
    exclusions.to_csv(OUTPUT_DIR / "excluded_cases.csv", index=False)
    print("Cases analyzed:", len(multi), "excluded:", len(exclusions))
    """),
    md("""
    ## Compute five agreement methods

    MACA formulation:

    `geometric_mean(answer agreement, probability agreement when available,
    confidence agreement) * participation * (1 - 0.25 * severity)`

    where confidence agreement is `1 - population_SD/0.5`, JSD uses base 2,
    and severity rises with the number of distinct answers. Components are
    clipped to `[0,1]`. Gold is not passed to this computation.
    """),
    code("""
    base = build_case_analysis(multi, allow_approximated_jsd=True)
    maca_rows = []
    ratings = []
    validation_issues = []
    for record in multi.to_dict("records"):
        opinions, issues = normalize_specialist_opinions(record["specialist_opinions"])
        validation_issues.extend({"case_id": record["fixed_case_id"], "issue": issue} for issue in issues)
        ratings.append([None if item["abstain"] else item["answer_letter"] for item in opinions])
        features = base.loc[base.fixed_case_id == record["fixed_case_id"]].iloc[0]
        maca = compute_maca_score(
            opinions,
            mean_pairwise_jsd=features.mean_pairwise_jsd,
            probability_source=features.jsd_probability_source,
        )
        maca_rows.append({"fixed_case_id": record["fixed_case_id"], **maca})

    cases = base.merge(pd.DataFrame(maca_rows), on="fixed_case_id", validate="one_to_one")
    cases = cases.rename(columns={
        "fixed_case_id": "case_id", "reference_letter": "gold_answer",
        "correct": "final_correct", "legacy_agreement_score": "legacy_score",
        "mean_pairwise_jsd": "JSD", "max_pairwise_jsd": "max_JSD",
        "maca_score": "MACA", "critic_used": "critic_triggered",
    })
    cases["Krippendorff_alpha"] = np.nan
    cases["krippendorff_scope"] = "dataset_level_only"
    cases["Kendall_W"] = np.nan
    cases["kendall_status"] = "unavailable_no_full_rankings_or_option_scores"
    cases["final_correct"] = cases.final_correct.astype(bool)
    cases["correct"] = cases["final_correct"]  # shared predictor API alias
    cases["JSD_agreement"] = 1 - cases.JSD
    cases["jsd_direction_for_correctness"] = "reversed_as_1_minus_JSD"

    overall_alpha = krippendorff_alpha_nominal(ratings)
    alpha_ci = bootstrap_alpha(ratings, N_BOOTSTRAP, RANDOM_SEED)
    overall_alpha.update(ci_low=alpha_ci[0], ci_high=alpha_ci[1])
    pd.DataFrame(validation_issues, columns=["case_id", "issue"]).to_csv(OUTPUT_DIR / "validation_issues.csv", index=False)
    requested_columns = [
        "case_id", "gold_answer", "predicted_letter", "final_correct", "legacy_score",
        "Krippendorff_alpha", "Kendall_W", "JSD", "max_JSD", "MACA", "difficulty",
        "critic_triggered", "jsd_probability_source", "maca_probability_source",
        "modal_vote_fraction", "vote_entropy", "number_unique_answers", "unanimous",
        "number_participating", "number_abstained", "maca_answer_agreement",
        "maca_probability_agreement", "maca_confidence_agreement", "maca_participation",
        "maca_disagreement_severity", "krippendorff_scope", "kendall_status",
    ]
    cases[requested_columns].to_csv(OUTPUT_DIR / "case_level_five_algorithm_scores.csv", index=False)
    print("Dataset-level alpha:", overall_alpha)
    print("Kendall W: unavailable (rankings absent)")
    display(cases[requested_columns].head())
    """),
    md("""
    ## Predictive evaluation and bootstrap uncertainty

    AUROC/AUPRC/correlation evaluate final correctness. Error detection uses the
    inverse agreement direction (or raw JSD). Dataset-level alpha and unavailable
    W cannot produce per-case predictive statistics and remain `NaN`.
    """),
    code("""
    algorithm_specs = {
        "Legacy Confidence Agreement": ("legacy_score", "higher_is_better"),
        "Krippendorff's Alpha": ("Krippendorff_alpha", "higher_is_better"),
        "Kendall's W": ("Kendall_W", "higher_is_better"),
        "Jensen-Shannon Divergence": ("JSD", "lower_is_better"),
        "MACA Score": ("MACA", "higher_is_better"),
    }

    def predictive_row(name, column, direction):
        result = evaluate_agreement_predictor(cases, column, direction, N_BOOTSTRAP, RANDOM_SEED)
        subset = cases[[column, "final_correct"]].copy()
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
        subset = subset.dropna()
        if len(subset) and subset.final_correct.nunique() == 2:
            correctness_score = -subset[column] if direction == "lower_is_better" else subset[column]
            error_score = -correctness_score
            error_y = (~subset.final_correct).astype(int)
            error_auroc = roc_auc_score(error_y, error_score)
            error_auprc = average_precision_score(error_y, error_score)
            boot = pd.DataFrame({"y": subset.final_correct.astype(int), "score": correctness_score})
            auprc_ci = bootstrap_statistic(
                boot, lambda x: average_precision_score(x.y, x.score)
                if x.y.nunique() == 2 else np.nan, N_BOOTSTRAP, RANDOM_SEED
            )
            corr_ci = bootstrap_statistic(
                boot, lambda x: stats.pointbiserialr(x.y, x.score).statistic
                if x.y.nunique() == 2 and x.score.nunique() > 1 else np.nan,
                N_BOOTSTRAP, RANDOM_SEED,
            )
            error_boot = boot.assign(error_y=1 - boot.y, error_score=-boot.score)
            error_ci = bootstrap_statistic(
                error_boot, lambda x: roc_auc_score(x.error_y, x.error_score)
                if x.error_y.nunique() == 2 else np.nan, N_BOOTSTRAP, RANDOM_SEED
            )
        else:
            error_auroc = error_auprc = np.nan
            auprc_ci = corr_ci = error_ci = (np.nan, np.nan)
        return {
            "Algorithm": name, "n": result["n"], "AUROC": result["auroc"],
            "AUPRC": result["auprc"], "Correlation": result["point_biserial_r"],
            "AUROC_CI_low": result["ci_low"], "AUROC_CI_high": result["ci_high"],
            "AUPRC_CI_low": auprc_ci[0], "AUPRC_CI_high": auprc_ci[1],
            "Correlation_CI_low": corr_ci[0], "Correlation_CI_high": corr_ci[1],
            "Error_Detection_AUROC": error_auroc, "Error_Detection_AUPRC": error_auprc,
            "Error_AUROC_CI_low": error_ci[0], "Error_AUROC_CI_high": error_ci[1],
            "Direction": direction,
        }

    predictive = pd.DataFrame([predictive_row(name, *spec) for name, spec in algorithm_specs.items()])
    predictive.to_csv(OUTPUT_DIR / "predictive_performance.csv", index=False)
    display(predictive)
    """),
    md("""
    ## Robustness, disagreement, and difficulty

    Stability is `1 - mean absolute score change` after randomly removing one
    participating specialist. Alpha stability is based on normalized bootstrap
    CI width. Kendall W remains unavailable.
    """),
    code("""
    def legacy_scorer(items, _):
        return compute_legacy_agreement(items)["legacy_agreement_score"]

    def jsd_scorer(items, record):
        distributions = []
        for item in items:
            if item.get("abstain") or not np.isfinite(item.get("confidence", np.nan)):
                continue
            distributions.append(approximate_distribution(item["answer_letter"], item["confidence"], list("ABCD")))
        return pairwise_jsd(distributions)["mean_pairwise_jsd"]

    def maca_scorer(items, record):
        jsd = jsd_scorer(items, record)
        return compute_maca_score(items, mean_pairwise_jsd=jsd, probability_source="approximated_from_selected_confidence")["maca_score"]

    robustness = {
        "Legacy Confidence Agreement": robustness_to_missing_agents(multi, legacy_scorer, n_repeats=1000, seed=RANDOM_SEED),
        "Jensen-Shannon Divergence": robustness_to_missing_agents(multi, jsd_scorer, n_repeats=1000, seed=RANDOM_SEED),
        "MACA Score": robustness_to_missing_agents(multi, maca_scorer, n_repeats=1000, seed=RANDOM_SEED),
    }
    alpha_width = overall_alpha["ci_high"] - overall_alpha["ci_low"]
    robustness["Krippendorff's Alpha"] = {"robustness_mae": np.nan, "stability": float(np.clip(1 - alpha_width / 2, 0, 1)), "robustness_trials": N_BOOTSTRAP}
    robustness["Kendall's W"] = {"robustness_mae": np.nan, "stability": np.nan, "robustness_trials": 0}
    robustness_table = pd.DataFrame([{"Algorithm": name, **value} for name, value in robustness.items()])

    strata_rows = []
    for name, (column, direction) in algorithm_specs.items():
        for stratum, mask in {
            "all": pd.Series(True, index=cases.index),
            "disagreement": cases.number_unique_answers.gt(1),
            "unanimous": cases.unanimous.astype(bool),
        }.items():
            subset = cases.loc[mask]
            values = pd.to_numeric(subset[column], errors="coerce")
            valid = pd.DataFrame({"score": values, "correct": subset.final_correct}).dropna()
            oriented = -valid.score if direction == "lower_is_better" else valid.score
            stratum_auroc = roc_auc_score(valid.correct.astype(int), oriented) if len(valid) and valid.correct.nunique() == 2 else np.nan
            strata_rows.append({"Algorithm": name, "stratum": stratum, "n": len(subset), "mean_score": values.mean(), "accuracy": subset.final_correct.mean(), "predictive_auroc": stratum_auroc})
        for difficulty, subset in cases.groupby("difficulty"):
            values = pd.to_numeric(subset[column], errors="coerce")
            valid = pd.DataFrame({"score": values, "correct": subset.final_correct}).dropna()
            oriented = -valid.score if direction == "lower_is_better" else valid.score
            stratum_auroc = roc_auc_score(valid.correct.astype(int), oriented) if len(valid) and valid.correct.nunique() == 2 else np.nan
            strata_rows.append({"Algorithm": name, "stratum": f"difficulty={difficulty}", "n": len(subset), "mean_score": values.mean(), "accuracy": subset.final_correct.mean(), "predictive_auroc": stratum_auroc})
    strata = pd.DataFrame(strata_rows)
    robustness_table.to_csv(OUTPUT_DIR / "robustness_missing_agents.csv", index=False)
    strata.to_csv(OUTPUT_DIR / "performance_by_disagreement_and_difficulty.csv", index=False)
    display(robustness_table)
    display(strata)
    """),
    md("""
    ## Multi-metric final ranking

    Overall score weights: AUROC 25%, AUPRC 20%, correlation 15%, error AUROC
    20%, stability 15%, case-level coverage 5%. Correlation is mapped from
    `[-1,1]` to `[0,1]`. Missing required predictive metrics score zero; this
    penalizes methods that cannot serve as per-case detectors. This ranking is
    specific to the requested operational use, not a claim that a dataset-level
    reliability statistic is intrinsically inferior.
    """),
    code("""
    ranking = predictive.merge(robustness_table[["Algorithm", "stability"]], on="Algorithm", how="left")
    ranking["Coverage"] = ranking.n / len(cases)
    ranking["Correlation_norm"] = (ranking.Correlation + 1) / 2
    for column in ["AUROC", "AUPRC", "Correlation_norm", "Error_Detection_AUROC", "stability", "Coverage"]:
        ranking[column] = ranking[column].fillna(0).clip(0, 1)
    ranking["Overall Score"] = (
        0.25 * ranking.AUROC + 0.20 * ranking.AUPRC +
        0.15 * ranking.Correlation_norm + 0.20 * ranking.Error_Detection_AUROC +
        0.15 * ranking.stability + 0.05 * ranking.Coverage
    )
    ranking = ranking.sort_values(["Overall Score", "AUROC"], ascending=False).reset_index(drop=True)
    ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))
    ranking = ranking.rename(columns={"Error_Detection_AUROC": "Error Detection", "stability": "Stability"})
    final_ranking = ranking[["Rank", "Algorithm", "AUROC", "AUPRC", "Correlation", "Error Detection", "Stability", "Overall Score"]]
    final_ranking.to_csv(OUTPUT_DIR / "final_algorithm_ranking.csv", index=False)
    display(final_ranking)
    """),
    md("""
    ## Targeted clinical comparisons
    """),
    code("""
    legacy_high = cases.legacy_score >= cases.legacy_score.quantile(0.75)
    categorical_disagreement = cases.number_unique_answers.gt(1)
    targeted = {
        "legacy_high_but_agents_disagree": cases.loc[legacy_high & categorical_disagreement],
        "unanimous_but_incorrect": cases.loc[cases.unanimous.astype(bool) & ~cases.final_correct],
        "disagreement_corrected_by_critic_judge": cases.loc[categorical_disagreement & cases.critic_triggered.astype(bool) & (cases.modal_answer != cases.gold_answer) & cases.final_correct],
    }
    for name, table in targeted.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        print(name, "n=", len(table)); display(table[["case_id", "gold_answer", "predicted_letter", "final_correct", "legacy_score", "JSD", "MACA", "difficulty", "critic_triggered"]])

    legacy_row = final_ranking.set_index("Algorithm").loc["Legacy Confidence Agreement"]
    maca_row = final_ranking.set_index("Algorithm").loc["MACA Score"]
    jsd_row = final_ranking.set_index("Algorithm").loc["Jensen-Shannon Divergence"]
    direct = pd.DataFrame([
        {"comparison": "Legacy vs MACA", "first_overall": legacy_row["Overall Score"], "second_overall": maca_row["Overall Score"], "difference_second_minus_first": maca_row["Overall Score"] - legacy_row["Overall Score"]},
        {"comparison": "JSD vs MACA", "first_overall": jsd_row["Overall Score"], "second_overall": maca_row["Overall Score"], "difference_second_minus_first": maca_row["Overall Score"] - jsd_row["Overall Score"]},
    ])
    direct.to_csv(OUTPUT_DIR / "direct_algorithm_comparisons.csv", index=False)
    display(direct)
    """),
    md("""
    ## Empirical conclusion

    The recommendation is generated from observed results. MACA is not assumed
    to win. Kendall W remains unavailable until future specialists save complete
    rankings or option scores; true-distribution JSD/MACA should be rerun once
    genuine option probabilities are collected.
    """),
    code("""
    best_overall = final_ranking.iloc[0].Algorithm
    available_error = final_ranking[final_ranking["Error Detection"] > 0]
    best_error = available_error.sort_values("Error Detection", ascending=False).iloc[0].Algorithm if len(available_error) else "unavailable"
    best_traditional = max(
        ["Krippendorff's Alpha", "Kendall's W", "Jensen-Shannon Divergence"],
        key=lambda name: float(final_ranking.set_index("Algorithm").loc[name, "Overall Score"]),
    )
    maca_rank = int(final_ranking.set_index("Algorithm").loc["MACA Score", "Rank"])
    maca_beats_legacy = maca_row["Overall Score"] > legacy_row["Overall Score"]
    recommendation = best_overall
    print("Best overall algorithm:", best_overall)
    print("Best disagreement detector:", best_error)
    print("Best traditional agreement statistic:", best_traditional)
    print("MACA rank:", maca_rank)
    print("Does MACA outperform the existing agreement algorithm?:", "Yes" if maca_beats_legacy else "No")
    print("Recommended algorithm for the future multi-agent system:", recommendation)
    conclusion = {
        "best_overall_algorithm": best_overall,
        "best_disagreement_detector": best_error,
        "best_traditional_agreement_statistic": best_traditional,
        "maca_rank": maca_rank,
        "maca_outperforms_legacy": bool(maca_beats_legacy),
        "recommended_future_algorithm": recommendation,
        "important_limitation": "JSD and MACA probability components are approximated from selected-answer confidence; Kendall W unavailable.",
    }
    (OUTPUT_DIR / "conclusion.json").write_text(json.dumps(conclusion, indent=2), encoding="utf-8")
    """),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
nbf.write(notebook, TARGET)
print(f"Created: {TARGET}")
