"""Generate the offline agreement and multi-agent evaluation notebook."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "Agreement_and_MultiAgent_Evaluation.ipynb"


def md(value: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(value).strip())


def code(value: str):
    return nbf.v4.new_code_cell(textwrap.dedent(value).strip())


cells = [
    md("""
    # Agreement and Multi-Agent Evaluation

    **Purpose.** Offline, paired evaluation of specialist agreement, calibration,
    disagreement resolution, and single-agent versus multi-agent MCQ performance.
    This notebook makes **no API calls** and does not modify the production
    consensus or Critic logic.

    **Conceptual safeguards:** Agreement is not correctness; confidence is not
    agreement or calibration; consensus is not clinical validity. Development is
    for exploratory threshold selection. Test is for one locked evaluation only.
    """),
    md("""
    ## 1-3. Configuration, imports, and reproducibility

    Historical V2 specialist opinions contain selected answer letters and
    confidence, but normally do not contain complete option probabilities or
    rankings. Therefore nominal alpha and vote measures are primary. Kendall W
    and true JSD are reported only when their required data exist. The optional
    selected-confidence JSD is explicitly labeled exploratory and is never mixed
    with reported-distribution JSD.
    """),
    code("""
    from pathlib import Path
    import json, sys, warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def find_root(start=Path.cwd()):
        for candidate in [start, *start.parents]:
            if (candidate / "evaluation" / "agreement_analysis.py").exists():
                return candidate
        raise FileNotFoundError("Run this notebook from inside the repository.")

    ROOT = find_root()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from evaluation.agreement_analysis import (
        VALID_LETTERS, bootstrap_alpha, bootstrap_statistic, build_case_analysis,
        calibration_summary, calibration_table, create_disagreement_trigger,
        evaluate_agreement_predictor, kendalls_w, krippendorff_alpha_nominal,
        normalize_specialist_opinions, paired_single_multi,
    )

    RANDOM_SEED = 2026
    N_BOOTSTRAP = 2000
    RUN_SPLIT = "development"  # change to test only after all choices are frozen
    ALLOW_EXPLORATORY_APPROXIMATED_JSD = True
    RESULTS_DIR = ROOT / "results" / "mcq_v2"
    OUTPUT_DIR = RESULTS_DIR / "agreement_analysis" / RUN_SPLIT
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    MULTI_PATH = RESULTS_DIR / f"multi_dynamic_one_gpt5_v2_{RUN_SPLIT}_predictions.csv"
    SINGLE_PATH = RESULTS_DIR / f"single_gpt5_{RUN_SPLIT}_predictions.csv"
    DISAGREEMENT_CONFIG = {
        "min_modal_vote_fraction": 0.75,
        "high_jsd_threshold": 0.30,
        "low_confidence_threshold": 0.60,
        "router_confidence_threshold": 0.50,
    }
    if RUN_SPLIT == "test":
        warnings.warn("LOCKED TEST: do not tune DISAGREEMENT_CONFIG from these outcomes.")
    print("Analysis split:", RUN_SPLIT)
    print("Multi:", MULTI_PATH)
    print("Single:", SINGLE_PATH)
    """),
    md("""
    ## 4-5. Load, detect, validate, and normalize saved results

    Detected V2 fields include `fixed_case_id`, `reference_letter`, specialist
    `answer_letter`, `confidence`, `specialty`/role, final `predicted_letter`,
    Judge confidence, Critic activation, difficulty, and correctness. V2 removed
    numerical Router confidence intentionally, so analyses requiring it remain
    unavailable rather than substituting another score.
    """),
    code("""
    def load_active_result(path, kind):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing active {kind} result: {path}\\n"
                "Run Notebook 02 (single) and Notebook 03 (multi) successfully first. "
                "Archived failed/mock runs are intentionally not accepted."
            )
        frame = pd.read_csv(path, keep_default_na=False)
        required = {"fixed_case_id", "reference_letter", "predicted_letter", "correct", "confidence", "split"}
        if kind == "multi": required.add("specialist_opinions")
        missing = required - set(frame)
        assert not missing, f"{kind} missing columns: {sorted(missing)}"
        assert not frame.fixed_case_id.duplicated().any(), f"Duplicate {kind} case IDs"
        assert set(frame.split) == {RUN_SPLIT}, f"Unexpected {kind} split"
        assert frame.reference_letter.astype(str).str.fullmatch(r"[A-H]").all(), f"Invalid {kind} gold letter"
        api_error = frame.get("api_error", pd.Series(False, index=frame.index)).astype(str).str.lower().isin(["true", "1"])
        if api_error.mean() == 1:
            raise RuntimeError(f"{kind} result is a technical-failure run (100% API errors), not an outcome dataset")
        return frame.loc[~api_error].copy()

    multi = load_active_result(MULTI_PATH, "multi")
    single = load_active_result(SINGLE_PATH, "single")
    case_analysis = build_case_analysis(multi, ALLOW_EXPLORATORY_APPROXIMATED_JSD)
    case_analysis["correct"] = case_analysis.correct.astype(bool)
    case_analysis.to_csv(OUTPUT_DIR / "case_level_agreement_analysis.csv", index=False)

    opinion_rows = []
    excluded = []
    for row in multi.to_dict("records"):
        opinions, issues = normalize_specialist_opinions(row["specialist_opinions"])
        excluded.extend({"fixed_case_id": row["fixed_case_id"], "reason": issue} for issue in issues)
        for index, opinion in enumerate(opinions):
            opinion_rows.append({
                "fixed_case_id": row["fixed_case_id"], "specialist_index": index,
                "specialty": opinion["specialty"], "answer_letter": opinion["answer_letter"],
                "confidence": opinion["confidence"], "abstain": opinion["abstain"],
                "reference_letter": row["reference_letter"],
                "correct": opinion["answer_letter"] == row["reference_letter"] and not opinion["abstain"],
                "difficulty": row.get("difficulty", "unknown"), "critic_used": row.get("critic_used", False),
            })
    specialist_long = pd.DataFrame(opinion_rows)
    exclusions = pd.DataFrame(excluded, columns=["fixed_case_id", "reason"])
    exclusions.to_csv(OUTPUT_DIR / "excluded_agreement_records.csv", index=False)
    print("Valid multi cases:", len(multi), "valid single cases:", len(single))
    print("Specialist records:", len(specialist_long), "normalization issues:", len(exclusions))
    print("JSD sources:", case_analysis.jsd_probability_source.value_counts(dropna=False).to_dict())
    """),
    md("""
    ## 6-10. Legacy confidence consistency, categorical agreement, alpha, Kendall W, and JSD

    `legacy_agreement_score` exactly reproduces the production confidence-based
    calculation: `mean_confidence * (1 - population_stdev / 0.5)`. It primarily
    measures confidence consistency, not true answer agreement.

    Krippendorff alpha is dataset-level nominal reliability. Per-case analysis
    uses modal vote fraction, unique answers, normalized vote entropy,
    unanimity, and majority agreement. JSD uses base-2 logarithms and lies in
    `[0,1]`. Approximation assigns selected-answer confidence to the selected
    option and spreads residual mass uniformly; it is exploratory only.
    """),
    code("""
    ratings_by_case = [
        group.sort_values("specialist_index").apply(
            lambda r: None if r.abstain else r.answer_letter, axis=1
        ).tolist()
        for _, group in specialist_long.groupby("fixed_case_id", sort=False)
    ]
    alpha = krippendorff_alpha_nominal(ratings_by_case)
    alpha_ci = bootstrap_alpha(ratings_by_case, N_BOOTSTRAP, RANDOM_SEED)
    alpha.update(alpha_ci_low=alpha_ci[0], alpha_ci_high=alpha_ci[1])

    def alpha_strata(column):
        rows = []
        for value, group in specialist_long.groupby(column, dropna=False):
            units = [g.apply(lambda r: None if r.abstain else r.answer_letter, axis=1).tolist() for _, g in group.groupby("fixed_case_id")]
            result = krippendorff_alpha_nominal(units)
            rows.append({"stratum": column, "value": value, **result})
        return rows

    specialist_long["number_specialists"] = specialist_long.groupby("fixed_case_id").fixed_case_id.transform("size")
    alpha_by_stratum = pd.DataFrame(
        alpha_strata("difficulty") + alpha_strata("number_specialists") +
        alpha_strata("specialty") + alpha_strata("critic_used")
    )
    alpha_by_stratum.to_csv(OUTPUT_DIR / "krippendorff_alpha_by_stratum.csv", index=False)

    has_rankings = any("ranking" in opinion for value in multi.specialist_opinions for opinion in normalize_specialist_opinions(value)[0])
    kendall_status = "available" if has_rankings else "unavailable_historical_outputs_have_no_full_rankings"
    print("Overall alpha:", alpha)
    print("Kendall W:", kendall_status)
    print("Future specialist schema: answer, confidence, option_scores {A:..., B:...}, ranking [B,A,C,D]")
    display(alpha_by_stratum)
    """),
    md("""
    ## 11. Proposed dynamic disagreement trigger

    This proposal is not installed in production. Thresholds are visible and may
    be tuned on development only. V2 has no Router confidence, so that trigger is
    skipped when absent. Every decision retains explicit reasons.
    """),
    code("""
    trigger_rows = []
    for row in case_analysis.to_dict("records"):
        result = create_disagreement_trigger(row, DISAGREEMENT_CONFIG)
        trigger_rows.append({"fixed_case_id": row["fixed_case_id"], **result})
    trigger_table = pd.DataFrame(trigger_rows)
    trigger_table["reasons"] = trigger_table.reasons.map(json.dumps)
    trigger_table.to_csv(OUTPUT_DIR / "proposed_disagreement_triggers.csv", index=False)
    display(trigger_table.head())
    """),
    md("""
    ## 12-14. Paired single vs multi performance, McNemar, and calibration

    Comparison is an inner one-to-one join on fixed case ID and gold answer,
    with matching split and fingerprints required. Small discordant samples use
    exact binomial McNemar; larger samples use continuity-corrected chi-square.
    Brier score is the primary calibration metric and uses valid answered cases.
    """),
    code("""
    paired, mcnemar = paired_single_multi(single, multi, N_BOOTSTRAP, RANDOM_SEED)
    paired.to_csv(OUTPUT_DIR / "paired_single_multi_cases.csv", index=False)
    pd.DataFrame([mcnemar]).to_csv(OUTPUT_DIR / "mcnemar_result.csv", index=False)

    def accuracy_ci(frame):
        return bootstrap_statistic(frame, lambda x: float(x.correct.astype(bool).mean()), N_BOOTSTRAP, RANDOM_SEED)

    system_rows = []
    for name, frame in (("single_gpt5", single), ("multi_dynamic_one_gpt5_v2", multi)):
        valid = frame[frame.predicted_letter.astype(str).isin(VALID_LETTERS)].copy()
        valid["correct"] = valid.correct.astype(bool)
        ci = accuracy_ci(valid)
        system_rows.append({"system": name, "n": len(valid), "accuracy": valid.correct.mean(), "ci_low": ci[0], "ci_high": ci[1], **calibration_summary(valid.confidence, valid.correct)})
    system_performance = pd.DataFrame(system_rows)

    specialist_calibration = calibration_summary(specialist_long.loc[~specialist_long.abstain, "confidence"], specialist_long.loc[~specialist_long.abstain, "correct"])
    calibration_tables = {
        "single_gpt5": calibration_table(single.confidence, single.correct),
        "multi_judge": calibration_table(multi.confidence, multi.correct),
        "specialists": calibration_table(specialist_long.loc[~specialist_long.abstain, "confidence"], specialist_long.loc[~specialist_long.abstain, "correct"]),
    }
    for name, table in calibration_tables.items(): table.to_csv(OUTPUT_DIR / f"calibration_{name}.csv", index=False)
    display(system_performance)
    display(pd.DataFrame([mcnemar]))
    display(pd.DataFrame([{"system": "individual_specialists", **specialist_calibration}]))
    """),
    md("""
    ## 15-18. Agreement predicts correctness, resolution value, strata, and bootstrap uncertainty

    Predictor direction is explicit: agreement/modal fraction should increase
    with correctness; entropy/JSD should decrease. AUROC confidence intervals
    use case-level bootstrap resampling. Small strata are shown with `n` and
    should not be overinterpreted.
    """),
    code("""
    predictor_specs = [
        ("legacy_agreement_score", "higher_is_better"),
        ("modal_vote_fraction", "higher_is_better"),
        ("vote_entropy", "lower_is_better"),
        ("mean_pairwise_jsd", "lower_is_better"),
    ]
    predictor_table = pd.DataFrame([
        evaluate_agreement_predictor(case_analysis, metric, direction, N_BOOTSTRAP, RANDOM_SEED)
        for metric, direction in predictor_specs
    ])
    predictor_table.to_csv(OUTPUT_DIR / "agreement_predicts_correctness.csv", index=False)

    disagreement = case_analysis[case_analysis.number_unique_answers.gt(1)].copy()
    disagreement["majority_correct"] = disagreement.modal_answer == disagreement.reference_letter
    disagreement["judge_correct"] = disagreement.correct.astype(bool)
    disagreement["transition"] = np.select(
        [~disagreement.majority_correct & disagreement.judge_correct,
         disagreement.majority_correct & disagreement.judge_correct,
         disagreement.majority_correct & ~disagreement.judge_correct],
        ["wrong -> correct", "correct -> correct", "correct -> wrong"], default="wrong -> wrong")
    resolution_table = disagreement.groupby(["critic_used", "transition"], dropna=False).size().rename("n").reset_index()
    resolution_summary = pd.DataFrame([{
        "initial_specialist_state": "categorical_disagreement", "n": len(disagreement),
        "majority_accuracy": disagreement.majority_correct.mean(),
        "final_judge_accuracy": disagreement.judge_correct.mean(),
        "critic_judge_accuracy": disagreement.loc[disagreement.critic_used.astype(bool), "judge_correct"].mean(),
        "judge_selected_correct_minority": int(((disagreement.predicted_letter != disagreement.modal_answer) & disagreement.judge_correct).sum()),
    }])
    resolution_table.to_csv(OUTPUT_DIR / "resolution_transitions.csv", index=False)

    strata = []
    for column in ["difficulty", "n_specialists", "critic_used", "unanimous"]:
        if column in case_analysis:
            for value, group in case_analysis.groupby(column, dropna=False):
                strata.append({"stratum": column, "value": value, "n": len(group), "multi_accuracy": group.correct.mean(), "mean_modal_vote_fraction": group.modal_vote_fraction.mean(), "mean_jsd": group.mean_pairwise_jsd.mean()})
    stratified = pd.DataFrame(strata)
    stratified.to_csv(OUTPUT_DIR / "stratified_analysis.csv", index=False)
    display(predictor_table)
    display(resolution_summary)
    display(resolution_table)
    display(stratified)
    """),
    md("""
    ## 19. Required summary tables

    All tables retain the active split label. Metrics unavailable from historical
    schemas are represented as unavailable/NaN, never fabricated.
    """),
    code("""
    table1 = pd.DataFrame([{
        "split": RUN_SPLIT, "cases": len(case_analysis), "specialist_responses": len(specialist_long),
        "average_specialists_per_case": len(specialist_long) / len(case_analysis),
        "abstention_rate": specialist_long.abstain.mean(),
        "difficulty_distribution": json.dumps(case_analysis.difficulty.value_counts().to_dict()),
    }])
    table2 = pd.DataFrame([
        {"split": RUN_SPLIT, "metric": "Legacy confidence agreement", "value": case_analysis.legacy_agreement_score.mean(), "availability": "available"},
        {"split": RUN_SPLIT, "metric": "Krippendorff alpha", "value": alpha["krippendorff_alpha"], "availability": "available"},
        {"split": RUN_SPLIT, "metric": "Kendall W", "value": np.nan, "availability": kendall_status},
        {"split": RUN_SPLIT, "metric": "Modal vote fraction", "value": case_analysis.modal_vote_fraction.mean(), "availability": "available"},
        {"split": RUN_SPLIT, "metric": "Vote entropy", "value": case_analysis.vote_entropy.mean(), "availability": "available"},
        {"split": RUN_SPLIT, "metric": "JSD", "value": case_analysis.mean_pairwise_jsd.mean(), "availability": ",".join(case_analysis.jsd_probability_source.unique())},
    ])
    table3 = system_performance.assign(split=RUN_SPLIT)
    table4 = predictor_table.assign(split=RUN_SPLIT)
    table5 = resolution_summary.assign(split=RUN_SPLIT)
    for number, table in enumerate([table1, table2, table3, table4, table5], 1):
        table.to_csv(OUTPUT_DIR / f"table_{number}.csv", index=False)
        print(f"Table {number}"); display(table)
    """),
    md("""
    ## 20. Research-quality figures

    Figures are descriptive and always labeled with the analyzed split.
    """),
    code("""
    plt.style.use("seaborn-v0_8-whitegrid")
    figure_specs = []
    def save(name):
        plt.suptitle(f"Split: {RUN_SPLIT}", fontsize=9, x=0.99, ha="right")
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=180, bbox_inches="tight"); plt.show()

    case_analysis.legacy_agreement_score.hist(bins=15, color="#176B87")
    plt.xlabel("Legacy confidence agreement"); plt.ylabel("Cases"); save("01_legacy_agreement_distribution")
    if case_analysis.mean_pairwise_jsd.notna().any():
        case_analysis.mean_pairwise_jsd.hist(bins=15, color="#D97706")
        plt.xlabel("Mean pairwise JSD (base 2)"); plt.ylabel("Cases"); save("02_jsd_distribution")
    case_analysis.groupby("legacy_agreement_level").correct.mean().plot(kind="bar", color="#176B87")
    plt.ylabel("Final accuracy"); plt.xlabel("Legacy agreement level"); save("03_accuracy_by_agreement")
    system_performance.set_index("system").accuracy.plot(kind="bar", color=["#65758B", "#0F766E"])
    plt.ylabel("Accuracy"); plt.ylim(0, 1); save("04_single_vs_multi_accuracy")
    predictor_table.dropna(subset=["auroc"]).set_index("metric").auroc.plot(kind="barh", color="#B45309")
    plt.xlabel("AUROC for final correctness"); plt.xlim(0, 1); save("05_agreement_predictor_auroc")
    for name, table in calibration_tables.items():
        valid = table[table.n > 0]; plt.plot(valid.mean_confidence, valid.observed_accuracy, marker="o", label=name)
    plt.plot([0,1], [0,1], "--", color="black", linewidth=1); plt.legend(); plt.xlabel("Mean confidence"); plt.ylabel("Observed accuracy"); save("06_calibration")
    if "difficulty" in case_analysis:
        merged = case_analysis[["fixed_case_id", "difficulty", "correct"]].merge(paired[["fixed_case_id", "single_correct"]], on="fixed_case_id")
        merged.groupby("difficulty")[["single_correct", "correct"]].mean().rename(columns={"correct":"multi_correct"}).plot(kind="bar")
        plt.ylabel("Accuracy"); plt.ylim(0,1); save("07_accuracy_by_difficulty")
    case_analysis.groupby("critic_used").correct.mean().plot(kind="bar", color="#7C3E22")
    plt.ylabel("Final accuracy"); plt.ylim(0,1); save("08_accuracy_by_critic")
    """),
    md("""
    ## 21-22. Empirical findings and recommended strategy

    Recommendations below are generated only when empirical values are
    available. They do not replace production logic.
    """),
    code("""
    available_predictors = predictor_table.dropna(subset=["auroc"])
    best = available_predictors.sort_values("auroc", ascending=False).iloc[0] if len(available_predictors) else None
    improvement = mcnemar["absolute_accuracy_difference"]
    significant = mcnemar["mcnemar_p_value"] < 0.05
    summary = {
        "split": RUN_SPLIT,
        "specialist_reliability_alpha": alpha["krippendorff_alpha"],
        "best_observed_error_indicator": best.metric if best is not None else "unavailable",
        "best_predictor_auroc": best.auroc if best is not None else np.nan,
        "specialist_confidence_brier": specialist_calibration["brier_score"],
        "critic_disagreement_cases": int(disagreement.critic_used.astype(bool).sum()) if len(disagreement) else 0,
        "judge_disagreement_accuracy": disagreement.judge_correct.mean() if len(disagreement) else np.nan,
        "multi_minus_single_accuracy": improvement,
        "mcnemar_p_value": mcnemar["mcnemar_p_value"],
        "statistically_significant_at_0_05": significant,
        "recommended_dataset_level_metric": "Krippendorff nominal alpha",
        "recommended_case_level_metric": (best.metric if best is not None else "categorical vote features; collect true option probabilities before selecting JSD"),
        "recommended_confidence_metric": "Brier score",
        "recommended_system_comparison": "Paired accuracy difference with McNemar test",
    }
    (OUTPUT_DIR / "empirical_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    display(pd.Series(summary, name="result").to_frame())
    print("No superiority claim should be made unless the locked test effect and uncertainty support it.")
    """),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
nbf.write(notebook, TARGET)
print(f"Created {TARGET}")
