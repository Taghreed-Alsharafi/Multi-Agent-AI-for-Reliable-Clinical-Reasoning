# Thesis Prompt Register

**Updated: September 15, 2026**

This register identifies the complete prompt sources used by the live system and the
thesis experiments. Prompt text remains in executable source files so the documented
method and the evaluated implementation cannot drift apart.

## Prompt inventory

| Prompt group | Roles | Version or source |
|---|---|---|
| Adaptive MCQ evaluation | Router, specialist, critic, judge, single-agent baseline | `evaluation/mcq_v2_prompts.py`, version `v2.1` |
| Final agreement experiment | Router, specialist, pre-decision safety, judge, final safety | `evaluation/final_agreement_experiment.py`, experiment version `dynamic_mixed_models_care_v2_v6` |
| Document-grounded baselines | Single GPT and Claude baselines | `evaluation/prompts.py` |
| Strategy-diverse panel | Differential, evidence, safety, and guideline perspectives | `thesis_pipeline/multi_agent.py` |
| Live supervisor | Dynamic specialty selection and lead assignment | `skills/medical-supervisor/SKILL.md` |
| Live specialists | Independent specialty-specific evidence review | `skills/medical-specialist/SKILL.md` |
| Live judge | Conflict resolution and consolidated conclusion | `skills/medical-judge/SKILL.md` |
| Live safety review | Grounding, contradiction, and unsupported-claim checks | `skills/medical-safety/SKILL.md` |

## Prompt design principles

- The router selects the smallest clinically useful team and does not answer the case.
- Specialists reason independently and cannot see one another's decisions.
- The critic examines meaningful disagreement without acting as a majority voter.
- The judge resolves conflicts from the original evidence and specialty relevance.
- The safety stages check unsupported claims, contradictions, and high-risk outcomes.
- Baseline prompts use the same case text, answer options, and structured-output rules.
- Gold labels are never inserted into model prompts.
- Prompt versions are recorded in run metadata and frozen before held-out evaluation.

## Reproducibility

The executable prompt constants, structured schemas, model assignments, dataset
fingerprints, and prompt-version identifiers are stored with each experimental run.
Historical prompt wording is retained for reproducibility even when the current live
application uses a newer unified model configuration.

These prompts support a research benchmark only. They are not medical instructions
and must not be used for patient-care decisions.
