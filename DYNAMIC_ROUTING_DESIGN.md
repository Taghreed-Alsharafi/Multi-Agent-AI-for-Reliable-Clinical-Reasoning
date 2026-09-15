# Dynamic Domain Routing with One Strong-Model Call

## Final per-case architecture

1. **Router — gpt-4o-mini**: reads the stem/options only, identifies clinical domains, determines complexity, chooses the smallest nonredundant set of 1–4 true clinical specialties, and names the lead specialty. It never answers the MCQ.
2. **Domain specialists — gpt-4o-mini**: each selected specialty independently evaluates the case and returns an option, confidence, stem evidence, strongest alternative, and uncertainty.
3. **Conditional critic — gpt-4o-mini**: runs only for specialist disagreement, low specialist confidence, low router confidence, or high-complexity routing. It diagnoses conflicts rather than voting.
4. **Final Judge — GPT-5**: exactly one strong-model call per case. It independently re-checks the stem and adjudicates evidence by domain relevance and quality, not majority vote.
5. **Independent Auditor — gpt-4o-mini**: audits option validity, evidence support, contradictions, unsupported facts, and excessive confidence. It can approve, cap confidence, or reject to abstention. It cannot trigger a second GPT-5 call.

## Why GPT-5 is assigned to the Judge

The Judge has the highest information-integration burden: it sees the original case plus all independent domain opinions and, when needed, the critic's conflict analysis. Assigning the single strong-model call here maximizes its value while keeping routing and parallel evidence generation inexpensive.

## Primary comparisons

- Hybrid multi-agent vs **single GPT-5**: controls the number of strong-model calls (one per case) and tests the added value of routing, specialty decomposition, conflict review, and audit.
- Hybrid multi-agent vs **single gpt-4o-mini**: measures improvement over a low-cost baseline.

## Frozen 20/80 protocol

All optimization occurs on the fixed 20% development split. The content-hash manifest protects the split against accidental reshuffling. After prompts, models, routing rules, thresholds, and specialist limits are frozen, run the 80% test split once with no further changes.
