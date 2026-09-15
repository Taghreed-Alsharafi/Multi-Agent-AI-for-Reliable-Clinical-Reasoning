SINGLE_AGENT_SYSTEM_PROMPT = """You are a clinical reasoning research assistant.
Answer only from the supplied clinical documents. Do not add absent facts. If the
documents cannot answer the question, abstain. Evidence quotes must be exact source
substrings. Identify safety concerns explicitly. Return only JSON:
{"answer":"concise answer","evidence_quotes":[],"confidence":0.0,
"insufficient_information":false,"predicted_label":null,"safety_flags":[],
"is_safe":true}. If allowed classification labels are supplied, predicted_label must
be exactly one of them or null when abstaining. Research only; not clinical advice."""

# Forced-choice multiple-choice prompt. Used when a case supplies allowed labels
# but no supporting documents (a standalone MCQ). The document-grounded prompt
# above tells the model to abstain when there are no documents, which for an MCQ
# produces an empty prediction and a misleadingly low score. This prompt keeps the
# identical JSON contract but instructs a forced choice from the supplied options.
MCQ_SINGLE_AGENT_SYSTEM_PROMPT = """You are a clinical reasoning research assistant
answering a forced-choice multiple-choice question. Use your clinical knowledge to
choose exactly ONE of the supplied answer options. Do not abstain and do not invent
an option that was not supplied. Return ONLY a compact JSON object and nothing else
-- no prose before or after, no markdown fences, no evidence quotes:
{"predicted_label":"<letter>","answer":"<letter>","confidence":0.0}
Both "answer" and "predicted_label" must be exactly one of the supplied allowed
labels (a single letter). "confidence" is a number in 0..1. Research only; not
clinical advice."""


def is_mcq_case(documents: list[str] | None, allowed_labels: list[str] | None) -> bool:
    """A standalone forced-choice MCQ: labels supplied, no grounding documents."""
    return bool(allowed_labels) and not (documents and any(str(d).strip() for d in documents))


def system_prompt_for(documents: list[str] | None, allowed_labels: list[str] | None) -> str:
    return MCQ_SINGLE_AGENT_SYSTEM_PROMPT if is_mcq_case(documents, allowed_labels) else SINGLE_AGENT_SYSTEM_PROMPT


def case_prompt(question: str, documents: list[str], allowed_labels: list[str] | None = None) -> str:
    labels = f"\n\n## Allowed labels\n{allowed_labels}" if allowed_labels else ""
    if is_mcq_case(documents, allowed_labels):
        return f"## Question\n{question}{labels}"
    joined = "\n\n--- DOCUMENT BREAK ---\n\n".join(documents)
    return f"## Question\n{question}{labels}\n\n## Clinical documents\n{joined}"
