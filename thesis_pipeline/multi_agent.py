"""Improved multi-agent MCQ system: diverse-reasoning panel + selective judge.

Design rationale
----------------
The original panel ran N agents with one identical prompt on two similar models,
so their errors were correlated and the vote could not exceed the base models.
This module decorrelates the panel by giving each agent a distinct reasoning
strategy, anchors the panel on the strongest available model, aggregates by
confidence-weighted vote, and calls a strong Judge (which sees every agent's
letter and rationale) ONLY when the panel disagrees -- the hard cases where
plain voting loses. The judge-trigger policy is tuned on development data only.

All calls honor VERIFY_SSL and run at temperature 0 (the panel diversity comes
from the prompts, not from sampling), so the system is deterministic and
reproducible.
"""
from __future__ import annotations
import asyncio, json, os, re
from collections import defaultdict
from pathlib import Path
import pandas as pd

from evaluation.providers import _json_object, _verify_ssl, _timeout
from .live import _case, _label, _bounded_gather
from .checkpoint import load_completed, append_checkpoint

# ---------------------------------------------------------------- prompts -----
_JSON_CONTRACT = (
    'Return ONLY compact JSON, no prose or fences: '
    '{"letter":"<one allowed letter>","confidence":0.0,"rationale":"<=25 words"}. '
    '"letter" must be exactly one supplied option letter. Do not abstain.')

STRATEGY_PROMPTS = {
    "direct": "You are a decisive clinical diagnostician. Identify the single most "
              "likely diagnosis/answer from the stem and pick the matching option. " + _JSON_CONTRACT,
    "eliminate": "You are a careful clinician who reasons by exclusion. Rule out each "
                 "option that is contradicted by the stem, then choose the one that "
                 "survives. " + _JSON_CONTRACT,
    "mechanism": "You are a pathophysiology expert. Reason from the underlying "
                 "mechanism / first principles to the option it best supports. " + _JSON_CONTRACT,
    "second_opinion": "You are an independent senior clinician giving a fresh second "
                      "opinion. Reason from scratch and choose the best option. " + _JSON_CONTRACT,
}

# (strategy, provider) -- 3 diverse strong-model voices + 1 independent Claude voice.
PANEL = [("direct", "openai"), ("eliminate", "openai"),
         ("mechanism", "openai"), ("second_opinion", "anthropic")]

JUDGE_SYSTEM = (
    "You are the final clinical adjudicator for a forced-choice MCQ. Several experts, "
    "each using a different reasoning strategy, have answered. A majority can be wrong "
    "and high confidence does not guarantee correctness -- weigh the QUALITY of the "
    "reasoning against the stem, not just the vote count. Choose exactly ONE supplied "
    "option and do not abstain. " + _JSON_CONTRACT)


# --------------------------------------------------------------- clients ------
def _openai_client():
    import httpx
    from openai import AsyncOpenAI
    hc = None if _verify_ssl() else httpx.AsyncClient(verify=False, timeout=_timeout())
    return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), **({"http_client": hc} if hc else {}))


def _anthropic_client():
    import httpx
    from anthropic import AsyncAnthropic
    hc = None if _verify_ssl() else httpx.AsyncClient(verify=False, timeout=_timeout())
    return AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), **({"http_client": hc} if hc else {}))


async def _ask_openai(client, model, system, user):
    r = await client.chat.completions.create(
        model=model, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content or "{}"


async def _ask_anthropic(client, model, system, user):
    r = await client.messages.create(model=model, max_tokens=600, temperature=0,
        system=system, messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


def _parse(raw, allowed):
    try:
        obj = _json_object(raw)
    except Exception:
        obj = {}
    letter = str(obj.get("letter") or obj.get("predicted_label") or obj.get("answer") or "").strip().upper()
    if letter not in allowed:
        letter = _label(letter, allowed)
    try:
        conf = float(obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return letter, max(0.0, min(1.0, conf)), str(obj.get("rationale", ""))[:300]


def _question(row) -> str:
    c = _case(row)  # options are already embedded in the question stem
    return f"## Question\n{c.question}\n\n## Allowed labels\n{c.allowed_labels}"


# --------------------------------------------------------------- panel --------
async def _run_panel(row, config, oa, an):
    c = _case(row); allowed = c.allowed_labels; user = _question(row)
    async def one(strategy, provider):
        sysmsg = STRATEGY_PROMPTS[strategy]
        try:
            if provider == "openai":
                raw = await _ask_openai(oa, config.gpt_model, sysmsg, user)
            else:
                raw = await _ask_anthropic(an, config.claude_model, sysmsg, user)
            letter, conf, rat = _parse(raw, allowed)
            return {"strategy": strategy, "provider": provider,
                    "model": config.gpt_model if provider == "openai" else config.claude_model,
                    "answer": letter, "confidence": conf, "rationale": rat,
                    "status": "success" if letter else "invalid_output"}
        except Exception as exc:
            return {"strategy": strategy, "provider": provider, "answer": "",
                    "confidence": 0.0, "rationale": "", "status": "api_error", "error": str(exc)[:200]}
    return list(await asyncio.gather(*[one(s, p) for s, p in PANEL]))


def _weighted_vote(outputs):
    tally = defaultdict(float); votes = defaultdict(int)
    for o in outputs:
        if o["answer"]:
            tally[o["answer"]] += float(o.get("confidence", 0.5)); votes[o["answer"]] += 1
    if not tally:
        return "", 0.0, 0
    ranked = sorted(tally, key=lambda k: (-tally[k], -votes[k], k))
    top = ranked[0]
    margin = tally[top] - (tally[ranked[1]] if len(ranked) > 1 else 0.0)
    return top, margin, len(tally)


async def _judge(row, config, outputs, oa):
    c = _case(row); allowed = c.allowed_labels
    panel = "\n".join(f"- Expert ({o['strategy']}): answer {o['answer']} "
                      f"(confidence {o.get('confidence',0):.2f}) -- {o.get('rationale','')}"
                      for o in outputs if o["answer"])
    user = f"## Question\n{c.question}\n\n## Allowed labels\n{allowed}\n\n## Expert opinions\n{panel}"
    raw = await _ask_openai(oa, config.gpt_model, JUDGE_SYSTEM, user)
    letter, conf, rat = _parse(raw, allowed)
    return letter, rat


def decide(outputs, allowed, judge_letter=None, distinct_trigger=2):
    """Combine panel outputs. If distinct answers >= trigger, the judge decides.

    ``distinct_trigger`` is the number of DISTINCT panel answers at or above which
    the judge is consulted (2 = judge on any disagreement). Tuned on development.
    """
    top, margin, n_distinct = _weighted_vote(outputs)
    if not top:
        return "", {"method": "no_valid_answers", "judge_used": False, "n_distinct": 0}
    if n_distinct >= distinct_trigger and judge_letter:
        return judge_letter, {"method": "judge", "judge_used": True, "n_distinct": n_distinct,
                              "weighted_vote": top}
    return top, {"method": "unanimous" if n_distinct == 1 else "weighted_vote",
                 "judge_used": False, "n_distinct": n_distinct}


async def run_improved_multi(cases: pd.DataFrame, config, checkpoint: Path,
                             distinct_trigger: int = 2, precompute_all_judge: bool = False) -> pd.DataFrame:
    """Run the diverse panel + selective judge over cases (checkpointed).

    When ``precompute_all_judge`` is True the judge is computed for EVERY case
    (including unanimous ones), so a development sweep can compare vote-only,
    judge-on-disagreement, and always-judge (mixture-of-agents) policies offline
    without further API calls. For the frozen test run, leave it False so the
    judge is called only when the chosen trigger needs it.
    """
    done = load_completed(checkpoint)
    oa, an = _openai_client(), _anthropic_client()
    async def handle(row):
        cid = str(row.case_id)
        if cid in done and done[cid].get("status") == "success":
            return done[cid]
        allowed = _case(row).allowed_labels
        outputs = await _run_panel(row, config, oa, an)
        _, _, n_distinct = _weighted_vote(outputs)
        judge_letter = ""
        need_judge = precompute_all_judge or (n_distinct >= max(2, distinct_trigger))
        if n_distinct >= 1 and need_judge:
            try:
                judge_letter, _ = await _judge(row, config, outputs, oa)
            except Exception:
                judge_letter = ""
        prediction, meta = decide(outputs, allowed, judge_letter, distinct_trigger)
        rec = {"case_id": cid, "prediction": prediction, "agent_outputs": outputs,
               "judge_letter": judge_letter, "critical_safety_error": False,
               "status": "success" if prediction else "api_error",
               "system": "improved_multi_agent", **meta}
        append_checkpoint(checkpoint, rec)
        return rec
    rows = await _bounded_gather([(lambda r=row: handle(r)) for _, row in cases.iterrows()])
    return pd.DataFrame(rows)


def simulate_trigger(records: list[dict], gold_by_id: dict, distinct_trigger: int) -> dict:
    """Offline: score a given trigger from precomputed panel+judge records."""
    correct = 0; n = 0; judged = 0
    for r in records:
        allowed = sorted({o["answer"] for o in r.get("agent_outputs", []) if o.get("answer")})
        pred, meta = decide(r.get("agent_outputs", []), allowed, r.get("judge_letter", ""), distinct_trigger)
        if meta["judge_used"]:
            judged += 1
        gold = str(gold_by_id.get(str(r["case_id"]), "")).upper()
        if gold:
            n += 1; correct += int(str(pred).upper() == gold)
    return {"distinct_trigger": distinct_trigger, "accuracy": correct / n if n else float("nan"),
            "n": n, "judge_rate": judged / len(records) if records else 0.0}
