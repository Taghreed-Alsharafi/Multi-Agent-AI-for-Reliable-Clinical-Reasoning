from __future__ import annotations
import json, os, time
from dataclasses import dataclass
from typing import Any
from .prompts import case_prompt, system_prompt_for
from .schema import EvaluationCase, StandardAnswer

@dataclass
class ProviderResult:
    answer: StandardAnswer
    raw: str
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None


def _verify_ssl() -> bool:
    """Honor the project-wide VERIFY_SSL flag (see config/settings.py and .env).

    Behind a TLS-intercepting corporate proxy the SDKs' default httpx clients
    raise ``APIConnectionError`` on every call, which the callers turn into empty
    predictions and a misleading accuracy of 0. Setting VERIFY_SSL=false in .env
    restores connectivity, matching the behavior of agents/base.py.
    """
    return os.environ.get("VERIFY_SSL", "true").strip().lower() not in {"false", "0", "no", "off"}


def _timeout() -> float:
    try:
        return float(os.environ.get("REQUEST_TIMEOUT", "60"))
    except (TypeError, ValueError):
        return 60.0


def _openai_http_client():
    # Only build an explicit client when we must disable verification; otherwise
    # let the SDK use its own default (which supports its retry/transport tuning).
    if _verify_ssl():
        return None
    import httpx
    return httpx.AsyncClient(verify=False, timeout=_timeout())


def _anthropic_http_client():
    if _verify_ssl():
        return None
    import httpx
    return httpx.AsyncClient(verify=False, timeout=_timeout())

import re as _re

def _json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:].lstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Salvage: some models wrap the object in prose or emit an evidence string
    # with an unescaped quote/comma that breaks strict JSON. Try the outermost
    # {...} span, then fall back to pulling the predicted label/answer letter so a
    # usable forced-choice answer is not lost to a formatting slip.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    m = _re.search(r'"(?:predicted_label|answer)"\s*:\s*"?([A-H])"?', text, _re.I)
    if m:
        conf = _re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        return {"answer": m.group(1).upper(), "predicted_label": m.group(1).upper(),
                "confidence": float(conf.group(1)) if conf else 0.5}
    raise json.JSONDecodeError("Could not parse a JSON answer object from the model response", text, 0)

async def call_openai(case: EvaluationCase, model: str, temperature: float) -> ProviderResult:
    from openai import AsyncOpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot call the OpenAI API.")
    http_client = _openai_http_client()
    client = AsyncOpenAI(api_key=api_key, **({"http_client": http_client} if http_client else {})); started = time.perf_counter()
    response = await client.chat.completions.create(
        model=model, temperature=temperature, response_format={"type":"json_object"},
        messages=[{"role":"system","content":system_prompt_for(case.documents, case.allowed_labels)},
                  {"role":"user","content":case_prompt(case.question, case.documents, case.allowed_labels)}])
    raw = response.choices[0].message.content or "{}"; usage = response.usage
    return ProviderResult(StandardAnswer.model_validate(_json_object(raw)), raw,
        time.perf_counter()-started, getattr(usage,"prompt_tokens",None),
        getattr(usage,"completion_tokens",None))

async def call_anthropic(case: EvaluationCase, model: str, temperature: float) -> ProviderResult:
    from anthropic import AsyncAnthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; cannot call the Anthropic API.")
    http_client = _anthropic_http_client()
    client = AsyncAnthropic(api_key=api_key, **({"http_client": http_client} if http_client else {})); started = time.perf_counter()
    response = await client.messages.create(model=model, max_tokens=1200,
        temperature=temperature, system=system_prompt_for(case.documents, case.allowed_labels),
        messages=[{"role":"user","content":case_prompt(case.question, case.documents, case.allowed_labels)}])
    raw = "".join(b.text for b in response.content if getattr(b,"type","")=="text")
    return ProviderResult(StandardAnswer.model_validate(_json_object(raw)), raw,
        time.perf_counter()-started, getattr(response.usage,"input_tokens",None),
        getattr(response.usage,"output_tokens",None))
