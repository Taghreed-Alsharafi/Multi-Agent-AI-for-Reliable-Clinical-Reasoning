import json
import pandas as pd
import pytest
from evaluation import mcq_hybrid as mh

@pytest.mark.asyncio
async def test_multi_uses_gpt5_exactly_once(monkeypatch, tmp_path):
    calls=[]
    async def fake_call(model, system, user, max_tokens=700, temperature=0.0, reasoning_effort=None):
        calls.append(model)
        usage={"input_tokens":10,"output_tokens":5}
        if "routing supervisor" in system:
            obj={"case_domains":["Dermatology"],"complexity":"moderate","routing_confidence":0.95,"n_specialists":2,"lead_specialty":"Pediatric Dermatology","specialists":[
                {"specialty":"Pediatric Dermatology","reason":"rash","focus":"distribution"},
                {"specialty":"Pediatric Infectious Disease","reason":"household itch","focus":"transmission"}]}
        elif "domain expert" in system:
            obj={"answer_letter":"A","confidence":0.9,"abstain":False,"key_evidence":["palms and soles"],"reasoning_summary":"supports scabies","strongest_alternative":"B","alternative_rejected_because":"distribution","uncertainties":[]}
        elif "final adjudicating clinician" in system:
            obj={"answer_letter":"A","confidence":0.92,"abstain":False,"explanation":"best fit","decisive_evidence":["household pruritus"],"dissent_summary":""}
        elif "independent clinical answer auditor" in system:
            obj={"decision":"approve","confidence_cap":0.95,"issues":[],"audit_explanation":"supported"}
        else:
            obj={"conflict_present":False,"issues":[],"most_relevant_specialty":"Pediatric Dermatology","resolution_guidance":"","critic_confidence":0.9}
        return obj,json.dumps(obj),0.01,usage
    monkeypatch.setattr(mh,"call_openai_json",fake_call)
    df=pd.DataFrame([{"fixed_case_id":"x","instruction":"Infant rash.\nA. Scabies\nB. Eczema","input":"","reference_letter":"A"}])
    out=await mh.run_multi_agent(df,"test_multi","development",tmp_path,resume=False)
    assert out.loc[0,"gpt5_call_count"]==1
    assert calls.count("gpt-5")==1
    assert out.loc[0,"correct"]


def test_metrics_do_not_count_abstentions_as_invalid_answers():
    frame = pd.DataFrame([
        {"reference_letter": "A", "predicted_letter": "", "abstain": True, "correct": False, "confidence": 0.0},
        {"reference_letter": "B", "predicted_letter": "B", "abstain": False, "correct": True, "confidence": 0.9},
    ])
    summary = mh.metrics(frame)
    assert summary["abstention_rate"] == 0.5
    assert summary["invalid_answer_rate"] == 0.0


def test_reference_letter_handles_embedded_final_answer_text():
    question = (
        "Please answer the following multiple-choice question:\n"
        "A 40 yr old female patient presents with depression.\n"
        "A. No treatment is necessary\n"
        "B. SSRI is the most efficacious\n"
        "C. Antidepressant treatment based on the side effect profile of the durgs\n"
        "D. Combination therapy of 2 antidepressant drugs"
    )
    output = (
        "Evaluation of Candidate Answers:\n"
        "Final Answer: Antidepressant treatment based on the side effect profile of the drugs."
    )
    assert mh.reference_letter(question, output) == "C"
