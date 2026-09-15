import json
from pathlib import Path
import pandas as pd

from thesis_pipeline import initialize_thesis
from thesis_pipeline.guards import assert_identical_final_inputs
from thesis_pipeline.methods import select_method
from thesis_pipeline.reproducibility import assignment
from thesis_pipeline.checkpoint import append_checkpoint, load_completed

def make_csv(path: Path, n=10):
    pd.DataFrame({"id": range(n), "question": [f"Q {i}\nA. x\nB. y" for i in range(n)], "context": ["" for _ in range(n)], "answer": ["A" for _ in range(n)]}).to_csv(path,index=False)

def spec(path): return {"name":"switchable","path":str(path),"columns":{"id":"id","instruction":"question","input":"context","gold":"answer"}}

def test_dataset_switching_and_fingerprints(tmp_path):
    first=tmp_path/"first.csv"; second=tmp_path/"second.csv"; make_csv(first); make_csv(second); pd.read_csv(second).assign(answer=["B"]+ ["A"]*9).to_csv(second,index=False)
    one=initialize_thesis(spec(first),RUN_MODE="test",GLOBAL_SEED=42,output_dir=tmp_path/"out1",mock=True)
    two=initialize_thesis(spec(second),RUN_MODE="test",GLOBAL_SEED=42,output_dir=tmp_path/"out2",mock=True)
    assert one.bundle.dataset_fingerprint==two.bundle.dataset_fingerprint
    assert one.bundle.label_fingerprint!=two.bundle.label_fingerprint

def test_assignment_is_deterministic_and_records_model():
    x=assignment("case","specialist",42,"gpt","claude")
    assert x==assignment("case","specialist",42,"gpt","claude")
    assert x["assigned_model"] in {"gpt","claude"}

def test_selection_is_development_only_and_frozen():
    result=select_method([{"method":"a","accuracy":.9,"critical_safety_error_rate":.2,"macro_f1":.8,"extra_llm_calls":0},{"method":"b","accuracy":.9,"critical_safety_error_rate":.1,"macro_f1":.7,"extra_llm_calls":1}])
    assert result["selected_method"]=="b" and result["selection_split"]=="development"

def test_final_inputs_and_checkpoint_resume(tmp_path):
    frame=pd.DataFrame({"case_id":["1"],"gold":["A"]})
    assert_identical_final_inputs({"gpt":frame,"claude":frame.copy()})
    path=tmp_path/"checkpoint.jsonl"; append_checkpoint(path,{"case_id":"1","status":"success"}); assert load_completed(path)["1"]["status"]=="success"
