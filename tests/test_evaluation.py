from evaluation.metrics import classification_metrics,evaluate_predictions,paired_bootstrap_difference,quote_is_grounded,token_f1
from evaluation.config import ExperimentConfig
from evaluation.schema import EvaluationCase,PredictionRecord

def test_grounding():
    assert quote_is_grounded("persistent nausea",["Persistent nausea was documented."])
    assert not quote_is_grounded("chest pain",["Persistent nausea was documented."])

def test_token_f1():
    assert token_f1("metformin stopped nausea","metformin stopped due to nausea")>.7
    assert token_f1("unrelated","metformin")==0

def test_evaluation_pairing():
    case=EvaluationCase(case_id="1",question="q",documents=["answer text"],reference_answer="answer",reference_evidence=["answer"])
    preds=[PredictionRecord(run_id="a",case_id="1",system="a",provider="x",model="x",answer="answer",evidence_quotes=["answer text"],confidence=.9,latency_seconds=1),PredictionRecord(run_id="b",case_id="1",system="b",provider="x",model="x",answer="wrong",evidence_quotes=["invented"],confidence=.9,latency_seconds=2)]
    detailed,summary=evaluate_predictions([case],preds);assert summary
    assert paired_bootstrap_difference(detailed,"a","b","token_f1",samples=100)["mean_difference"]>0

def test_classification_metrics_and_config():
    cases=[EvaluationCase(case_id="1",question="q",documents=["x"],reference_answer="a",
        reference_label="positive",allowed_labels=["positive","negative"]),
        EvaluationCase(case_id="2",question="q",documents=["x"],reference_answer="b",
        reference_label="negative",allowed_labels=["positive","negative"])]
    preds=[PredictionRecord(run_id="1",case_id="1",system="model",provider="x",model="x",
        answer="a",predicted_label="positive",latency_seconds=1),
        PredictionRecord(run_id="2",case_id="2",system="model",provider="x",model="x",
        answer="b",predicted_label="positive",latency_seconds=1)]
    report=classification_metrics(cases,preds,positive_label="positive")[0]
    assert report["accuracy"]==.5
    assert report["sensitivity"]==1
    config=ExperimentConfig.from_yaml("configs/experiment.example.yaml")
    assert config.multi_agent_models.safety
