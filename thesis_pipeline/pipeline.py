from __future__ import annotations
from pathlib import Path
import asyncio, json
from concurrent.futures import ThreadPoolExecutor
import os
import pandas as pd
from .config import ThesisConfig
from .data import initialize_dataset, DatasetBundle
from .methods import discover_agreement_methods, aggregate, aggregate_agent_outputs, mock_agent_outputs, select_method
from .reproducibility import assignment, run_metadata, save_json
from .evaluation import score, metrics, final_comparison
from .artifacts import build_reports, build_figures
from .guards import assert_identical_final_inputs

def _run_async(coro):
    """Run a coroutine from both scripts and Jupyter's active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Jupyter/IPykernel owns the current loop. Run the coroutine in a separate
    # thread with its own loop while keeping the public notebook API synchronous.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()

class ThesisPipeline:
    def __init__(self, config: ThesisConfig, root: str | Path = "."):
        self.config=config; self.root=Path(root)
        try:
            from dotenv import load_dotenv
            load_dotenv(self.root/".env")
        except ImportError:
            pass
        self.bundle=initialize_dataset(config.dataset, root, config.global_seed); self.output=config.resolved_output(root); self.output.mkdir(parents=True,exist_ok=True); self.selected=None
        save_json(self.output/"dataset_validation.json", self.bundle.validation)
        self.bundle.frame.loc[~self.bundle.frame.eligible_for_evaluation, ["case_id","instruction","gold","eligible_for_evaluation"]].to_csv(self.output/"excluded_cases.csv", index=False)
    def _cases(self, split):
        frame=self.bundle.cases(split); limit={"test":self.config.test_size,"pilot":self.config.pilot_size}.get(self.config.run_mode)
        return frame.head(limit) if limit else frame
    def _mock_predictions(self, cases, system, agents=False):
        rows=[]
        for _,r in cases.iterrows():
            if agents:
                assigns=[assignment(r.case_id,f"agent_{i+1}",self.config.global_seed,self.config.gpt_model,self.config.claude_model) for i in range(self.config.n_agents)]
                save_json(self.output/"model_assignments.json", assigns if not (self.output/"model_assignments.json").exists() else json.loads((self.output/"model_assignments.json").read_text())+assigns)
                labels=sorted(set(__import__('re').findall(r"(?m)^\s*([A-H])[.)/:]",str(r.instruction),__import__('re').I))) or list("ABCD")
                outputs=mock_agent_outputs(str(r.case_id),labels,self.config.n_agents,self.config.global_seed)
                pred=aggregate_agent_outputs(outputs,"care_consensus")
                rows.append({"case_id":r.case_id,"prediction":pred,"agent_outputs":json.dumps(outputs),"critical_safety_error":False,"status":"success","system":system})
                continue
            # Offline validation must never read the gold answer to generate a prediction.
            labels=sorted(set(__import__('re').findall(r"(?m)^\s*([A-H])[.)/:]",str(r.instruction),__import__('re').I))) or list("ABCD")
            digest=__import__('hashlib').sha256(f"{r.case_id}:{system}:{self.config.global_seed}".encode()).hexdigest(); pred=labels[int(digest[:8],16)%len(labels)]
            rows.append({"case_id":r.case_id,"prediction":pred,"critical_safety_error":False,"status":"success","system":system})
        return pd.DataFrame(rows)
    def _single_predictions(self, cases, name, provider, model, split):
        if self.config.mock or self.config.report_only or not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
            return self._mock_predictions(cases,name)
        from .live import run_single_live
        return _run_async(run_single_live(cases,provider,model,self.output/f"{name}_{split}.jsonl",name))
    def _multi_predictions(self, cases, split):
        if self.config.mock or self.config.report_only or not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
            return self._mock_predictions(cases,"selected_multi_agent",True)
        from .live import run_mixed_live
        return _run_async(run_mixed_live(cases,self.config,self.output/f"selected_multi_agent_{split}.jsonl",self.output/"model_assignments.jsonl"))
    def _save(self, name, frame, split):
        path=self.output/f"{name}_{split}.csv"; frame.to_csv(path,index=False); save_json(self.output/f"{name}_{split}_metadata.json",run_metadata(self.config,self.bundle,split,frame.case_id.astype(str).tolist(),self.selected)); return path
    def run_single_baselines(self):
        split="test"; cases=self._cases(split); results=[]
        for name in ("single_gpt","single_claude"):
            pred=self._single_predictions(cases,name,"openai" if name.endswith("gpt") else "anthropic",self.config.gpt_model if name.endswith("gpt") else self.config.claude_model,split); self._save(name,pred,split); row=metrics(score(cases,pred),name); row["split"]=split; results.append(row)
        out=pd.DataFrame(results); out.to_csv(self.output/"single_baseline_metrics.csv",index=False); return out
    def run_agreement_experiments(self):
        cases=self._cases("development")
        if self.config.mock or self.config.report_only or (not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY")):
            base=self._mock_predictions(cases,"multi_agent",True)
        else:
            from .live import run_mixed_shared
            base=_run_async(run_mixed_shared(cases,self.config,self.output/"shared_agent_outputs.jsonl",self.output/"model_assignments.jsonl"))
        if base.empty or "agent_outputs" not in base.columns or not base.agent_outputs.map(lambda x: bool(json.loads(x) if isinstance(x,str) else x)).any():
            raise RuntimeError("Agreement experiment produced no agent outputs. Check API connectivity, API keys, and the anthropic package installation.")
        shared=self.output/"shared_agent_outputs.csv"; base.to_csv(shared,index=False); results=[]
        for method in discover_agreement_methods():
            pred=base.copy(); pred["prediction"]=[aggregate_agent_outputs(json.loads(x) if isinstance(x,str) else x,method) for x in pred.agent_outputs]; m=metrics(score(cases,pred),method); m.update({"method":method,"split":"development","extra_llm_calls":0}); results.append(m)
        results=sorted(results,key=lambda x:(-x["accuracy"],x["critical_safety_error_rate"],-x["macro_f1"],x["extra_llm_calls"],x["method"]))
        for rank,row in enumerate(results,1): row["rank"]=rank; row["selected"] = rank==1
        save_json(self.output/"agreement_method_results.json",results); return pd.DataFrame(results)
    def select_best_agreement_method(self):
        results=json.loads((self.output/"agreement_method_results.json").read_text()); selection=select_method(results); self.selected=selection["selected_method"]; save_json(self.output.parent/"selection"/"selected_agreement_method.json",selection); return selection
    def run_final_comparison(self):
        selected=self.selected or json.loads((self.output.parent/"selection"/"selected_agreement_method.json").read_text())["selected_method"]; self.selected=selected; cases=self._cases("test"); frames={}
        for name in ("single_gpt","single_claude","selected_multi_agent"):
            p=self._multi_predictions(cases,"test") if name=="selected_multi_agent" else self._single_predictions(cases,name,"openai" if name.endswith("gpt") else "anthropic",self.config.gpt_model if name.endswith("gpt") else self.config.claude_model,"test"); self._save(name,p,"test"); frames[name]=score(cases,p)
        assert_identical_final_inputs(frames)
        result=final_comparison(frames)
        for row in result["systems"].values(): row["split"]="test"
        save_json(self.output/"final_comparison.json",result); return result
    def build_thesis_outputs(self):
        final=json.loads((self.output/"final_comparison.json").read_text()); agreements=json.loads((self.output/"agreement_method_results.json").read_text()); paths=build_reports(self.output,{"dataset":self.bundle.validation,"final":final,"agreements":agreements}); figures=build_figures(self.output,final,agreements); return {"reports":[str(x) for x in paths],"figures":[str(x) for x in figures]}

_CURRENT=None
def initialize_thesis(dataset: dict, RUN_MODE="test", GLOBAL_SEED=42, root: str | Path = ".", **kwargs):
    global _CURRENT; _CURRENT=ThesisPipeline(ThesisConfig(dataset=dataset,run_mode=RUN_MODE,global_seed=GLOBAL_SEED,**kwargs), root=root); return _CURRENT
