from __future__ import annotations
import hashlib,json,platform,sys
from datetime import datetime,timezone
def assignment(case_id,agent_role,global_seed,gpt_model,claude_model):
    key=f"{case_id}:{agent_role}:{global_seed}"; digest=hashlib.sha256(key.encode()).hexdigest(); provider="openai" if int(digest[:8],16)%2==0 else "anthropic"
    return {"case_id":case_id,"agent_role":agent_role,"assigned_provider":provider,"assigned_model":gpt_model if provider=="openai" else claude_model,"assignment_seed":key,"assignment_hash":digest}
def run_metadata(config,bundle,split,case_ids,agreement_method=None):
    return {"run_id":hashlib.sha256(f"{datetime.now(timezone.utc).isoformat()}:{config.configuration_hash}".encode()).hexdigest()[:16],"timestamp":datetime.now(timezone.utc).isoformat(),"dataset_name":bundle.dataset_name,"dataset_path":bundle.dataset_path,"dataset_fingerprint":bundle.dataset_fingerprint,"label_fingerprint":bundle.label_fingerprint,"split":split,"case_ids":case_ids,"global_seed":config.global_seed,"gpt_model":config.gpt_model,"claude_model":config.claude_model,"prompt_versions":config.prompt_versions,"agreement_method":agreement_method,"configuration_hash":config.configuration_hash,"code_version":{"python":sys.version,"platform":platform.platform()}}
def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,default=str),encoding="utf-8")
