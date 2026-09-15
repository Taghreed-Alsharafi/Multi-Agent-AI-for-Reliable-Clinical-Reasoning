from collections import Counter
import hashlib, math
EXISTING_AGREEMENT_METHODS=("jsd","kendall_w","krippendorff_alpha","care_consensus","vote_entropy","ua_kalpha")
def discover_agreement_methods(): return list(EXISTING_AGREEMENT_METHODS)
def aggregate(responses, method):
    values=[str(x).strip().upper() for x in responses if str(x).strip()]
    return sorted(Counter(values),key=lambda x:(-Counter(values)[x],x))[0] if values else ""

def aggregate_agent_outputs(outputs, method):
    """Aggregate one shared panel, without accessing gold labels."""
    usable=[x for x in outputs if x.get("answer")]
    if not usable: return ""
    counts=Counter(str(x["answer"]).upper() for x in usable)
    if method == "vote_entropy":
        probs=[n/len(usable) for n in counts.values()]
        if len(counts)>1 and -sum(p*math.log2(p) for p in probs)/math.log2(len(counts)) > .9: return ""
    if method in {"jsd","ua_kalpha","care_consensus"}:
        weighted={k:sum(float(x.get("confidence",.5)) for x in usable if str(x["answer"]).upper()==k) for k in counts}
        return sorted(weighted,key=lambda k:(-weighted[k],k))[0]
    return sorted(counts,key=lambda k:(-counts[k],k))[0]

def mock_agent_outputs(case_id, labels, n_agents, seed):
    """Deterministic offline panel; deliberately independent of the gold answer."""
    out=[]
    for i in range(n_agents):
        digest=hashlib.sha256(f"{case_id}:agent_{i+1}:{seed}".encode()).hexdigest()
        answer=labels[int(digest[:8],16)%len(labels)] if labels else ""
        out.append({"agent_role":f"agent_{i+1}","answer":answer,"confidence":.55+(int(digest[8:12],16)%35)/100})
    return out
def select_method(results, accuracy_tolerance=.01):
    if not results: raise ValueError("No development agreement results")
    top=max(float(x.get("accuracy",0)) for x in results); eligible=[x for x in results if float(x.get("accuracy",0))>=top-accuracy_tolerance]
    chosen=sorted(eligible,key=lambda x:(float(x.get("critical_safety_error_rate",1)),-float(x.get("macro_f1",0)),float(x.get("extra_llm_calls",0)),str(x["method"])))[0]
    return {"selected_method":chosen["method"],"selection_split":"development","accuracy_tolerance":accuracy_tolerance,"hierarchy":["accuracy","critical_safety_error_rate","macro_f1","extra_llm_calls"],"candidates":results}
