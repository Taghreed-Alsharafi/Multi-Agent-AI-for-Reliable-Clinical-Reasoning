import math
import numpy as np
from scipy.stats import binomtest
def bootstrap_ci(values,seed=42,samples=2000):
    x=np.asarray(list(values),float); x=x[np.isfinite(x)]
    if not len(x): return (math.nan,math.nan)
    rng=np.random.default_rng(seed); draws=[np.mean(rng.choice(x,len(x),replace=True)) for _ in range(samples)]; return tuple(np.quantile(draws,[.025,.975]))
def paired_difference(a,b,seed=42,samples=2000):
    d=np.asarray(a,float)-np.asarray(b,float); d=d[np.isfinite(d)]; lo,hi=bootstrap_ci(d,seed,samples); return {"difference":float(np.mean(d)),"ci_low":float(lo),"ci_high":float(hi),"n":int(len(d))}
def exact_mcnemar(a,b):
    a,b=np.asarray(a,bool),np.asarray(b,bool); aw=int((a&~b).sum()); bw=int((~a&b).sum()); n=aw+bw
    return {"a_correct_b_wrong":aw,"a_wrong_b_correct":bw,"p_value":float(binomtest(min(aw,bw),n,.5).pvalue) if n else 1.0}
