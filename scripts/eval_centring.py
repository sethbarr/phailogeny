"""Compare style-removal methods: none, repo-mean 50%, and twin-based offsets (5-fold CV by agent name).

Writes out/traits/twin_centring_cv.json. Run: .venv/bin/python scripts/eval_centring.py
"""
import json, numpy as np, random
from pathlib import Path
from phailogeny.estate import *
from phailogeny.analysis.measure_eval import relation_of, same_repo_neighbour_rate
R=['wshobson','voltagent','lst97','voltagent-codex','buildwithclaude','0xfurai','contains-studio','pentest','vijaythecoder','taches']
recs=dedupe_copies(load_subagent_dirs([f'data/raw/{d}' for d in R]))
src=[str(r['source']) for r in recs]; S=np.array(src)
P,Q=embedding_vectors([str(r['purpose']) for r in recs], strip_corpus_boilerplate(recs))
tw=twin_pairs(recs)
stems=[Path(str(recs[i]['source_path'])).stem.lower() for i,_ in tw]
ustem=sorted(set(stems)); random.Random(0).shuffle(ustem); fold={s:k%5 for k,s in enumerate(ustem)}
def dist(Pv,Qv): return (vector_distances(Pv)+vector_distances(Qv))/2
def ranks(D,pairs):
    within,glob,rel=[],[],[]
    for i,j in pairs:
        for a,b in ((i,j),(j,i)):
            pool=D[a,S==S[b]]; within.append(1+int((pool<D[a,b]).sum())+((pool==D[a,b]).sum()-1)/2)
            d=D[a].copy(); d[a]=np.inf; glob.append(1+int((d<d[b]).sum())); rel.append(relation_of(recs,i,j))
    return within,glob,rel
def summarise(w,g,rel):
    w,g,rel=np.array(w),np.array(g),np.array(rel)
    out={"top1 within repo":round(float((w==1).mean()),3),"global top5":round(float((g<=5).mean()),3),"global median rank":float(np.median(g))}
    for r in ["format change (same org, Claude -> Codex)","full rewrite (documented ancestry)","edited copy"]:
        out[f"top1 | {r.split(' (')[0]}"]=round(float((w[rel==r]==1).mean()),3)
    return out
def pentest(D):
    k=S=='pentest'; Sim=1-D; return round(float(Sim[np.ix_(k,k)].mean()-Sim[np.ix_(k,~k)].mean()),3)
results={}
D0=dist(P,Q); w,g,rel=ranks(D0,tw)
results["uncentred"]={**summarise(w,g,rel),"same-repo NN":round(same_repo_neighbour_rate(D0,S),3),"pentest cohesion":pentest(D0)}
def repo_center(X,a):
    X=X.copy()
    for s in set(src): k=S==s; X[k]-=a*X[k].mean(0)
    return X/np.linalg.norm(X,axis=1,keepdims=True).clip(1e-12)
Dr=dist(repo_center(P,0.5),repo_center(Q,0.5)); w,g,rel=ranks(Dr,tw)
results["repo-mean 50%"]={**summarise(w,g,rel),"same-repo NN":round(same_repo_neighbour_rate(Dr,S),3),"pentest cohesion":pentest(Dr)}
for ridge in [1,5,20]:
    W,G,REL=[],[],[]
    for f in range(5):
        train=[p for p,s in zip(tw,stems) if fold[s]!=f]; test=[p for p,s in zip(tw,stems) if fold[s]==f]
        Pc=remove_style(P,src,fit_style_offsets(P,src,train,ridge)); Qc=remove_style(Q,src,fit_style_offsets(Q,src,train,ridge))
        w,g,rel=ranks(dist(Pc,Qc),test); W+=w;G+=g;REL+=rel
    Pc=remove_style(P,src,fit_style_offsets(P,src,tw,ridge)); Qc=remove_style(Q,src,fit_style_offsets(Q,src,tw,ridge)); Df=dist(Pc,Qc)
    results[f"twin-based ridge={ridge} (5-fold CV)"]={**summarise(W,G,REL),"same-repo NN":round(same_repo_neighbour_rate(Df,S),3),"pentest cohesion":pentest(Df)}
print(json.dumps(results,indent=1)); Path('out/traits/twin_centring_cv.json').write_text(json.dumps(results,indent=2))
