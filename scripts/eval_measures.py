"""Score every similarity measure on finding same-named twins across repos (uncentred embeddings).

Writes out/traits/eval.json and out/traits/traits.csv. Run: .venv/bin/python scripts/eval_measures.py
"""
import json, numpy as np, time
from collections import Counter
from pathlib import Path
from phailogeny.estate import load_subagent_dirs, dedupe_copies, analyse_estate
from phailogeny.characters.traits import load_taxonomy, trait_matrix, trait_distances, trait_names
from phailogeny.analysis.measure_eval import evaluate, twin_pairs
R=['wshobson','voltagent','lst97','voltagent-codex','buildwithclaude','0xfurai','contains-studio','pentest','vijaythecoder','taches']
recs=dedupe_copies(load_subagent_dirs([f'data/raw/{d}' for d in R]))
res=analyse_estate(recs,weights={"purpose":1,"prompt":1,"capability":0,"interface":1},style_centring=False)
tax=load_taxonomy('data/traits.yaml'); names=trait_names(tax)
t=time.time(); M=trait_matrix(recs,tax); print("extract s",round(time.time()-t,1))
Tall=trait_distances(M); Tact=trait_distances(M,categories=['activity'],names=names); Ttech=trait_distances(M,categories=['language','platform'],names=names)
measures={
 "text overlap (MinHash)":1-res['lineage'],
 "embedding: description":res['blocks']['purpose'],
 "embedding: prompt":res['blocks']['prompt'],
 "embedding: combined (current)":res['functional'],
 "traits: all":Tall,
 "traits: activity only":Tact,
 "traits: language+platform":Ttech,
 "traits + embeddings (mean)":(Tall+res['functional'])/2,
}
ev=evaluate(recs,measures)
print(json.dumps({k:v for k,v in ev.items() if k!='measures'}))
cols=list(next(iter(ev['measures'].values())).keys())
for c in cols:
    print(f"\n{c}"); [print(f"   {m:<32} {v[c]}") for m,v in ev['measures'].items()]
# extraction quality
nz=(M>0).sum(1); print("\ntraits per agent median",np.median(nz),"zero-trait agents",int((nz==0).sum()))
freq=Counter({names[k]:int((M[:,k]>0).sum()) for k in range(len(names))})
print("most common", freq.most_common(8)); print("never matched",[n for n,c in freq.items() if c==0])
src=np.array([r['source'] for r in recs]); ids=[r['agent_id'] for r in recs]
def show(a):
    i=ids.index(a); top=np.argsort(-M[i])[:8]; print(a, [names[k].split(':')[1] for k in top if M[i,k]>0])
for a in ['voltagent/backend-developer','voltagent-codex/backend-developer','wshobson/debugging-toolkit-debugger','lst97/debugger','voltagent/business-analyst','voltagent-codex/business-analyst']:
    try: show(a)
    except ValueError: print("missing",a)
out=Path('out/traits'); out.mkdir(parents=True,exist_ok=True)
(out/'eval.json').write_text(json.dumps(ev,indent=2))
import csv
with open(out/'traits.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['agent_id']+names)
    for a,row in zip(ids,M): w.writerow([a]+[round(x,3) for x in row])
