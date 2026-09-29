"""Per-repo and repo-pair statistics used in the write-up (copies, shared names, text overlap, clusters).

Run: .venv/bin/python scripts/repo_pair_stats.py
"""
import numpy as np, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from phailogeny.estate import *
R=['wshobson','voltagent','lst97','voltagent-codex','buildwithclaude','0xfurai','contains-studio','pentest','vijaythecoder','taches']
raw=load_subagent_dirs([f'data/raw/{d}' for d in R])
recs=dedupe_copies(raw)
lc=Counter(r['source'] for r in raw); uc=Counter(r['source'] for r in recs)
cp=Counter(); 
for r in recs: cp[r['source']]+=len(r.get('copies',[]))
print("repo loaded unique copies"); [print(d,lc[d],uc[d],cp[d]) for d in R]
# cross-repo exact copies: raw records identical text across different sources
key=lambda r:(" ".join(r['purpose'].split()),)
res=analyse_estate(recs,weights={"purpose":1,"prompt":1,"capability":0,"interface":1},style_centring=False)
ids=res['ids']; src=np.array([r['source'] for r in recs]); S=1-res['functional']; L=res['lineage']; n=len(ids)
stem=np.array([Path(str(r['source_path'])).stem.lower() for r in recs])
# strip plugin prefixes for wshobson names: use stem
D=res['functional'].copy(); np.fill_diagonal(D,np.inf); nn=D.argmin(1)
print("\nNN same repo overall", round((src[nn]==src).mean(),2), "chance", round(np.mean([((src==s).sum()-1)/(n-1) for s in src]),2))
for d in R:
    m=src==d; print(f"  {d}: NN same {np.mean(src[nn][m]==d):.2f} chance {(m.sum()-1)/(n-1):.2f}")
print("\nrepo-pair: shared names, twin func mean, twin lineage median/max, twin in top5")
rows=[]
for ai,a in enumerate(R):
  for b in R[ai+1:]:
    ia=np.where(src==a)[0]; ib=np.where(src==b)[0]
    names_b={stem[j]:j for j in ib}
    tw=[(i,names_b[stem[i]]) for i in ia if stem[i] in names_b]
    if not tw: continue
    f=[S[i,j] for i,j in tw]; l=[L[i,j] for i,j in tw]
    top5=np.mean([ (S[i]>S[i,j]).sum()<=5 for i,j in tw])
    allL=L[np.ix_(ia,ib)]
    rows.append((a,b,len(tw),np.mean(f),st.median(l),max(l),top5,allL.max()))
for r in sorted(rows,key=lambda r:-r[2]): print(f"{r[0]:>16} {r[1]:<16} n={r[2]:3d} func={r[3]:.2f} linmed={r[4]:.3f} linmax={r[5]:.2f} top5={r[6]:.2f} pairmaxlin={r[7]:.2f}")
iu=np.triu_indices(n,1); print("\nall pairs func mean",round(S[iu].mean(),3))
tw_all=[(i,j) for i,j in zip(*iu) if stem[i]==stem[j] and src[i]!=src[j]]
print("all cross-repo twins",len(tw_all),"func mean",round(np.mean([S[i,j] for i,j in tw_all]),3),"lineage median",round(st.median([L[i,j] for i,j in tw_all]),3), ">0.5:",sum(L[i,j]>0.5 for i,j in tw_all))
cross=src[iu[0]]!=src[iu[1]]
print("cross-repo pairs with lineage>0.5:", int(((L[iu]>0.5)&cross).sum()), ">0.3:", int(((L[iu]>0.3)&cross).sum()))
o=np.argsort(-(L[iu]*cross))[:8]
for k in o: i,j=iu[0][k],iu[1][k]; print(f"  lin {L[i,j]:.2f} func {S[i,j]:.2f} {ids[i]} <> {ids[j]}")
lab=np.array(res['cluster_labels']); cl=[c for c in set(lab) if c>=0]
print("clusters",len(cl),"noise",(lab==-1).sum(),"mixed",sum(len(set(src[lab==c]))>1 for c in cl))
cnt=list(Counter(lab[lab>=0]).values())+[1]*int((lab==-1).sum()); p=np.array(cnt)/sum(cnt)
print("hill q0",len(p),"q1",round(np.exp(-(p*np.log(p)).sum()),1),"q2",round(1/(p**2).sum(),1))
# pentest outgroup: mean sim to others
m=src=='pentest'; print("pentest mean sim to others", round(S[np.ix_(m,~m)].mean(),3), "others among themselves", round(S[np.ix_(~m,~m)].mean(),3))
