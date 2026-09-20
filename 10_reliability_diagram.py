"""
Five-bin reliability diagrams for clean and attacked predictions

Paper element : Figure S4
Source        : reliability_5bin.py

Part of SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models.
"""

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

te=pd.read_csv('pllr_pgd_calibration.csv'); tr=pd.read_csv('pllr_pgd_train.csv')
y=te['label'].values.astype(float)
sig=lambda x:1/(1+np.exp(-x)); native=lambda a:2*sig(a)-1
pa=LogisticRegression().fit(tr['pllr_abs_clean'].values.reshape(-1,1),tr['label'].values)
ps=LogisticRegression().fit(tr['pllr_signed_clean'].values.reshape(-1,1),tr['label'].values)
P=lambda m,x:m.predict_proba(x.reshape(-1,1))[:,1]
def ece(p,y,nb=10):
    p=np.clip(p,0,1);e=np.linspace(0,1,nb+1);s=0.;N=len(y)
    for i in range(nb):
        m=(p>e[i])&(p<=e[i+1])
        if m.sum(): s+=abs(y[m].mean()-p[m].mean())*m.sum()/N
    return s
def rel(p,y,nb=5):
    p=np.clip(p,0,1);e=np.linspace(0,1,nb+1);xs=[];ys=[];se=[]
    for i in range(nb):
        m=((p>=e[i])if i==0 else(p>e[i]))&(p<=e[i+1])
        if m.sum():
            o=y[m].mean();xs.append(p[m].mean());ys.append(o);se.append(np.sqrt(max(o*(1-o),1e-6)/m.sum()))
    return map(np.array,(xs,ys,se))

conds=[('clean','pllr_abs_clean','pllr_signed_clean'),('attacked','pllr_abs_att','pllr_signed_att')]
# distinct, colorblind-friendly palette (no near-black); distinct markers & linestyles
maps=[(r'native $2\sigma(|\mathrm{PLLR}|)-1$', '#C44E52','abs','o','-'),
      (r'learned Platt $(|\mathrm{PLLR}|)$',   '#4C72B0','abs','s','-'),
      (r'learned Platt (signed)',              '#55A868','sgn','^','--')]
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,5))
for ax,(cond,ca,cs) in zip(axes,conds):
    ax.plot([0,1],[0,1],':',color='0.55',lw=1.4,label='perfect calibration',zorder=1)
    for z,(name,c,kind,mk,ls) in enumerate(maps):
        if kind=='abs': pv=(native(te[ca].values) if 'native' in name else P(pa,te[ca].values))
        else: pv=P(ps,te[cs].values)
        xs,ys,se=rel(pv,y,5); e10=ece(pv,y,10)
        ax.errorbar(xs,ys,yerr=se,marker=mk,ms=7,lw=2.2,ls=ls,color=c,capsize=3,
                    markeredgecolor='white',markeredgewidth=0.8,alpha=0.95,zorder=3+z,
                    label=f'{name}  (ECE={e10:.3f})')
    ax.set_xlim(-0.02,1.02);ax.set_ylim(-0.02,1.02)
    ax.set_xlabel('Predicted probability');ax.set_ylabel('Observed frequency')
    ax.set_title(f'Reliability diagram ({cond})')
    ax.legend(fontsize=8.5,loc='lower right',framealpha=0.92)
plt.tight_layout()
out='/sessions/zealous-eager-darwin/mnt/SafeGenes__Evaluating_the_Adversarial_Robustness_of_Genomic_Foundation_Models_Arxiv__0623/images/reliability_clean_attacked_5bin.pdf'
plt.savefig(out,bbox_inches='tight'); print("saved:",out)
