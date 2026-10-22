import json,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]; out=root/'results'/'final_figures_v2'; D=[50,100,150,200]
curves={'Low computation load':{'Minimum-Finish':[.611,.365,.279,.230],'SPARC-V2X':[.556,.351,.301,.257]},'Nominal computation load':{'Minimum-Finish':[.492,.291,.207,.153],'SPARC-V2X':[.477,.310,.242,.176]},'High computation load':{'Minimum-Finish':[.442,.238,.146,.110],'SPARC-V2X':[.372,.256,.191,.139]}}
diag=json.loads((root/'results'/'final_mechanism_diagnostics.json').read_text())['by_density']; dmap={x['density']:x['deviation_rate'] for x in diag}; dev=[dmap[d] for d in D]
FONT=50; LW=6; MS=16; colors={'SPARC-V2X':'#1f77b4','Minimum-Finish':'#d62728'}; markers={'SPARC-V2X':'o','Minimum-Finish':'s'}
def finish(fig,stem):
 fig.tight_layout(); fig.savefig(out/(stem+'.png'),dpi=300,bbox_inches='tight'); fig.savefig(out/(stem+'.pdf'),bbox_inches='tight'); plt.close(fig)
def line_panel(title,vals,stem):
 fig,ax=plt.subplots(figsize=(20,18))
 for m,y in vals.items(): ax.plot(D,y,color=colors[m],marker=markers[m],markersize=MS,linewidth=LW,label=m)
 ax.set_xlabel('Vehicle density',fontsize=FONT); ax.set_ylabel('R_value',fontsize=FONT); ax.set_title(title,fontsize=FONT); ax.set_xticks(D); ax.tick_params(axis='both',labelsize=FONT-6); ax.grid(True,alpha=.28); ax.legend(fontsize=FONT-8,frameon=False); finish(fig,stem)
for title,vals in curves.items(): line_panel(title,vals,'fig7_'+title.split()[0].lower()+'_load')
fig,ax=plt.subplots(figsize=(20,18)); ax.plot(D,dev,color=colors['SPARC-V2X'],marker=markers['SPARC-V2X'],markersize=MS,linewidth=LW,label='SPARC-V2X'); ax.set_xlabel('Vehicle density',fontsize=FONT); ax.set_ylabel('Anchor-deviation rate',fontsize=FONT); ax.set_title('Anchor-deviation rate',fontsize=FONT); ax.set_xticks(D); ax.tick_params(axis='both',labelsize=FONT-6); ax.grid(True,alpha=.28); ax.legend(fontsize=FONT-8,frameon=False); finish(fig,'fig7_anchor_deviation')
fig,axes=plt.subplots(2,2,figsize=(20,18)); panels=[('Low computation load',curves['Low computation load'],'R_value'),('Nominal computation load',curves['Nominal computation load'],'R_value'),('High computation load',curves['High computation load'],'R_value'),('Anchor-deviation rate',{'SPARC-V2X':dev},'Anchor-deviation rate')]
for ax,(title,vals,ylabel) in zip(axes.flat,panels):
 for m,y in vals.items(): ax.plot(D,y,color=colors[m],marker=markers[m],markersize=MS,linewidth=LW,label=m)
 ax.set_xlabel('Vehicle density',fontsize=30); ax.set_ylabel(ylabel,fontsize=30); ax.set_title(title,fontsize=32); ax.set_xticks(D); ax.tick_params(axis='both',labelsize=25); ax.grid(True,alpha=.28); ax.legend(fontsize=22,frameon=False)
fig.suptitle('SPARC-V2X robustness across computation load',fontsize=38); fig.tight_layout(rect=[0,0,1,.95]); fig.savefig(out/'fig7_robustness.png',dpi=300,bbox_inches='tight'); fig.savefig(out/'fig7_robustness.pdf',bbox_inches='tight'); plt.close(fig)
