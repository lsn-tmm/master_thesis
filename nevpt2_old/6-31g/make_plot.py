import numpy as np
import scipy.optimize as opt
import sys
sys.path.append('../../nevpt2/src/')
from plot import make_empty_elist, morse, find_minimum, fill_panel, append_energies_f, append_energies, c_list, kcal

R_list    = [#'0.70','0.72','0.74','0.76','0.78',
             '0.80','0.82','0.84','0.86','0.88',
             '0.90','0.92','0.94','0.96','0.98','1.00','1.02','1.04','1.06','1.08','1.10','1.12','1.14','1.16','1.18','1.20']
E_anion   = make_empty_elist()
E_radical = make_empty_elist()

for R in R_list:
    E_anion   = append_energies('anion',R,E_anion)
    E_radical = append_energies('radical',R,E_radical) 

R_list = [float(R) for R in R_list]

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['cmu serif'],'size':12})
rc('text', usetex=True)

L = 5.0
fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(3*L,1*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.20)

m = 0
for x,k1,k2,l,c,mrk,ms,ls in zip([R_list,R_list,R_list,R_list,R_list,R_list,R_list,R_list,R_list],
    ['full','iao','iao','iao','iao','iao','iao','iao','full'],
    ['scf','Ry','qUCCSD','fci','nevpt2(Ry,qse)','nevpt2(qUCCSD,qse)','nevpt2(fci,qse)','nevpt2(fci,fci)','fci'],
    ['SCF','$R_y$/IAO','q-UCCSD/IAO','FCI/IAO','NEVPT2($R_y$,QSE)','NEVPT2(q-UCCSD,QSE)','NEVPT2(FCI,QSE)','NEVPT2(FCI,FCI)','FCI'],
    [c_list['cobalt'],c_list['yellow_green'],c_list['mid_green'],c_list['dark_green'],c_list['gold'],c_list['earwax'],c_list['orange'],c_list['red'],c_list['palatinate']],
    ['*','P','X','o','P','X','o','s','D'],
    [16,24,16,8,24,16,8,10,10],
    [':','-.','-.','-.','--','--','--','--','-']):
    y = E_anion[k1][k2]
    z = E_radical[k1][k2]
    ax1.plot(x,y,c=c,ls=ls)
    ax2.plot(x,z,c=c,ls=ls)
    x0,y0,dx0,dy0 = find_minimum(x,y); ax1.scatter(x0,y0,marker=mrk,s=3*ms,c=c,edgecolor='black',linewidth=0.5)
    x1,y1,dx1,dy1 = find_minimum(x,z); ax2.scatter(x1,y1,marker=mrk,s=3*ms,c=c,edgecolor='black',linewidth=0.5)
    ax1.plot(x,[np.nan]*len(y),label=l,c=c,ls=ls,marker=mrk,ms=7,mec='black',mew=0.5)
    print("%s & %.5f(%d) & %.5f(%d) & %.3f(%d) \\\\" % (l,x0,int(dx0*1e5),x1,int(dx1*1e5),kcal*(y1-y0),int(1e3*kcal*np.sqrt(dy0**2+dy1**2))))
    ax3.scatter(m,kcal*(y1-y0),c=c,marker=mrk,s=4*ms,edgecolor='black',linewidth=0.5)
    m += 1

fill_panel(ax1,'$R_{\mathrm{OH}} \, [\mathrm{\AA}]$',[0.8,1.2],[0.8,0.9,1.0,1.1,1.2],['0.8','0.9','1.0','1.1','1.2'],
               '$E_{\mathrm{anion}} \, [E_h]$',[-75.43,-75.27],[-75.43,-75.39,-75.35,-75.31,-75.27],
                                                  ['-75.43','-75.39','-75.35','-75.31','-75.27'])
fill_panel(ax2,'$R_{\mathrm{OH}} \, [\mathrm{\AA}]$',[0.8,1.2],[0.8,0.9,1.0,1.1,1.2],['0.8','0.9','1.0','1.1','1.2'],
               '$E_{\mathrm{radical}} \, [E_h]$',[-75.43,-75.27],[-75.43,-75.39,-75.35,-75.31,-75.27],
                                                  ['-75.43','-75.39','-75.35','-75.31','-75.27'])
fill_panel(ax3,'',[0,8],[0,1,2,3,4,5,6,7,8],['SCF','$R_y$/IAO','q-UCCSD/IAO','FCI/IAO','NEVPT2($R_y$,QSE)','NEVPT2(q-UCCSD,QSE)','NEVPT2(FCI,QSE)','NEVPT2(FCI,FCI)','FCI'],
                  '$E_{\mathrm{radical}}-E_{\mathrm{anion}}$ [kcal/mol]',[-35,-5],[-5,-10,-15,-20,-25,-30,-35][::-1],[-5,-10,-15,-20,-25,-30,-35][::-1])
ax3.set_xticklabels(ax3.get_xticklabels(),rotation = 30,fontsize=9,ha='right')

x0L,y0L,dxL,dyL = 0.00,1.05,3.44,0.20
handles,labels  = ax1.get_legend_handles_labels()
lgd = ax1.legend(handles,labels,ncol=5,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=3)

fig.savefig('fig.pdf',format='pdf',bbox_inches='tight')

