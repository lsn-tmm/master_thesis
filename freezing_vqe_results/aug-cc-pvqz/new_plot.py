import numpy as np
import scipy.optimize as opt
import sys
sys.path.append('../../nevpt2/src/')
from plot import make_empty_elist, morse, find_minimum, fill_panel, append_energies_f, append_energies, c_list, kcal

path = '../../nevpt2/aug-cc-pvqz'

R_list    = [#'0.70','0.72','0.74','0.76','0.78',
             '0.80','0.82','0.84','0.86','0.88',
             '0.90','0.92','0.94','0.96','0.98','1.00','1.02','1.04','1.06','1.08','1.10','1.12','1.14','1.16','1.18','1.20']

E_anion_n   = make_empty_elist()
E_anion_f   = make_empty_elist()
E_radical_n = make_empty_elist()
E_radical_f = make_empty_elist()

for R in R_list:
    E_anion_n   = append_energies('anion',R,E_anion_n, path=path)
    E_radical_n = append_energies('radical',R,E_radical_n, path=path) 
    E_anion_f   = append_energies_f('anion',R,E_anion_f, path=path)
    E_radical_f = append_energies_f('radical',R,E_radical_f, path=path)
    
R_list = [float(R) for R in R_list]

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['cmu serif'],'size':12})
rc('text', usetex=True)

L = 5.0
fig,ax = plt.subplots(1,3,figsize=(3*L,1.15*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.40)
ax1=ax[0]
ax2=ax[1]
ax3=ax[2]

m=3
fline, nline = np.empty(0), np.empty(0)
for x,k1,k2,l,c,mrk,ms,ls in zip([R_list,R_list],
    ['iao','iao'],
    ['nevpt2(Ry,qse)','nevpt2(qUCCSD,qse)'],
    [ ['NEVPT2($R_y$,QSE) (1s)', 'NEVPT2($R_y$,QSE) (1s2s)'], ['NEVPT2(q-UCCSD,QSE) (1s)','NEVPT2(q-UCCSD,QSE) (1s2s)'] ],
    [ [c_list['light_blue'],c_list['dark_blue']], [c_list['orange'],c_list['red']] ],
    ['o','s'],
    [24,16],
    ['--','--']):
    Ean = E_anion_n[k1][k2]
    Ern = E_radical_n[k1][k2]
    Eaf = E_anion_f[k1][k2]
    Erf = E_radical_f[k1][k2]
    ax1.plot(x,Eaf,c=c[1],ls=ls)
    ax1.plot(x,Ean,c=c[0],ls=ls)
    ax2.plot(x,Erf,c=c[1],ls=ls)
    ax2.plot(x,Ern,c=c[0],ls=ls)  
        
    x0,y0,dx0,dy0 = find_minimum(x,Eaf); ax1.scatter(x0,y0,marker=mrk,s=3*ms,c=c[1],edgecolor='black',linewidth=0.5)
    x1,y1,dx1,dy1 = find_minimum(x,Ean); ax1.scatter(x1,y1,marker=mrk,s=3*ms,c=c[0],edgecolor='black',linewidth=0.5) 
    x2,y2,dx2,dy2 = find_minimum(x,Erf); ax2.scatter(x2,y2,marker=mrk,s=3*ms,c=c[1],edgecolor='black',linewidth=0.5)
    x3,y3,dx3,dy3 = find_minimum(x,Ern); ax2.scatter(x3,y3,marker=mrk,s=3*ms,c=c[0],edgecolor='black',linewidth=0.5)
    
    print("%s & %.5f(%d) & %.5f(%d) & %.3f(%d) \\\\" % (l[0],x0,int(dx0*1e5),x2,int(dx2*1e5),kcal*(y2-y0),int(1e3*kcal*np.sqrt(dy0**2+dy2**2))))
    print("%s & %.5f(%d) & %.5f(%d) & %.3f(%d) \\\\" % (l[1],x1,int(dx1*1e5),x3,int(dx3*1e5),kcal*(y3-y1),int(1e3*kcal*np.sqrt(dy3**2+dy1**2))))
    ax3.scatter(m,kcal*(y2-y0),c=c[1],marker=mrk,s=4*ms,edgecolor='black',linewidth=0.5)
    m += 8
    ax3.scatter(m,kcal*(y3-y1),c=c[0],marker=mrk,s=4*ms,edgecolor='black',linewidth=0.5)
    
    fline = np.append(fline, kcal*(y2-y0))
    nline = np.append(nline, kcal*(y3-y1))
    
    ax1.plot(x,[np.nan]*len(Eaf),label=l[0],c=c[1],ls=ls,marker=mrk,ms=7,mec='black',mew=0.5)
    ax1.plot(x,[np.nan]*len(Ean),label=l[1],c=c[0],ls=ls,marker=mrk,ms=7,mec='black',mew=0.5)  
    m -= 4
    
#ax5.plot( [0,10], np.ones(2)*np.mean(fline), c='black', ls='--', linewidth=0.5)
#ax6.plot( [0,10], np.ones(2)*np.mean(nline), c='black', ls='--', linewidth=0.5) 
#ax5.annotate(str(round(np.mean(fline),2)) + ' kcal/mol', xy=(7.5,np.mean(fline)+0.3), fontsize=10, color='black')
#ax6.annotate(str(round(np.mean(nline),2)) + ' kcal/mol', xy=(7.5,np.mean(nline)+0.3), fontsize=10, color='black')


    
fill_panel(ax1,'$R_{\mathrm{OH}} \, [\mathrm{\AA}]$',[0.8,1.2],[0.8,0.9,1.0,1.1,1.2],['0.8','0.9','1.0','1.1','1.2'],
               '$E_{\mathrm{anion}} \, [E_h]$', [-75.65,-75.45],[-75.65,-75.60,-75.55,-75.50,-75.45], ['-75.65','-75.60','-75.55','-75.50','-75.45']) 

fill_panel(ax2,'$R_{\mathrm{OH}} \, [\mathrm{\AA}]$',[0.8,1.2],[0.8,0.9,1.0,1.1,1.2],['0.8','0.9','1.0','1.1','1.2'],
               '$E_{\mathrm{radical}} \, [E_h]$',[-75.65,-75.45],[-75.65,-75.60,-75.55,-75.50,-75.45], ['-75.65','-75.60','-75.55','-75.50','-75.45'])

fill_panel(ax3,'',[0,18],[3,7,11,15],['NEVPT2($R_y$,QSE) (1s)', 'NEVPT2(q-UCCSD,QSE) (1s)', 'NEVPT2($R_y$,QSE) (1s2s)', 'NEVPT2(q-UCCSD,QSE) (1s2s)'],
                  '$E_{\mathrm{radical}}-E_{\mathrm{anion}}$ [kcal/mol]',[-10,40],[-10,0,10,20,30,40],[-10,0,10,20,30,40])

ax3.set_xticklabels(ax3.get_xticklabels(),rotation = 30,fontsize=9,ha='right')

x0L,y0L,dxL,dyL = 0.00,1.05,3.44,0.20
handles,labels  = ax1.get_legend_handles_labels()
lgd = ax1.legend(handles,labels,ncol=5,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=3)

fig.savefig('new_freezing_qz.pdf',format='pdf',bbox_inches='tight')


