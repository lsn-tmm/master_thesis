import numpy as np
import scipy.optimize as opt
import sys
sys.path.append('../../nevpt2/src/')
from plot import fill_panel, c_list, kcal, find_minimum

data = {}
err = {}
hardware = ['kolkata','auckland','qasm_kolkata','qasm_auckland']
R_list = ['0.80','0.90','1.00','1.10','1.20']

for hw in hardware:
    for s in ['anion','radical']:
        for p in ['raw','em','em_re']:
            data['%s/%s_%s' % (s,hw,p)] = []
            err['%s/%s_%s' % (s,hw,p)] = []

for s in ['anion','radical']:
    data['%s' % (s)] = []

for hw in hardware:
    for s in ['anion','radical']:
        for R in ['0.80','0.90','1.00','1.10','1.20']:
            for p in ['raw','em','em_re']:
                f = open('%s/%s/R_%s_%s_%s/results_vqe.txt' % (s,R,R,hw,p),'r')
                f = [x.split() for x in f.readlines()]
                f = f[len(f)-1]
                ave = float(f[8])
                std = float(f[10])
                data['%s/%s_%s' % (s,hw,p)].append(ave) 
                err['%s/%s_%s' % (s,hw,p)].append(std)
                
                         
for s in ['anion','radical']:
    for R in ['0.80','0.90','1.00','1.10','1.20']:
        f = open('%s/%s/R_%s/results.txt' % (s,R,R),'r')
        f = [x.split() for x in f.readlines()]
        f = f[len(f)-2]
        ave = float(f[3])
        data['%s' % (s)].append(ave)
        

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['cmu serif'],'size':12})
rc('text', usetex=True)

L = 5.0
fig,ax = plt.subplots(2,3,figsize=(3*L,2.5*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.20)
ax1 = ax[0,0]
ax2 = ax[0,1]
ax3 = ax[0,2]
ax4 = ax[1,0]
ax5 = ax[1,1]
ax6 = ax[1,2]

x=-3
ms=8
R_list = [float(R) for R in R_list]
colors = [ [c_list['red'],c_list['orange'],c_list['light_yellow']],[c_list['dark_blue'],c_list['cobalt'],c_list['cerulean']],[c_list['brown'],c_list['earwax'],c_list['gold']],[c_list['dark_green'],c_list['mid_green'],c_list['yellow_green']]]
markers = [ ['o','s','+'],['o','s','+'],['o','s','+'],['o','s','+']]

state_radical = np.array(data['radical'])
state_anion   = np.array(data['anion'])
state_E       = state_radical - state_anion
ax1.plot(R_list,state_anion,c='black', marker='+', label='statevector', ms=2)
ax2.plot(R_list,state_radical,c='black', marker='+', label='statevector',ms=2)
ax4.plot(R_list,state_anion-state_anion,c='black', marker='+', label='statevector',ms=2)
ax5.plot(R_list,state_radical-state_radical,c='black', marker='+', label='statevector',ms=2)
        
x0,s0,dx0,dy0 = find_minimum(R_list,state_anion); ax1.scatter(x0,s0,marker='X',s=3*ms,c='black',edgecolor='black',linewidth=0.5)
x1,s1,dx1,dy1 = find_minimum(R_list,state_radical); ax2.scatter(x1,s1,marker='X',s=3*ms,c='black',edgecolor='black',linewidth=0.5)
ax3.plot([0,13],kcal*np.array(s1-s0)*np.ones(2),c='black', ls='--')
ax3.annotate(str(round(kcal*np.array(s1-s0),2)) + 'kcal/mol', xy=(0.2,kcal*np.array(s1-s0)+0.2), fontsize=10, color='black')
ax6.plot([0,13],np.zeros(2),c='black', ls='--')

for hw,col,mrk in zip(hardware,colors,markers):
    x+=6
    for p,c,m in zip(['em_re','em','raw'],col,mrk):
        radical = np.array(data['radical/%s_%s' % (hw,p)])
        anion   = np.array(data['anion/%s_%s' % (hw,p)])
        dr      = np.array(err['radical/%s_%s' % (hw,p)])
        da      = np.array(err['anion/%s_%s' % (hw,p)])
        E       = radical - anion
        dE      = np.sqrt(dr**2+da**2)
        l = ''
        for word in hw.split('_'):
            l = l + word + ' '
        if   (p == 'em_re') : l = l +'EM+RE'
        elif (p == 'em') : l = l +'EM'
        elif (p == 'raw') : l = l +'RAW'
        ax1.errorbar(R_list,anion,yerr=da,c=c, marker=m,capsize=3, elinewidth=3,label=l, ms=2)
        ax2.errorbar(R_list,radical,yerr=dr,c=c, marker=m,capsize=3, elinewidth=3,label=l,ms=2)
        ax4.errorbar(R_list,anion-state_anion, yerr=da,c=c, marker=m,capsize=3, elinewidth=3,label=l, ms=2)
        ax5.errorbar(R_list,radical-state_radical, yerr=dr,c=c, marker=m,capsize=3, elinewidth=3,label=l, ms=2)
        
        try:
            x0,y0,dx0,dy0 = find_minimum(R_list,anion); ax1.scatter(x0,y0,marker='X',s=3*ms,c=c,edgecolor='black',linewidth=0.5)
            x1,y1,dx1,dy1 = find_minimum(R_list,radical); ax2.scatter(x1,y1,marker='X',s=3*ms,c=c,edgecolor='black',linewidth=0.5)
            ax3.errorbar(x,kcal*np.array(y1-y0),yerr=kcal*np.array(np.sqrt(dy0**2+dy1**2)),c=c,marker='X',capsize=5, elinewidth=3,label=l, ls='none')
            ax6.errorbar(x,kcal*np.array((y1-y0)-(s1-s0)),yerr=kcal*np.array(np.sqrt(dy0**2+dy1**2)),c=c,marker='X',capsize=5, elinewidth=3,label=l, ls='none')
            print('Dati: ', hw, ' ' , p,' ', kcal*np.array(y1-y0), ' ' ,kcal*np.array(np.sqrt(dy0**2+dy1**2)))
        except:
            pass
        x-=1

fill_panel(ax1,'$R_{\mathrm{OH}} \, [\mathrm{\AA}]$',[0.8,1.2],[0.8,0.9,1.0,1.1,1.2],['0.8','0.9','1.0','1.1','1.2'],
               '$E_{\mathrm{anion}} \, [E_h]$',[-75.44,-75.38],[-75.44,-75.42,-75.40,-75.38],
                                                  ['-75.44','-75.42','-75.41','-75.40'])

fill_panel(ax4,'$R_{\mathrm{OH}} \, [\mathrm{\AA}]$',[0.8,1.2],[0.8,0.9,1.0,1.1,1.2],['0.8','0.9','1.0','1.1','1.2'],
               '$E_{\mathrm{anion}} - E_{\mathrm{statevector}} \, [E_h]$',[0.00,0.025],[0.000,0.005,0.010,0.015,0.020,0.025],
                                             ['0.000','0.005','0.010','0.015','0.020','0.025'])

#[0.00,0.02],[0.000,0.005,0.010,0.015,0.020],
#                                                  ['0.000','0.005','0.010','0.015','0.020'])
                                                  
fill_panel(ax2,'$R_{\mathrm{OH}} \, [\mathrm{\AA}]$',[0.8,1.2],[0.8,0.9,1.0,1.1,1.2],['0.8','0.9','1.0','1.1','1.2'],
               '$E_{\mathrm{radical}} \, [E_h]$',[-75.44,-75.38],[-75.44,-75.42,-75.40,-75.38],
                                                  ['-75.44','-75.42','-75.41','-75.40'])

fill_panel(ax5,'$R_{\mathrm{OH}} \, [\mathrm{\AA}]$',[0.8,1.2],[0.8,0.9,1.0,1.1,1.2],['0.8','0.9','1.0','1.1','1.2'],
               '$E_{\mathrm{radical}} - E_{\mathrm{statevector}} \, [E_h]$',[0.00,0.025],[0.000,0.005,0.010,0.015,0.020,0.025],
                                             ['0.000','0.005','0.010','0.015','0.020','0.025'])

fill_panel(ax3,'',[0,13],[1,2,3,4,5,6,7,8,9,10,11,12],['kolkata RAW','kolkata EM','kolkata EM+RE','auckland RAW','auckland EM','auckland EM+RE','qasm kolkata RAW','qasm kolkata EM','qasm kolkata EM+RE','qasm auckland RAW','qasm auckland EM','qasm auckland EM+RE'],
                  '$E_{\mathrm{radical}}-E_{\mathrm{anion}}$ [kcal/mol]',[-8,8],[-8,-5,-1,2,5,8],[-8,-5,-1,2,5,8])

fill_panel(ax6,'',[0,13],[1,2,3,4,5,6,7,8,9,10,11,12],['kolkata RAW','kolkata EM','kolkata EM+RE','auckland RAW','auckland EM','auckland EM+RE','qasm kolkata RAW','qasm kolkata EM','qasm kolkata EM+RE','qasm auckland RAW','qasm auckland EM','qasm auckland EM+RE'],
                  '$\Delta E-E_{\mathrm{statevector}}$ [kcal/mol]',[-8,8],[-8,-5,-1,2,5,8],[-8,-5,-1,2,5,8])

ax3.set_xticklabels(ax3.get_xticklabels(),rotation = 30,fontsize=9,ha='right')
ax6.set_xticklabels(ax6.get_xticklabels(),rotation = 30,fontsize=9,ha='right')

x0L,y0L,dxL,dyL = 0.00,1.05,3.44,0.20
handles,labels  = ax1.get_legend_handles_labels()
lgd = ax1.legend(handles,labels,ncol=4,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=3)

fig.savefig('vqe_0.5.pdf',format='pdf',bbox_inches='tight')
