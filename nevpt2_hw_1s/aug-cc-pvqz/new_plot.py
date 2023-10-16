import numpy as np
import scipy.optimize as opt
import sys
sys.path.append('../../nevpt2/src/')
from plot import fill_panel, c_list, kcal, morse

def find_minimum(x,y):
    x,y = np.array(x),np.array(y)
    p0 = [x[np.argmin(y)],np.min(y),0.1,0.1] 
    fit_params,pcov = opt.curve_fit(morse,x,y,p0=p0, maxfev=5000)
    return fit_params[0],fit_params[1],np.sqrt(pcov[0,0]),np.sqrt(pcov[1,1]),fit_params

data = {}
err = {}
hardware = ['kolkata','auckland'] #,'qasm_kolkata','qasm_auckland']
R_list = ['0.80','0.90','1.00','1.10','1.20']
Range = np.arange(0.8,1.2,0.001)

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
                f = open('%s/%s/R_%s_%s_%s/results.txt' % (s,R,R,hw,p),'r')
                f = [x.split() for x in f.readlines()]
                f = f[len(f)-1]
                ave = float(f[9])
                std = float(f[11])
                data['%s/%s_%s' % (s,hw,p)].append(ave) 
                err['%s/%s_%s' % (s,hw,p)].append(std)
                
                         
for s in ['anion','radical']:
    for R in ['0.80','0.90','1.00','1.10','1.20']:
        f = open('%s/%s/R_%s/results.txt' % (s,R,R),'r')
        f = [x.split() for x in f.readlines()]
        f = f[len(f)-3]
        ave = float(f[5])
        data['%s' % (s)].append(ave)
        

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['cmu serif'],'size':12})
rc('text', usetex=True)

L = 5.0
fig,ax = plt.subplots(2,3,figsize=(3.2*L,2.5*0.66*L))
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

x0,s0,dx0,dy0,par = find_minimum(R_list,state_anion); ax1.scatter(x0,s0,marker='X',s=3*ms,c='black',edgecolor='black',linewidth=0.5)
E_anion   = [morse(Ri,par[0],par[1],par[2],par[3]) for Ri in Range]
ax1.plot(Range,E_anion, c='black', ls='--', label='statevector', ms=2)
#ax1.plot(R_list,state_anion,c='black', marker='+', label='statevector', ms=2)


x1,s1,dx1,dy1,par = find_minimum(R_list,state_radical); ax2.scatter(x1,s1,marker='X',s=3*ms,c='black',edgecolor='black',linewidth=0.5)
E_radical = [morse(Ri,par[0],par[1],par[2],par[3]) for Ri in Range]
ax2.plot(Range,E_radical,c='black', ls='--', label='statevector',ms=2)
#ax2.plot(R_list,state_radical,c='black', marker='+', label='statevector',ms=2)

ax4.plot(R_list,np.zeros(len(R_list)),c='black', ls='--', label='statevector',ms=2)
ax5.plot(R_list,np.zeros(len(R_list)),c='black', ls='--', label='statevector',ms=2)


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
        ax1.errorbar(R_list,anion,yerr=da,c=c, marker=m,capsize=3, elinewidth=3, ms=2, fmt=' ')
        ax2.errorbar(R_list,radical,yerr=dr,c=c, marker=m,capsize=3, elinewidth=3, ms=2, fmt=' ')
        #ax4.errorbar(R_list,anion-state_anion, yerr=da,c=c, marker=m,capsize=3, elinewidth=3,label=l, ms=2, fmt=' ')
        #ax5.errorbar(R_list,radical-state_radical, yerr=dr,c=c, marker=m,capsize=3, elinewidth=3,label=l, ms=2, fmt=' ')
        
        try:
            x0,y0,dx0,dy0,par = find_minimum(R_list,anion); ax1.scatter(x0,y0,marker='X',s=3*ms,c=c,edgecolor='black',linewidth=0.5)
            anion = [morse(Ri,par[0],par[1],par[2],par[3]) for Ri in Range]
            ax1.plot(Range,anion, c=c, ls='--',label=l)
            ax4.plot(Range,np.array(anion)-np.array(E_anion), c=c, ls='--', label=l)


            x1,y1,dx1,dy1,par = find_minimum(R_list,radical); ax2.scatter(x1,y1,marker='X',s=3*ms,c=c,edgecolor='black',linewidth=0.5)
            radical = [morse(Ri,par[0],par[1],par[2],par[3]) for Ri in Range]
            ax2.plot(Range,radical, c=c, ls='--',label=l)
            ax5.plot(Range,np.array(radical)-np.array(E_radical), c=c, ls='--', label=l)


            ax3.errorbar(x,kcal*np.array(y1-y0),yerr=kcal*np.array(np.sqrt(dy0**2+dy1**2)),c=c,marker='X',capsize=5, elinewidth=3,label=l, ls='none')
            ax6.errorbar(x,kcal*np.array((y1-y0)-(s1-s0)),yerr=kcal*np.array(np.sqrt(dy0**2+dy1**2)),c=c,marker='X',capsize=5, elinewidth=3,label=l, ls='none')
            print('Dati: ', hw, ' ' , p,' ', kcal*np.array(y1-y0), ' ' ,kcal*np.array(np.sqrt(dy0**2+dy1**2)))
        except:
            pass
        x-=1


fill_panel(ax1,'$R_{\mathrm{OH}} \, [\mathrm{\AA}]$',[0.8,1.2],[0.8,0.9,1.0,1.1,1.2],['0.8','0.9','1.0','1.1','1.2'],
               '$E_{\mathrm{anion}} \, [E_h]$',[-75.65,-75.51],[-75.65,-75.62,-75.59,-75.56,-75.53],
                                                  ['-75.65','-75.62','-75.59','-75.56','-75.53'])
fill_panel(ax4,'$R_{\mathrm{OH}} \, [\mathrm{\AA}]$',[0.8,1.2],[0.8,0.9,1.0,1.1,1.2],['0.8','0.9','1.0','1.1','1.2'],
               '$E_{\mathrm{anion}} - E_{\mathrm{statevector}} \, [E_h]$',[0.00,0.045],[0.00,0.01,0.02,0.03,0.04],
                                                  ['0.00','0.01','0.02','0.03','0.04'])

#[0.00,0.02],[0.000,0.005,0.010,0.015,0.020],
#                                                  ['0.000','0.005','0.010','0.015','0.020'])
                                                  
fill_panel(ax2,'$R_{\mathrm{OH}} \, [\mathrm{\AA}]$',[0.8,1.2],[0.8,0.9,1.0,1.1,1.2],['0.8','0.9','1.0','1.1','1.2'],
               '$E_{\mathrm{radical}} \, [E_h]$',[-75.65,-75.51],[-75.65,-75.62,-75.59,-75.56,-75.53],
                                                  ['-75.65','-75.62','-75.59','-75.56','-75.53'])

fill_panel(ax5,'$R_{\mathrm{OH}} \, [\mathrm{\AA}]$',[0.8,1.2],[0.8,0.9,1.0,1.1,1.2],['0.8','0.9','1.0','1.1','1.2'],
               '$E_{\mathrm{radical}} - E_{\mathrm{statevector}} \, [E_h]$',[0.00,0.045],[0.00,0.01,0.02,0.03,0.04],
                                                  ['0.00','0.01','0.02','0.03','0.04'])

fill_panel(ax3,'',[0,7],[1,2,3,4,5,6],['kolkata RAW','kolkata EM','kolkata EM+RE','auckland RAW','auckland EM','auckland EM+RE'],
                  '$E_{\mathrm{radical}}-E_{\mathrm{anion}}$ [kcal/mol]',[25,49],[28,35,42,49],[28,35,42,49])
fill_panel(ax6,'',[0,7],[1,2,3,4,5,6],['kolkata RAW','kolkata EM','kolkata EM+RE','auckland RAW','auckland EM','auckland EM+RE'],
                  '$\Delta E-E_{\mathrm{statevector}}$ [kcal/mol]',[-8,8],[-8,-5,-1,2,5,8],[-8,-5,-1,2,5,8])

ax3.set_xticklabels(ax3.get_xticklabels(),rotation = 30,fontsize=9,ha='right')
ax6.set_xticklabels(ax6.get_xticklabels(),rotation = 30,fontsize=9,ha='right')

x0L,y0L,dxL,dyL = 0.00,1.05,3.44,0.20
handles,labels  = ax1.get_legend_handles_labels()
lgd = ax1.legend(handles,labels,ncol=2,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=3)

fig.savefig('new_fig.pdf',format='pdf',bbox_inches='tight')
