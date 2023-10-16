import numpy as np
import sys
sys.path.append('../nevpt2/src/')
from plot import fill_panel, c_list, kcal

# location of data aug-cc-pv*z/*/R_1.00_*
# cc-pvtz/cc-pvqz
# anion/radical
# statevector/qasm/nm/nm_em/nm_em_re

v = {}
hardware = ['qasm','bogota','lima','santiago','quito','manila']

for hw in hardware:
    for b in ['aug-cc-pvtz','aug-cc-pvqz']:
        for s in ['anion','radical']:
            for p in ['raw','em','em_re']:
                f = open('%s/%s/R_1.00_%s_%s/results.txt' % (b,s,hw,p),'r')
                f = [x.split() for x in f.readlines()]
                f = f[len(f)-1]
                ave = float(f[9])
                std = float(f[11])
                v['%s/%s/%s_%s' % (b,s,hw,p)] = [ave,std]
                
path = '../nevpt2_old/'
                            
for b in ['aug-cc-pvtz','aug-cc-pvqz']:
    for s in ['anion','radical']:
        f = open(path+'%s/%s/R_1.00_statevector/results.txt' % (b,s),'r')
        f = [x.split() for x in f.readlines()]
        f = f[len(f)-1]
        ave = float(f[9])
        std = float(f[11])
        v['%s/%s/statevector' % (b,s)] = [ave,std]
            

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['cmu serif'],'size':12})
rc('text', usetex=True)


outfile1 = open('table_tz.txt','w')
outfile2 = open('table_qz.txt','w')
    
L = 5.0
fig,ax = plt.subplots(1,2,figsize=(3.5*L,1.2*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.40)
ax1=ax[0]
ax2=ax[1]
    
x=1
colors = [ ['#736F6E','#B6B6B4','#D1D0CE'],[c_list['red'],c_list['orange'],c_list['light_yellow']],[c_list['dark_blue'],c_list['cobalt'],c_list['cerulean']],[c_list['dark_green'],c_list['mid_green'],c_list['yellow_green']],['#342D7E',c_list['jacaranda'],c_list['purple']],[c_list['brown'],c_list['earwax'],c_list['gold']] ]
markers = [['o','s','+'],['o','s','+'],['o','s','+'],['o','s','+'],['o','s','+'],['o','s','+']]

for hw,col,mrk in zip(hardware,colors,markers):
    
    
    for p,c,m in zip(['em_re','em','raw'],col,mrk):
        y   = [v['%s/radical/%s_%s' % (b,hw,p)][0]-v['%s/anion/%s_%s' % (b,hw,p)][0]                for b in ['aug-cc-pvtz','aug-cc-pvqz']]
        dy  = [np.sqrt(v['%s/radical/%s_%s' % (b,hw,p)][1]**2+v['%s/anion/%s_%s' % (b,hw,p)][1]**2) for b in ['aug-cc-pvtz','aug-cc-pvqz']]
        if   (p == 'em_re') : l = hw+' EM+RE'
        elif (p == 'em') : l = hw+' EM'
        elif (p == 'raw') : l = hw+' RAW'
        ax1.errorbar(x,kcal*np.array(y[0]),yerr=kcal*np.array(dy[0]),c=c, marker=m,capsize=5, elinewidth=3,label=l, ls='none')
        ax2.errorbar(x,kcal*np.array(y[1]),yerr=kcal*np.array(dy[1]),c=c, marker=m,capsize=5, elinewidth=3,label=l, ls='none')
        
        ey  = int( (kcal*np.array(y[0])*1e3-int(kcal*np.array(y[0])*1e3))*1e2 )
        edy = int( ( kcal*np.array(dy[0])*1e3-int(kcal*np.array(dy[0])*1e3))*1e2 )
        
        outfile1.write("%s %s & %.3f(%d) \pm %.3f(%d) \\\\ \n" % (hw, l, kcal*np.array(y[0]), ey, kcal*np.array(dy[0]), edy) )
        
        ey  = int( (kcal*np.array(y[1])*1e3-int(kcal*np.array(y[1])*1e3))*1e2 )
        edy = int( ( kcal*np.array(dy[1])*1e3-int(kcal*np.array(dy[1])*1e3))*1e2 )
        
        outfile2.write("%s %s & %.3f(%d) \pm %.3f(%d) \\\\ \n" % ( hw, l, kcal*np.array(y[1]), ey, kcal*np.array(dy[1]), edy ) )
        
    x+=1
        
line = [v['%s/radical/statevector' % (b)][0]-v['%s/anion/statevector' % (b)][0]                for b in ['aug-cc-pvtz','aug-cc-pvqz']]
        
ax1.plot( [0,7], kcal*np.ones(2)*float(line[0]), c='black', ls='--', linewidth=0.5, label='statevector')
ax2.plot( [0,7], kcal*np.ones(2)*float(line[1]), c='black', ls='--', linewidth=0.5)

#ax1.plot( [0,7], np.ones(2)*32.17014, c='black', ls='--', linewidth=0.5, label='statevector')
#ax2.plot( [0,7], np.ones(2)*33.38716, c='black', ls='--', linewidth=0.5)
    
    
ey = int( (kcal*np.array(line[0])*1e3-int(kcal*np.array(line[0])*1e3))*1e2 )

outfile1.write("statevector & %.3f(%d) \pm / \\\\ \n" % (kcal*np.array(line[0]), ey))

ey = int( (kcal*np.array(line[1])*1e3-int(kcal*np.array(line[1])*1e3))*1e2 )

outfile2.write("statevector & %.3f(%d) \pm / \\\\ \n" % (kcal*np.array(line[1]), ey  ))
        
        
    
        
      
fill_panel(ax1,'',[0,7],[1,2,3,4,5,6],['qasm','bogota','lima','santiago','quito','manila'],
                  '$E_{\mathrm{radical}}-E_{\mathrm{anion}}$ [kcal/mol]',[10,40],[10,20,30,40],[10,20,30,40])
fill_panel(ax2,'',[0,7],[1,2,3,4,5,6],['qasm','bogota','lima','santiago','quito','manila'],
                  '$E_{\mathrm{radical}}-E_{\mathrm{anion}}$ [kcal/mol]',[10,40],[10,20,30,40],[10,20,30,40])


ax1.set_xticklabels(ax1.get_xticklabels(),rotation = 30,fontsize=9,ha='right') 
ax2.set_xticklabels(ax2.get_xticklabels(),rotation = 30,fontsize=9,ha='right')   
       
            
x0L,y0L,dxL,dyL = 0.00,1.05,2.15,0.20
handles,labels  = ax1.get_legend_handles_labels()
lgd2 = ax1.legend([handles[0]],['statevector'], loc='upper right')
lgd = ax1.legend(handles[1:],labels[1:],ncol=6,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=3)
fig.savefig('new_nevpt2_hw.pdf',format='pdf',bbox_inches='tight')


