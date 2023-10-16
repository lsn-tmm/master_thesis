import numpy as np
import scipy.optimize as opt
import sys
sys.path.append('../')

kcal   = 627.509
c_list = {'purple'       : '#B284BE',
          'jacaranda'    : '#888FC7',
          'light_yellow' : '#EEEA62',
          'gold'         : '#FFD300',
          'earwax'       : '#DAA520',
          'brown'        : '#7B3F00',
          'light_blue'   : '#bcd9ea',
          'cerulean'     : '#5ba4cf',
          'cobalt'       : '#0079bf',
          'dark_blue'    : '#055a8c',
          'light_green'  : '#acdf87',
          'yellow_green' : '#B9D146',
          'mid_green'    : '#68bb59',
          'dark_green'   : '#1e5631',
          'orange'       : '#FF8856',
          'red'          : '#DC343B',
          'light-gray'   : '#C0C0C0',
          'palatinate'   : '#72246C'}

def make_empty_elist():
    E_dict = {'iao'  : {'Ry':[],'qUCCSD':[],'nevpt2(fci,fci)':[],'nevpt2(fci,qse)':[],'nevpt2(qUCCSD,qse)':[],'nevpt2(Ry,qse)':[],'fci':[]},
              'full' : {'scf':[],'fci':[]}}
    return E_dict

def morse(x,Re,E0,De,ke):
    return E0+De*( 1-np.exp(-np.sqrt(ke/2.0/De)*(x-Re)) )**2

def find_minimum(x,y):
    x,y = np.array(x),np.array(y)
    p0 = [x[np.argmin(y)],np.min(y),0.1,0.1] 
    fit_params,pcov = opt.curve_fit(morse,x,y,p0=p0)
    return fit_params[0],fit_params[1],np.sqrt(pcov[0,0]),np.sqrt(pcov[1,1])

def fill_panel(pan,xlabel,xlim,xticks,xticklabels,ylabel,ylim,yticks,yticklabels,p=20.0,q=20.0):
    x0,x1 = xlim
    xlim  = [x0-(x1-x0)/p,x1+(x1-x0)/p]
    pan.set_xlabel(xlabel)
    pan.set_xlim(xlim)
    pan.set_xticks(xticks)
    pan.set_xticklabels(xticklabels)
    pan.set_ylabel(ylabel)
    y0,y1 = ylim
    ylim  = [y0-(y1-y0)/q,y1+(y1-y0)/q]
    pan.set_ylim(ylim)
    pan.set_yticks(yticks)
    pan.set_yticklabels(yticklabels)
    pan.tick_params(direction='in',which='both')

R_list    = ['0.80','0.82','0.84','0.86','0.88','0.90','0.92','0.94','0.96','0.98','1.00','1.02','1.04','1.06','1.08','1.10','1.12','1.14','1.16','1.18','1.20']
E_anion   = make_empty_elist()
E_radical = make_empty_elist()

def append_energies(species,R,E_list):
    f = open('%s/R_%s/results.txt'%(species,R),'r')
    f = f.readlines()
    f = [x.split() for x in f]
    E_list['full']['scf'].append(float(f[0][5]))
    E_list['iao']['fci'].append(float(f[7][5]))
    E_list['full']['fci'].append(float(f[9][5]))
    E_list['iao']['Ry'].append(float(f[15][3]))
    E_list['iao']['qUCCSD'].append(float(f[13][2]))
    E_list['iao']['nevpt2(fci,fci)'].append(float(f[10][5]))
    E_list['iao']['nevpt2(fci,qse)'].append(float(f[11][5]))
    E_list['iao']['nevpt2(qUCCSD,qse)'].append(float(f[12][5]))
    E_list['iao']['nevpt2(Ry,qse)'].append(float(f[14][5]))
    return E_list

for R in R_list:
    E_anion   = append_energies('anion',R,E_anion)
    print("anion")
    E_radical = append_energies('radical',R,E_radical) 
    print(E_radical)
    print(R)

R_list = [float(R) for R in R_list]

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['cmu serif'],'size':12})
rc('text', usetex=True)

L = 5.5 
fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(3*L,1*0.66*L))
plt.subplots_adjust(wspace=0.2,hspace=0.2)

m = 0
for x,k1,k2,l,c,ls in zip([R_list,R_list,R_list,R_list,R_list,R_list,R_list,R_list],
    ['full','iao','iao','iao','iao','iao','iao','iao'],
    ['scf','Ry','qUCCSD','fci','nevpt2(Ry,qse)','nevpt2(qUCCSD,qse)','nevpt2(fci,qse)','nevpt2(fci,fci)'],
    ['SCF','$R_y$/IAO','q-UCCSD/IAO','FCI/IAO','NEVPT2(R$_y$,QSE)','NEVPT2(q-UCCSD,QSE)','NEVPT2(FCI,QSE)','NEVPT2(FCI,FCI)'],
    [c_list['cobalt'],c_list['yellow_green'],c_list['mid_green'],c_list['dark_green'],c_list['gold'],c_list['earwax'],c_list['orange'],c_list['red']],
    [':','-.','-.','-.','--','--','--','--']):
    y = E_anion[k1][k2]
    z = E_radical[k1][k2]
    ax1.plot(x,y,c=c,ls=ls)
    ax2.plot(x,z,c=c,ls=ls)
    x0,y0,dx0,dy0 = find_minimum(x,y); ax1.scatter(x0,y0,marker='X',c=c)
    x1,y1,dx1,dy1 = find_minimum(x,z); ax2.scatter(x1,y1,marker='X',c=c)
    ax1.plot(x,[np.nan]*len(y),label=l,c=c,ls=ls,marker='X')
    print("%s & %.5f(%d) & %.5f(%d) & %.3f(%d) \\\\" % (l,x0,int(dx0*1e5),x1,int(dx1*1e5),kcal*(y0-y1),int(1e3*kcal*np.sqrt(dy0**2+dy1**2))))
    ax3.scatter(m,kcal*(y1-y0),c=c)
    m += 1

'''
fill_panel(ax1,'$R_{\mathrm{OH}} [\mathrm{\AA}]$',[0.9,1.2],[0.9,1.0,1.1,1.2],['0.9','1.0','1.1','1.2'],
               '$E_{\mathrm{anion}} [E_h]$',[-75.44,-75.28],[-75.44,-75.40,-75.36,-75.32,-75.28],
                                                  ['-75.44','-75.40','-75.36','-75.32','-75.28'])
fill_panel(ax2,'$R_{\mathrm{OH}} [\mathrm{\AA}]$',[0.9,1.2],[0.9,1.0,1.1,1.2],['0.9','1.0','1.1','1.2'],
               '$E_{\mathrm{radical}} [E_h]$',[-75.44,-75.28],[-75.44,-75.40,-75.36,-75.32,-75.28],
                                                  ['-75.44','-75.40','-75.36','-75.32','-75.28'])
fill_panel(ax3,'',[0,8],[0,1,2,3,4,5,6,7,8],['SCF','$R_y$/IAO','q-UCCSD/IAO','FCI/IAO','NEVPT2(R$_y$,QSE)','NEVPT2(q-UCCSD,QSE)','NEVPT2(FCI,QSE)','NEVPT2(FCI,FCI)','FCI'],
                  '$E_{\mathrm{anion}}-E_{\mathrm{radical}}$ [kcal/mol]',[10,35],[10,15,20,25,30,35],[10,15,20,25,30,35])
ax3.set_xticklabels(ax3.get_xticklabels(),rotation = 30,fontsize=9,ha='right')
'''

x0L,y0L,dxL,dyL = 0.00,1.05,3.40,0.20
handles,labels  = ax1.get_legend_handles_labels()
lgd = ax1.legend(handles,labels,ncol=5,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=3,fontsize=10)

fig.savefig('fig.pdf',format='pdf',bbox_inches='tight')

