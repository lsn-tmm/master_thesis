import numpy as np
import scipy.optimize as opt
import sys
sys.path.append('../../nevpt2/src/')
from plot import fill_panel, c_list, kcal



basis = ['sto-6g','6-31g','6-31++g','6-31g**','6-31++g**','cc-pvdz','cc-pvtz','cc-pvqz','aug-cc-pvdz','aug-cc-pvtz','aug-cc-pvqz']

colors = ['blue','red','red','red','red','orange','orange','orange','green','green','green']

data = np.load('oh_data.npy',allow_pickle=True).item()
for k in data.keys():
  data[k] = np.array(data[k])
x = [i for i in range(len(basis))]

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['cmu serif'],'size':12})
rc('text', usetex=True)

L = 5.0
fig,ax1 = plt.subplots(1,1,figsize=(2.0*L,1.3*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.20)

m=0
for n,c in zip(range(len(basis)),colors):
  ax1.scatter(m,kcal*(data['E_HF_radical'][n]-data['E_HF_anion'][n]), c=c, marker='o', s=16,edgecolor='black',linewidth=0.5)
  ax1.scatter(m,kcal*(data['E_CCSD(T)_radical'][n]-data['E_CCSD(T)_anion'][n]), c=c, marker='s', s=16,edgecolor='black',linewidth=0.5)
  m+=1


p = [i-1 for i in range(len(basis)+5)]
ax1.plot(p, np.ones(len(p))*42.15, c='black', ls='--', linewidth=0.5, label='experiment')
ax1.annotate(str(42.15) + ' kcal/mol', xy=(-0.25,43), fontsize=10, color='black')
ax1.plot(p, np.zeros(len(p)), c='#D1D0CE', linewidth=0.5)
ax1.scatter(0,np.nan,c='black', marker='o',label='HF')
ax1.scatter(0,np.nan,c='black', marker='s',label='CCSD(T)')

fill_panel(ax1,'',[0,10],[0,1,2,3,4,5,6,7,8,9,10],basis,
                  '$E_{\mathrm{radical}}-E_{\mathrm{anion}}$ [kcal/mol]',[-200,60],[-200,-150,-100,-50,0,50],[-200,-150,-100,-50,0,50])

ax1.set_xticklabels(ax1.get_xticklabels(),rotation = 30,fontsize=9,ha='right')

x0L,y0L,dxL,dyL = 0.00,1.05,1.00,0.20
handles,labels  = ax1.get_legend_handles_labels()
lgd = ax1.legend(handles,labels,ncol=3,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=3)

fig.savefig('basis.pdf',format='pdf',bbox_inches='tight')
