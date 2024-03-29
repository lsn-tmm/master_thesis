import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['cmu serif'],'size':12})
rc('text', usetex=True)

import sys
sys.path.append('../')
from utilities import index_to_label

def filter(bloch,n):
    for mu in range(4**n):
         label = index_to_label(mu,n)
         ny = len([x for x in label if x==2])
         if(ny%2==1):
            print(mu,label,ny)
            bloch[mu,:] = (0,0)
    return bloch

ideal         = np.load('tomo_statevector_bloch.npy',allow_pickle=True).item()
auckland_em    = np.load('auckland_em_bloch.npy',allow_pickle=True).item()
auckland_em_re = np.load('auckland_em_re_extrapolated_bloch.npy',allow_pickle=True).item()
auckland_raw   = np.load('auckland_raw_bloch.npy',allow_pickle=True).item()

A = np.load('circuits.npy',allow_pickle=True).item()

L = 5.0
fig,axs = plt.subplots(5,2,figsize=(8*L,1.85*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.22)

for (x,y),k in zip([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1)],A.keys()):
    a1 = filter(ideal[k],4)
    #a2 = filter(bloch_qasm[k],4)
    a3 = filter(auckland_raw[k],4)
    a4 = filter(auckland_em[k],4)
    a5 = filter(auckland_em_re[k],4)
    pa = [x for x in range(a1.shape[0])]
    axs[x,y].plot(pa,np.zeros(len(a1[:,0])),label='exact',marker='o',ls='none',ms=3)
    #axs[x,y].errorbar(pa,a2[:,0]-a1[:,0],yerr=a2[:,1],label='qasm(id)',marker='+',ls='none',c='#bcd9ea')
    axs[x,y].errorbar(pa,a3[:,0]-a1[:,0],yerr=a3[:,1],label='auckland RAW',marker='x', ms=5, ls='none', c='#DC343B')
    axs[x,y].errorbar(pa,a4[:,0]-a1[:,0],yerr=a4[:,1],label='auckland EM',marker='1', ms=5, ls='none', c='#FF8856')
    axs[x,y].errorbar(pa,a5[:,0]-a1[:,0],yerr=a5[:,1],label='auckland EM+RE',marker='2',ms=6,ls='none', c='#228B22')
    axs[x,y].text(2048,0.15,k.replace('_',' '),ha='center')
    axs[x,y].set_xlabel(r'index $\mu$')
    axs[x,y].set_ylabel(r'$\mbox{Tr}[\rho P_\mu] - \mbox{Tr}[\rho_{exact} P_\mu]$')

x0L,y0L,dxL,dyL = 0.00,1.05,2.22,0.20
handles,labels  = axs[0,0].get_legend_handles_labels()
lgd = axs[0,0].legend(handles,labels,ncol=2,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=2)
plt.savefig('bloch_auckland.pdf',format='pdf')

# ----------------------------------------------------------------------
