import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['cmu serif'],'size':12})
rc('text', usetex=True)

import sys
sys.path.append('../../../')
from utilities import index_to_label

def filter(bloch,n):
    for mu in range(4**n):
         label = index_to_label(mu,n)
         ny = len([x for x in label if x==2])
         if(ny%2==1):
            #print(mu,label,ny)
            bloch[mu,:] = (0,0)
    return bloch

def collect_data(hardware):
    data = {}
    for hw in hardware:
        data[hw] = {}
        data[hw]['EM'] = np.load(hw+'/tomo_HW_em_bloch.npy',allow_pickle=True).item()
        data[hw]['EM+RE'] = np.load(hw+'/tomo_HW_em_re_extrapolated_bloch.npy',allow_pickle=True).item()
    return data

hardware = ['manila','lima','quito','santiago','bogota']
data = collect_data(hardware)

A = np.load('../circuits.npy',allow_pickle=True).item()

for hw in data.keys():
    for k in A.keys():
        data[hw]['EM'][k] = filter(data[hw]['EM'][k],4)
        data[hw]['EM+RE'][k] = filter(data[hw]['EM+RE'][k],4)
    
for hw in data.keys():
    print('Keys: ', data[hw].keys())
    
ideal = np.load('../tomo_statevector_bloch.npy',allow_pickle=True).item()

# ------------------------------------------------ Average data all hardware

L = 5.0
fig,axs = plt.subplots(2,2,figsize=(2*L,2*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.22)

for (x,y),k in zip([(0,0),(0,1),(1,0),(1,1)],A.keys()):

    a = ideal[k]
    
    data_em    = a[:,0]-a[:,0]
    data_em_re = a[:,0]-a[:,0]
    count = 0
    
    for hw in data.keys():
        data_em     = data_em + (data[hw]['EM'][k][:,0] - a[:,0])
        data_em_re  = data_em_re + (data[hw]['EM+RE'][k][:,0] - a[:,0])
        count += 1
            
    data_em = data_em/count
    data_em_re = data_em_re/count

    pa = [x for x in range(a.shape[0])]
    
    axs[x,y].plot(pa,a[:,0]-a[:,0],label='exact',marker='o',ls='none',ms=5)
    axs[x,y].plot(pa,data_em,label='hw average EM',marker='+',ls='none')
    axs[x,y].plot(pa,data_em_re,label='hw average EM RE',marker='x',ls='none')


    axs[x,y].text(128,0.15,k.replace('_',' '),ha='center')
    axs[x,y].set_xlabel(r'index $\mu$')
    axs[x,y].set_ylabel(r'$\mbox{Tr}[\rho P_\mu] - \mbox{Tr}[\rho_{exact} P_\mu]$')

x0L,y0L,dxL,dyL = 0.00,1.05,2.22,0.20
handles,labels  = axs[0,0].get_legend_handles_labels()
lgd = axs[0,0].legend(handles,labels,ncol=5,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=2)
plt.savefig('all_hw_average.pdf',format='pdf')

# ------------------------------------------------ Average EM and all hw

L = 5.0
fig,axs = plt.subplots(2,2,figsize=(2*L,2*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.22)

for (x,y),k in zip([(0,0),(0,1),(1,0),(1,1)],A.keys()):

    a = ideal[k]
    
    m2 = data['manila']['EM'][k]
    s2 = data['santiago']['EM'][k]
    l2 = data['lima']['EM'][k]
    q2 = data['quito']['EM'][k]
    #b2 = data['belem']['EM'][k]
    t2 = data['bogota']['EM'][k]
    
    data_em    = ((s2[:,0]-a[:,0]) + (l2[:,0]-a[:,0]) + (q2[:,0]-a[:,0]) + (m2[:,0] - a[:,0]) + (t2[:,0]-a[:,0]))/5
    manila     = m2[:,0]-a[:,0]
    santiago   = s2[:,0]-a[:,0]
    lima       = l2[:,0]-a[:,0]
    quito      = q2[:,0]-a[:,0]
    #belem      = b2[:,0]-a[:,0]
    bogota     = t2[:,0]-a[:,0]
    
    pa = [x for x in range(a.shape[0])]

    axs[x,y].plot(pa,manila,label='manila EM',marker='+',ls='none')
    axs[x,y].plot(pa,santiago,label='santiago EM',marker='+',ls='none')
    axs[x,y].plot(pa,lima,label='lima EM',marker='+',ls='none')
    axs[x,y].plot(pa,quito,label='quito EM',marker='+',ls='none')
    #axs[x,y].plot(pa,belem,label='belem EM',marker='+',ls='none')
    axs[x,y].plot(pa,bogota,label='bogota EM',marker='+',ls='none')
    axs[x,y].plot(pa,data_em,label='average EM',marker='+',ls='none', c='black')

   
    axs[x,y].text(128,0.15,k.replace('_',' '),ha='center')
    axs[x,y].set_xlabel(r'index $\mu$')
    axs[x,y].set_ylabel(r'$\mbox{Tr}[\rho P_\mu] - \mbox{Tr}[\rho_{exact} P_\mu]$')

x0L,y0L,dxL,dyL = 0.00,1.05,2.22,0.20
handles,labels  = axs[0,0].get_legend_handles_labels()
lgd = axs[0,0].legend(handles,labels,ncol=5,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=2)
plt.savefig('hw_re_data.pdf',format='pdf')

# ------------------------------------------------ Average EM RE and all hw

L = 5.0
fig,axs = plt.subplots(2,2,figsize=(2*L,2*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.22)

for (x,y),k in zip([(0,0),(0,1),(1,0),(1,1)],A.keys()):

    a = ideal[k]

    m3 = data['manila']['EM+RE'][k]
    s3 = data['santiago']['EM+RE'][k]
    l3 = data['lima']['EM+RE'][k]
    q3 = data['quito']['EM+RE'][k]
    #b3 = data['belem']['EM'][k]
    t3 = data['bogota']['EM'][k]

    data_em_re = ((s3[:,0]-a[:,0]) + (l3[:,0]-a[:,0]) + (q3[:,0]-a[:,0]) + (m3[:,0]-a[:,0]) + (t3[:,0]-a[:,0]))/5
    manila     = m3[:,0]-a[:,0]
    santiago   = s3[:,0]-a[:,0]
    lima       = l3[:,0]-a[:,0]
    quito      = q3[:,0]-a[:,0]
    #belem      = b3[:,0]-a[:,0]
    bogota     = t3[:,0]-a[:,0]
    
    pa = [x for x in range(a.shape[0])]
    
    axs[x,y].plot(pa,manila,label='manila EM RE',marker='+',ls='none')
    axs[x,y].plot(pa,santiago,label='santiago EM RE',marker='+',ls='none')
    axs[x,y].plot(pa,lima,label='lima EM RE',marker='+',ls='none')
    axs[x,y].plot(pa,quito,label='quito EM RE',marker='+',ls='none') 
    #axs[x,y].plot(pa,belem,label='belem EM RE',marker='+',ls='none')
    axs[x,y].plot(pa,bogota,label='bogota EM RE',marker='+',ls='none')
    axs[x,y].plot(pa,data_em_re,label='average EM RE',marker='+',ls='none', c='black')

    axs[x,y].text(128,0.15,k.replace('_',' '),ha='center')
    axs[x,y].set_xlabel(r'index $\mu$')
    axs[x,y].set_ylabel(r'$\mbox{Tr}[\rho P_\mu] - \mbox{Tr}[\rho_{exact} P_\mu]$')

x0L,y0L,dxL,dyL = 0.00,1.05,2.22,0.20
handles,labels  = axs[0,0].get_legend_handles_labels()
lgd = axs[0,0].legend(handles,labels,ncol=5,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=2)
plt.savefig('hw_re_em_data.pdf',format='pdf')
