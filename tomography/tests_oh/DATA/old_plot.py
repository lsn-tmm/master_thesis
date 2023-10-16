import numpy as np
import sys,os

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['cmu serif'],'size':12})
rc('text', usetex=True)

sys.path.append('../../')
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
    names = {}
    for hw in hardware:
        base = os.walk('./'+hw)       
        names[hw] = []
        for a, dirs, files in os.walk('./'+hw):
            names[hw] += dirs
        names[hw].sort()
        print(hw, ' ', names[hw])
    for hw in hardware:
        data[hw] = {}
        for n in names[hw]:
            data[hw][n] = {}
            data[hw][n]['RAW'] = np.load(hw+'/'+n+'/tomo_HW_raw_bloch.npy',allow_pickle=True).item()
            data[hw][n]['EM'] = np.load(hw+'/'+n+'/tomo_HW_em_bloch.npy',allow_pickle=True).item()
            data[hw][n]['EM+RE'] = np.load(hw+'/'+n+'/tomo_HW_em_re_extrapolated_bloch.npy',allow_pickle=True).item()
    return data


hardware = ['manila'] #,'lima','quito','santiago','bogota']#, 'belem']
data = collect_data(hardware)

A = np.load('../circuits.npy',allow_pickle=True).item()

for hw in data.keys():
    for n in data[hw].keys():
        for k in A.keys():
            data[hw][n]['RAW'][k] = filter(data[hw][n]['RAW'][k],4)
            data[hw][n]['EM'][k] = filter(data[hw][n]['EM'][k],4)
            data[hw][n]['EM+RE'][k] = filter(data[hw][n]['EM+RE'][k],4)
    
for hw in data.keys():
    print('Keys: ', data[hw].keys())
    
ideal = np.load('tomo_statevector_bloch.npy',allow_pickle=True).item()

# ------------------------------------------------ Average data all hardware

L = 5.0
fig,axs = plt.subplots(2,2,figsize=(2*L,2*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.22)

for (x,y),k in zip([(0,0),(0,1),(1,0),(1,1)],A.keys()):

    a = ideal[k]

    data_raw   = a[:,0]-a[:,0]
    data_em    = a[:,0]-a[:,0]
    data_em_re = a[:,0]-a[:,0]
    count = 0
    
    for hw in data.keys():
        for n in data[hw].keys():
            data_raw    = data_raw + (data[hw][n]['RAW'][k][:,0] - a[:,0])
            data_em     = data_em + (data[hw][n]['EM'][k][:,0] - a[:,0])
            data_em_re  = data_em_re + (data[hw][n]['EM+RE'][k][:,0] - a[:,0])
            count += 1
            
    data_raw = data_raw/count
    data_em = data_em/count
    data_em_re = data_em_re/count
            
    '''        
    m2 = data['manila']['0']['EM'][k]
    l2 = data['lima']['0']['EM'][k]
    q2 = data['quito']['0']['EM'][k]
    b2 = data['belem']['0']['EM'][k]
    m3 = data['manila']['0']['EM+RE'][k]
    l3 = data['lima']['0']['EM+RE'][k]
    q3 = data['quito']['0']['EM+RE'][k]
    b3 = data['belem']['0']['EM+RE'][k]
    data_em    = ((m2[:,0]-a[:,0]) + (l2[:,0]-a[:,0]) + (q2[:,0]-a[:,0]) + (b2[:,0]-a[:,0]))/4
    data_em_re = ((m3[:,0]-a[:,0]) + (l3[:,0]-a[:,0]) + (q3[:,0]-a[:,0]) + (b3[:,0]-a[:,0]))/4
    '''
    
    pa = [x for x in range(a.shape[0])]
    
    axs[x,y].plot(pa,a[:,0]-a[:,0],label='exact',marker='o',ls='none',ms=3)
    axs[x,y].plot(pa,data_raw,label='hw average RAW',marker='x',ls='none', ms=5, c='#DC343B')
    axs[x,y].plot(pa,data_em,label='hw average EM',marker='1',ls='none', ms=5, c='#FF8856')
    axs[x,y].plot(pa,data_em_re,label='hw average EM RE',marker='2',ms=6,ls='none', c='#228B22')


    axs[x,y].text(128,0.15,k.replace('_',' '),ha='center')
    axs[x,y].set_xlabel(r'index $\mu$')
    axs[x,y].set_ylabel(r'$\mbox{Tr}[\rho P_\mu] - \mbox{Tr}[\rho_{exact} P_\mu]$')

x0L,y0L,dxL,dyL = 0.00,1.05,2.22,0.20
handles,labels  = axs[0,0].get_legend_handles_labels()
lgd = axs[0,0].legend(handles,labels,ncol=5,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=2)
plt.savefig('plot/average_manila_hw.pdf',format='pdf')
