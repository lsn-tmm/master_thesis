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
            data[hw][n]['RAW']   = np.load(hw+'/'+n+'/tomo_HW_raw_bloch.npy',allow_pickle=True).item()
            data[hw][n]['EM']    = np.load(hw+'/'+n+'/tomo_HW_em_bloch.npy',allow_pickle=True).item()
            data[hw][n]['EM+RE'] = np.load(hw+'/'+n+'/tomo_HW_em_re_extrapolated_bloch.npy',allow_pickle=True).item()
    return data


hardware = ['manila','lima','quito','santiago','bogota']
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

data_raw   = {}
data_em    = {}
data_em_re = {}

print(ideal['radical_aug-cc-pvtz'].shape)


L = 5.0
fig,axs = plt.subplots(2,2,figsize=(2*L,1.85*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.22)

delta = 0

for (x,y),k in zip([(0,0),(0,1),(1,0),(1,1)],A.keys()):

    a = ideal[k]
    
    data_raw[k]   = np.zeros(a.shape)
    data_em[k]    = np.zeros(a.shape)
    data_em_re[k] = np.zeros(a.shape)
    count = 0
    
    for hw in data.keys():
        for n in data[hw].keys():
            data_raw[k][:,0]    = data_raw[k][:,0]+(data[hw][n]['RAW'][k][:,0]-a[:,0]*delta)
            data_em[k][:,0]     = data_em[k][:,0]+(data[hw][n]['EM'][k][:,0]-a[:,0]*delta)
            data_em_re[k][:,0]  = data_em_re[k][:,0]+(data[hw][n]['EM+RE'][k][:,0]-a[:,0]*delta)
            data_raw[k][:,1]    = data_raw[k][:,1]+data[hw][n]['RAW'][k][:,1]**2
            data_em[k][:,1]     = data_em[k][:,1]+data[hw][n]['EM'][k][:,1]**2
            data_em_re[k][:,1]  = data_em_re[k][:,1]+data[hw][n]['EM+RE'][k][:,1]**2
            count += 1
          
    data_raw[k][:,0] = data_raw[k][:,0]/count
    data_em[k][:,0] = data_em[k][:,0]/count
    data_em_re[k][:,0] = data_em_re[k][:,0]/count
    
          
    data_raw[k][:,1] = np.sqrt(data_raw[k][:,1]/ (count**2))
    data_em[k][:,1] = np.sqrt(data_em[k][:,1]/ (count**2))
    data_em_re[k][:,1] = np.sqrt(data_em_re[k][:,1]/ (count**2))
        
    pa = [x for x in range(a.shape[0])]
    
    axs[x,y].plot(pa,a[:,0]-a[:,0],label='exact',marker='o',ls='none',ms=3)
    
    axs[x,y].errorbar(pa,data_raw[k][:,0],yerr=data_raw[k][:,1],label='manila average RAW',marker='x',ls='none', ms=5, c='#DC343B')
    
    axs[x,y].errorbar(pa,data_em[k][:,0], yerr=data_em[k][:,1],label='manila average EM',marker='1',ls='none', ms=5, c='#FF8856')
    axs[x,y].errorbar(pa,data_em_re[k][:,0],yerr=data_em_re[k][:,1],label='manila average EM+RE',marker='2',ms=6,ls='none', c='#228B22')


    axs[x,y].text(128,0.15,k.replace('_',' '),ha='center')
    axs[x,y].set_xlabel(r'index $\mu$')
    axs[x,y].set_ylabel(r'$\mbox{Tr}[\rho P_\mu] - \mbox{Tr}[\rho_{exact} P_\mu]$')
    axs[x,y].set_ylim([-0.45,0.45])
    
np.save('all_raw_bloch.npy',data_raw,allow_pickle=True)
np.save('all_em_bloch.npy',data_em,allow_pickle=True)
np.save('all_em_re_extrapolated_bloch.npy',data_em_re,allow_pickle=True)
    

x0L,y0L,dxL,dyL = 0.00,1.05,2.22,0.20
handles,labels  = axs[0,0].get_legend_handles_labels()
lgd = axs[0,0].legend(handles,labels,ncol=5,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=2)
#plt.savefig('plot/'+ hardware[0] + '_average_hw.pdf',format='pdf')
#plt.savefig('plot/manila_average.pdf',format='pdf')
