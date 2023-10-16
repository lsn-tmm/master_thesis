import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['cmu serif'],'size':12})
rc('text', usetex=True)

import sys
sys.path.append('../../')
from utilities import index_to_label

def filter(bloch,n):
    for mu in range(4**n):
         label = index_to_label(mu,n)
         ny = len([x for x in label if x==2])
         if(ny%2==1):
            print(mu,label,ny)
            bloch[mu,:] = (0,0)
    return bloch



bloch_ideal    = np.load('../manila/tomo_statevector_bloch.npy',allow_pickle=True).item()

manila_bloch_hw_em    = np.load('../manila/tomo_HW_em_bloch.npy',allow_pickle=True).item()
manila_bloch_hw_em_re = np.load('../manila/tomo_HW_em_re_extrapolated_bloch.npy',allow_pickle=True).item()

lima_bloch_hw_em    = np.load('../lima/tomo_HW_em_bloch.npy',allow_pickle=True).item()
lima_bloch_hw_em_re = np.load('../lima/tomo_HW_em_re_extrapolated_bloch.npy',allow_pickle=True).item()

quito_bloch_hw_em    = np.load('../quito/tomo_HW_em_bloch.npy',allow_pickle=True).item()
quito_bloch_hw_em_re = np.load('../quito/tomo_HW_em_re_extrapolated_bloch.npy',allow_pickle=True).item()

belem_bloch_hw_em    = np.load('../belem/tomo_HW_em_bloch.npy',allow_pickle=True).item()
belem_bloch_hw_em_re = np.load('../belem/tomo_HW_em_re_extrapolated_bloch.npy',allow_pickle=True).item()

A = np.load('../circuits.npy',allow_pickle=True).item()

# ------------------------------------------------ Average data all hardware

L = 5.0
fig,axs = plt.subplots(2,2,figsize=(2*L,2*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.22)

for (x,y),k in zip([(0,0),(0,1),(1,0),(1,1)],A.keys()):

	a = filter(bloch_ideal[k],4)

	m2 = filter(manila_bloch_hw_em[k],4)
	l2 = filter(lima_bloch_hw_em[k],4)
	q2 = filter(quito_bloch_hw_em[k],4)
	b2 = filter(belem_bloch_hw_em[k],4)

	m3 = filter(manila_bloch_hw_em_re[k],4)
	l3 = filter(lima_bloch_hw_em_re[k],4)
	q3 = filter(quito_bloch_hw_em_re[k],4)
	b3 = filter(belem_bloch_hw_em_re[k],4)

	data_em    = ((m2[:,0]-a[:,0]) + (l2[:,0]-a[:,0]) + (q2[:,0]-a[:,0]) + (b2[:,0]-a[:,0]))/4
	data_em_re = ((m3[:,0]-a[:,0]) + (l3[:,0]-a[:,0]) + (q3[:,0]-a[:,0]) + (b3[:,0]-a[:,0]))/4
	
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

	a = filter(bloch_ideal[k],4)

	m2 = filter(manila_bloch_hw_em[k],4)
	l2 = filter(lima_bloch_hw_em[k],4)
	q2 = filter(quito_bloch_hw_em[k],4)
	b2 = filter(belem_bloch_hw_em[k],4)

	data_em    = ((m2[:,0]-a[:,0]) + (l2[:,0]-a[:,0]) + (q2[:,0]-a[:,0]) + (b2[:,0]-a[:,0]))/4
	manila     = m2[:,0]-a[:,0]
	lima       = l2[:,0]-a[:,0]
	quito      = q2[:,0]-a[:,0]
	belem      = b2[:,0]-a[:,0]
	
	pa = [x for x in range(a.shape[0])]

	axs[x,y].plot(pa,manila,label='manila EM',marker='+',ls='none')
	axs[x,y].plot(pa,lima,label='lima EM',marker='+',ls='none')
	axs[x,y].plot(pa,quito,label='quito EM',marker='+',ls='none')
	axs[x,y].plot(pa,belem,label='belem EM',marker='+',ls='none')
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

	a = filter(bloch_ideal[k],4)

	m3 = filter(manila_bloch_hw_em_re[k],4)
	l3 = filter(lima_bloch_hw_em_re[k],4)
	q3 = filter(quito_bloch_hw_em_re[k],4)
	b3 = filter(belem_bloch_hw_em_re[k],4)

	data_em_re = ((m3[:,0]-a[:,0]) + (l3[:,0]-a[:,0]) + (q3[:,0]-a[:,0]) + (b3[:,0]-a[:,0]))/4
	manila     = m3[:,0]-a[:,0]
	lima       = l3[:,0]-a[:,0]
	quito      = q3[:,0]-a[:,0]
	belem      = b3[:,0]-a[:,0]
	
	pa = [x for x in range(a.shape[0])]
    
	axs[x,y].plot(pa,manila,label='manila EM RE',marker='+',ls='none')
	axs[x,y].plot(pa,lima,label='lima EM RE',marker='+',ls='none')
	axs[x,y].plot(pa,quito,label='quito EM RE',marker='+',ls='none')
	axs[x,y].plot(pa,belem,label='belem EM RE',marker='+',ls='none')
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



# ------------------------------------------------ Average lines EM

L = 5.0
fig,axs = plt.subplots(2,2,figsize=(2*L,2*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.22)

for (x,y),k in zip([(0,0),(0,1),(1,0),(1,1)],A.keys()):

	a = filter(bloch_ideal[k],4)

	m2 = filter(manila_bloch_hw_em[k],4)
	l2 = filter(lima_bloch_hw_em[k],4)
	q2 = filter(quito_bloch_hw_em[k],4)
	b2 = filter(belem_bloch_hw_em[k],4)

	data_em    = (m2[:,0]-a[:,0]) + (l2[:,0]-a[:,0]) + (q2[:,0]-a[:,0]) + (b2[:,0]-a[:,0])
	mean_em    = np.mean(data_em/4)
	manila     = np.mean(m2[:,0]-a[:,0])
	lima       = np.mean(l2[:,0]-a[:,0])
	quito      = np.mean(q2[:,0]-a[:,0])
	belem      = np.mean(b2[:,0]-a[:,0])
	
	pa = [x for x in range(a.shape[0])]
    
	#axs[x,y].plot(pa,a[:,0]-a[:,0],label='exact',marker='o',ls='none',ms=5)
	#axs[x,y].plot(pa,data_em/4,label='hw EM',marker='+',ls='none')

	axs[x,y].plot(pa, np.ones(256)*mean_em, label='mean EM', c='black', ls='--', linewidth=0.5)
	axs[x,y].plot(pa, np.ones(256)*manila, label='manila mean EM', ls='--', linewidth=0.5)
	axs[x,y].plot(pa, np.ones(256)*lima, label='lima mean EM', ls='--', linewidth=0.5)
	axs[x,y].plot(pa, np.ones(256)*quito, label='quito mean EM', ls='--', linewidth=0.5)
	axs[x,y].plot(pa, np.ones(256)*belem, label='belem mean EM', ls='--', linewidth=0.5)
	

	axs[x,y].annotate(str(round(mean_em,3)), xy=(1,mean_em), fontsize=10)
	axs[x,y].annotate(str(round(manila,3)), xy=(1,manila), fontsize=10)
	axs[x,y].annotate(str(round(lima,3)), xy=(1,lima), fontsize=10)
	axs[x,y].annotate(str(round(quito,3)), xy=(1,quito), fontsize=10)
	axs[x,y].annotate(str(round(belem,3)), xy=(1,belem), fontsize=10)
	
    	
	axs[x,y].text(128,0.15,k.replace('_',' '),ha='center')
	axs[x,y].set_xlabel(r'index $\mu$')
	axs[x,y].set_ylabel(r'$\mbox{Tr}[\rho P_\mu] - \mbox{Tr}[\rho_{exact} P_\mu]$')

	lim = max([manila,belem,lima,quito])+0.001
	axs[x,y].set_ylim([-lim,lim])

x0L,y0L,dxL,dyL = 0.00,1.05,2.22,0.20
handles,labels  = axs[0,0].get_legend_handles_labels()
lgd = axs[0,0].legend(handles,labels,ncol=5,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=2)
plt.savefig('hw_em_average.pdf',format='pdf')


# --------------------------------------- Average lines EM RE

L = 5.0
fig,axs = plt.subplots(2,2,figsize=(2*L,2*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.22)

for (x,y),k in zip([(0,0),(0,1),(1,0),(1,1)],A.keys()):

	a = filter(bloch_ideal[k],4)

	m3 = filter(manila_bloch_hw_em_re[k],4)
	l3 = filter(lima_bloch_hw_em_re[k],4)
	q3 = filter(quito_bloch_hw_em_re[k],4)
	b3 = filter(belem_bloch_hw_em_re[k],4)

	data_em_re = (m3[:,0]-a[:,0]) + (l3[:,0]-a[:,0]) + (q3[:,0]-a[:,0]) + (b3[:,0]-a[:,0])
	mean_em_re = np.mean(data_em_re/4)
	manila     = np.mean(m3[:,0]-a[:,0])
	lima       = np.mean(l3[:,0]-a[:,0])
	quito      = np.mean(q3[:,0]-a[:,0])
	belem      = np.mean(b3[:,0]-a[:,0])

	pa = [x for x in range(a.shape[0])]

	#axs[x,y].plot(pa,a[:,0]-a[:,0],label='exact',marker='o',ls='none',ms=5)
	#axs[x,y].plot(pa,data_em_re/4,label='hw, EM RE',marker='x',ls='none')

	axs[x,y].plot(pa, np.ones(256)*mean_em_re, label='mean EM RE', c='black', ls='--', linewidth=0.5)
	axs[x,y].plot(pa, np.ones(256)*manila, label='manila mean EM', ls='--', linewidth=0.5)
	axs[x,y].plot(pa, np.ones(256)*lima, label='lima mean EM', ls='--', linewidth=0.5)
	axs[x,y].plot(pa, np.ones(256)*quito, label='quito mean EM', ls='--', linewidth=0.5)
	axs[x,y].plot(pa, np.ones(256)*belem, label='belem mean EM', ls='--', linewidth=0.5)

	axs[x,y].annotate(str(round(mean_em_re,3)), xy=(1,mean_em_re), fontsize=10)
	axs[x,y].annotate(str(round(manila,3)), xy=(1,manila), fontsize=10)
	axs[x,y].annotate(str(round(lima,3)), xy=(1,lima), fontsize=10)
	axs[x,y].annotate(str(round(quito,3)), xy=(1,quito), fontsize=10)
	axs[x,y].annotate(str(round(belem,3)), xy=(1,belem), fontsize=10)

	axs[x,y].text(128,0.15,k.replace('_',' '),ha='center')
	axs[x,y].set_xlabel(r'index $\mu$')
	axs[x,y].set_ylabel(r'$\mbox{Tr}[\rho P_\mu] - \mbox{Tr}[\rho_{exact} P_\mu]$')
	
	lim = max([manila,belem,lima,quito])+0.001
	axs[x,y].set_ylim([-lim,lim])

x0L,y0L,dxL,dyL = 0.00,1.05,2.22,0.20
handles,labels  = axs[0,0].get_legend_handles_labels()
lgd = axs[0,0].legend(handles,labels,ncol=5,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=2)
plt.savefig('hw_em_re_average.pdf',format='pdf')

# ---------------------------- Histogram EM abs

L = 5.0
fig,axs = plt.subplots(2,2,figsize=(2*L,2*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.22)

for (x,y),k in zip([(0,0),(0,1),(1,0),(1,1)],A.keys()):

	a = filter(bloch_ideal[k],4)

	m2 = filter(manila_bloch_hw_em[k],4)
	l2 = filter(lima_bloch_hw_em[k],4)
	q2 = filter(quito_bloch_hw_em[k],4)
	b2 = filter(belem_bloch_hw_em[k],4)

	abs_data_em    = (abs(m2[:,0]-a[:,0]) + abs(l2[:,0]-a[:,0]) + abs(q2[:,0]-a[:,0]) + abs(b2[:,0]-a[:,0]))/4
	abs_mean_em    = np.mean(abs_data_em)

	manila = abs(m2[:,0]-a[:,0])
	lima   = abs(l2[:,0]-a[:,0])
	quito  = abs(q2[:,0]-a[:,0])
	belem  = abs(b2[:,0]-a[:,0])
	
	bins = 10
	
	pa = [x for x in range(a.shape[0])]
   
	axs[x,y].hist(manila, bins=bins,label='manila EM')
	axs[x,y].hist(lima, bins=bins,label='lima EM')
	axs[x,y].hist(quito, bins=bins,label='quito EM')
	axs[x,y].hist(belem, bins=bins,label='belem EM')
	axs[x,y].hist(abs_data_em, bins=bins,label='average EM')
    	
	axs[x,y].text(128,0.15,k.replace('_',' '),ha='center')
	axs[x,y].set_xlabel(r'index $\mu$')
	axs[x,y].set_ylabel(r'$\mbox{Tr}[\rho P_\mu] - \mbox{Tr}[\rho_{exact} P_\mu]$')

x0L,y0L,dxL,dyL = 0.00,1.05,2.22,0.20
handles,labels  = axs[0,0].get_legend_handles_labels()
lgd = axs[0,0].legend(handles,labels,ncol=5,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=2)
plt.savefig('hist_abs_em.pdf',format='pdf')


# ----------------------------------- Histogram EM RE abs

L = 5.0
fig,axs = plt.subplots(2,2,figsize=(2*L,2*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.22)

for (x,y),k in zip([(0,0),(0,1),(1,0),(1,1)],A.keys()):

	a = filter(bloch_ideal[k],4)

	m3 = filter(manila_bloch_hw_em_re[k],4)	
	l3 = filter(lima_bloch_hw_em_re[k],4)	
	q3 = filter(quito_bloch_hw_em_re[k],4)	
	b3 = filter(belem_bloch_hw_em_re[k],4)

	abs_data_em_re = (abs(m3[:,0]-a[:,0]) + abs(l3[:,0]-a[:,0]) + abs(q3[:,0]-a[:,0]) + abs(b3[:,0]-a[:,0]))/4
	abs_mean_em_re = np.mean(abs_data_em_re)

	manila = abs(m3[:,0]-a[:,0])
	lima   = abs(l3[:,0]-a[:,0])
	quito  = abs(q3[:,0]-a[:,0])
	belem  = abs(b3[:,0]-a[:,0])
	
	#bins = 6
	
	pa = [x for x in range(a.shape[0])]
   
	axs[x,y].hist(manila, bins=bins,label='manila EM RE')
	axs[x,y].hist(lima, bins=bins,label='lima EM RE')
	axs[x,y].hist(quito, bins=bins,label='quito EM RE')
	axs[x,y].hist(belem, bins=bins,label='belem EM RE')
	axs[x,y].hist(abs_data_em_re, bins=bins,label='average EM RE')

	axs[x,y].text(128,0.15,k.replace('_',' '),ha='center')
	axs[x,y].set_xlabel(r'index $\mu$')
	axs[x,y].set_ylabel(r'$\mbox{Tr}[\rho P_\mu] - \mbox{Tr}[\rho_{exact} P_\mu]$')

x0L,y0L,dxL,dyL = 0.00,1.05,2.22,0.20
handles,labels  = axs[0,0].get_legend_handles_labels()
lgd = axs[0,0].legend(handles,labels,ncol=5,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=2)
plt.savefig('hist_abs_em_re.pdf',format='pdf')


# ---------------------------- Histogram EM abs2

L = 5.0
fig,axs = plt.subplots(2,2,figsize=(2*L,2*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.22)

for (x,y),k in zip([(0,0),(0,1),(1,0),(1,1)],A.keys()):

	a = filter(bloch_ideal[k],4)

	m2 = filter(manila_bloch_hw_em[k],4)
	l2 = filter(lima_bloch_hw_em[k],4)
	q2 = filter(quito_bloch_hw_em[k],4)
	b2 = filter(belem_bloch_hw_em[k],4)

	abs_data_em    = (abs(m2[:,0]-a[:,0])**2 + abs(l2[:,0]-a[:,0])**2 + abs(q2[:,0]-a[:,0])**2 + abs(b2[:,0]-a[:,0])**2)/4

	manila = abs(m2[:,0]-a[:,0])**2
	lima   = abs(l2[:,0]-a[:,0])**2
	quito  = abs(q2[:,0]-a[:,0])**2
	belem  = abs(b2[:,0]-a[:,0])**2
	
	#bins = 6
	
	pa = [x for x in range(a.shape[0])]
   
	axs[x,y].hist(manila, bins=bins,label='manila EM')
	axs[x,y].hist(lima, bins=bins,label='lima EM')
	axs[x,y].hist(quito, bins=bins,label='quito EM')
	axs[x,y].hist(belem, bins=bins,label='belem EM')
	axs[x,y].hist(abs_data_em, bins=bins,label='average EM')
    	
	axs[x,y].text(128,0.15,k.replace('_',' '),ha='center')
	axs[x,y].set_xlabel(r'index $\mu$')
	axs[x,y].set_ylabel(r'$\mbox{Tr}[\rho P_\mu] - \mbox{Tr}[\rho_{exact} P_\mu]$')

x0L,y0L,dxL,dyL = 0.00,1.05,2.22,0.20
handles,labels  = axs[0,0].get_legend_handles_labels()
lgd = axs[0,0].legend(handles,labels,ncol=5,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=2)
plt.savefig('hist_abs2_em.pdf',format='pdf')


# ----------------------------------- Histogram EM RE abs

L = 5.0
fig,axs = plt.subplots(2,2,figsize=(2*L,2*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.22)

for (x,y),k in zip([(0,0),(0,1),(1,0),(1,1)],A.keys()):

	a = filter(bloch_ideal[k],4)

	m3 = filter(manila_bloch_hw_em_re[k],4)	
	l3 = filter(lima_bloch_hw_em_re[k],4)	
	q3 = filter(quito_bloch_hw_em_re[k],4)	
	b3 = filter(belem_bloch_hw_em_re[k],4)

	abs_data_em_re = (abs(m3[:,0]-a[:,0])**2 + abs(l3[:,0]-a[:,0])**2 + abs(q3[:,0]-a[:,0])**2 + abs(b3[:,0]-a[:,0])**2)/4

	manila = abs(m3[:,0]-a[:,0])**2
	lima   = abs(l3[:,0]-a[:,0])**2
	quito  = abs(q3[:,0]-a[:,0])**2
	belem  = abs(b3[:,0]-a[:,0])**2
	
	#bins = 6
	
	pa = [x for x in range(a.shape[0])]
   
	axs[x,y].hist(manila, bins=bins,label='manila EM RE')
	axs[x,y].hist(lima, bins=bins,label='lima EM RE')
	axs[x,y].hist(quito, bins=bins,label='quito EM RE')
	axs[x,y].hist(belem, bins=bins,label='belem EM RE')
	axs[x,y].hist(abs_data_em_re, bins=bins,label='average EM RE')

	axs[x,y].text(128,0.15,k.replace('_',' '),ha='center')
	axs[x,y].set_xlabel(r'index $\mu$')
	axs[x,y].set_ylabel(r'$\mbox{Tr}[\rho P_\mu] - \mbox{Tr}[\rho_{exact} P_\mu]$')

x0L,y0L,dxL,dyL = 0.00,1.05,2.22,0.20
handles,labels  = axs[0,0].get_legend_handles_labels()
lgd = axs[0,0].legend(handles,labels,ncol=5,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=2)
plt.savefig('hist_abs2_em_re.pdf',format='pdf')








################################################################################################

'''

for dir in ['manila','lima','quito','belem']:

	bloch_ideal    = np.load('tomo_statevector_bloch.npy',allow_pickle=True).item()

	bloch_qasm     = np.load('tomo_qasm_bloch.npy',allow_pickle=True).item()
	bloch_nm       = np.load('tomo_nm_bloch.npy',allow_pickle=True).item()
	bloch_nm_em    = np.load('tomo_nm_em_bloch.npy',allow_pickle=True).item()
	bloch_nm_em_re = np.load('tomo_nm_em_re_extrapolated_bloch.npy',allow_pickle=True).item()

	bloch_hw       = np.load('tomo_HW_raw_bloch.npy',allow_pickle=True).item()
	bloch_hw_em    = np.load('tomo_HW_em_bloch.npy',allow_pickle=True).item()
	bloch_hw_em_na = np.load('tomo_HW_em_re_bloch.npy',allow_pickle=True).item()
	bloch_hw_em_re = np.load('tomo_HW_em_re_extrapolated_bloch.npy',allow_pickle=True).item()
	A = np.load('../circuits.npy',allow_pickle=True).item()

	L = 5.0
	fig,axs = plt.subplots(2,2,figsize=(2*L,2*0.66*L))
	plt.subplots_adjust(wspace=0.22,hspace=0.22)

	for (x,y),k in zip([(0,0),(0,1),(1,0),(1,1)],A.keys()):
    		a1 = filter(bloch_ideal[k],4)
    
    		a2 = filter(bloch_qasm[k],4)
    		a3 = filter(bloch_nm[k],4)
    		a4 = filter(bloch_nm_em[k],4)
    		a5 = filter(bloch_nm_em_re[k],4)
    
    		a6 = filter(bloch_hw[k],4)
    		a7 = filter(bloch_hw_em[k],4)
    		a8 = filter(bloch_hw_em_re[k],4)
    
    		a9 = filter(bloch_hw_em_na[k],4)
    		pa = [x for x in range(a1.shape[0])]
    
    		axs[x,y].plot(pa,a1[:,0]-a1[:,0],label='exact',marker='o',ls='none',ms=5)
    
    		axs[x,y].errorbar(pa,a2[:,0]-a1[:,0],yerr=a2[:,1],label='qasm(id)',marker='+',ls='none')
    		axs[x,y].errorbar(pa,a3[:,0]-a1[:,0],yerr=a3[:,1],label='qasm(nm)',marker='x',ls='none')
    		axs[x,y].errorbar(pa,a4[:,0]-a1[:,0],yerr=a4[:,1],label='qasm(nm), EM',marker='1',ls='none')
    		axs[x,y].errorbar(pa,a5[:,0]-a1[:,0],yerr=a5[:,1],label='qasm(nm), EM+RE',marker='2',ms=3,ls='none')
    
    		axs[x,y].errorbar(pa,a6[:,0]-a1[:,0],yerr=a3[:,1],label='hw, RAW',marker='3',ls='none')
    		axs[x,y].errorbar(pa,a7[:,0]-a1[:,0],yerr=a4[:,1],label='hw, EM',marker='*',ls='none')
    		axs[x,y].errorbar(pa,a8[:,0]-a1[:,0],yerr=a5[:,1],label='hw, EM+RE',marker='p',ms=3,ls='none')
    
    		axs[x,y].text(128,0.15,k.replace('_',' '),ha='center')
    		axs[x,y].set_xlabel(r'index $\mu$')
    		axs[x,y].set_ylabel(r'$\mbox{Tr}[\rho P_\mu] - \mbox{Tr}[\rho_{exact} P_\mu]$')

    
	x0L,y0L,dxL,dyL = 0.00,1.05,2.22,0.20
	handles,labels  = axs[0,0].get_legend_handles_labels()
	lgd = axs[0,0].legend(handles,labels,ncol=5,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=2)
	plt.savefig(dir+'bloch_qasm_hw.pdf',format='pdf')

'''
