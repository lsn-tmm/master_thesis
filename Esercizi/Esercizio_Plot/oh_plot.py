import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(15,8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

#---------------------------------NOT IAO-----------------------------------------
res  = np.load('oh_data.npy',allow_pickle=True).item()
B    = res['basis']
x    = [i for i in range(len(B))]

HF_A = np.array(res['E_HF_anion'])
MO_A = np.array(res['E_MP_anion'])
CISD_A = np.array(res['E_CISD_anion'])
CCSD_A = np.array(res['E_CCSD_anion'])
CCSDT_A = np.array(res['E_CCSD(T)_anion'])
CASCI_A = np.array(res['E_CASCI_anion'])
CASSCF_A = np.array(res['E_CASSCF_anion'])

HF_R = np.array(res['E_HF_radical'])
MO_R = np.array(res['E_MP_radical'])
CISD_R = np.array(res['E_CISD_radical'])
CCSD_R = np.array(res['E_CCSD_radical'])
CCSDT_R = np.array(res['E_CCSD(T)_radical'])
CASCI_R = np.array(res['E_CASCI_radical'])
CASSCF_R = np.array(res['E_CASSCF_radical'])



ax1.set_ylabel('E(radical)-E(anion) [Ha]')
ax1.set_xlabel('basis')
ax1.set_xticks(x)
ax1.set_xticklabels(B, rotation=45)
ax1.plot(x,HF_R-HF_A,label='HF')
ax1.plot(x,MO_R-MO_A,label='MP')
ax1.plot(x,CISD_R-CISD_A,label='CISD')
ax1.plot(x,CCSD_R-CCSD_A,label='CCSD')
ax1.plot(x,CCSDT_R-CCSDT_A,label='CCSD(T)')
ax1.plot(x,CASCI_R-CASCI_A,label='CASCI')
ax1.plot(x,CASSCF_R-CASSCF_A,label='CASSCF')


ax1.axhline(0,c='black',ls=':',lw=0.75)
for i in list(x):
    ax1.axvline(i,c='black',ls=':',lw=0.75)
    
ax1.set_title('Original set')
    
#------------------------------------IAO---------------------------- 
res  = np.load('iao_oh_data.npy',allow_pickle=True).item()
for i in range(len(res['basis'])):
    res['basis'][i] = 'IAO/'+res['basis'][i]
B = res['basis']
x    = [i for i in range(len(B))]

HF_A = np.array(res['E_HF_anion'])
MO_A = np.array(res['E_MP_anion'])
CISD_A = np.array(res['E_CISD_anion'])
CCSD_A = np.array(res['E_CCSD_anion'])
CCSDT_A = np.array(res['E_CCSD(T)_anion'])
CASCI_A = np.array(res['E_CASCI_anion'])
CASSCF_A = np.array(res['E_CASSCF_anion'])

HF_R = np.array(res['E_HF_radical'])
MO_R = np.array(res['E_MP_radical'])
CISD_R = np.array(res['E_CISD_radical'])
CCSD_R = np.array(res['E_CCSD_radical'])
CCSDT_R = np.array(res['E_CCSD(T)_radical'])
CASCI_R = np.array(res['E_CASCI_radical'])
CASSCF_R = np.array(res['E_CASSCF_radical'])

ax2.set_ylabel('E(radical)-E(anion) [Ha]')
ax2.set_xlabel('basis')
ax2.set_xticks(x)
ax2.set_xticklabels(B, rotation=45)
ax2.plot(x,HF_R-HF_A,label='HF')
ax2.plot(x,MO_R-MO_A,label='MP')
ax2.plot(x,CISD_R-CISD_A,label='CISD')
ax2.plot(x,CCSD_R-CCSD_A,label='CCSD')
ax2.plot(x,CCSDT_R-CCSDT_A,label='CCSD(T)')
ax2.plot(x,CASCI_R-CASCI_A,label='CASCI')
ax2.plot(x,CASSCF_R-CASSCF_A,label='CASSCF')


ax2.axhline(0,c='black',ls=':',lw=0.75)
for i in list(x):
    ax2.axvline(i,c='black',ls=':',lw=0.75)
    
ax2.set_title('IAO')
    
ax1.legend()
ax2.legend()

ax1.set_ylim(-0.07, 0.07)
ax2.set_ylim(-0.07, 0.07)
fig.suptitle('OH radical versus OH anion')
plt.show()

