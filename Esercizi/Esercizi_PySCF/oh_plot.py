import numpy as np
import matplotlib.pyplot as plt

res  = np.load('results.npy',allow_pickle=True).item()
B    = res['basis']
x    = [i for i in range(len(B))]
HF_A = np.array(res['E_hf_anion'])
CI_A = np.array(res['E_casci_anion'])
HF_R = np.array(res['E_hf_radical'])
CI_R = np.array(res['E_casci_radical'])

plt.ylabel('E(radical)-E(anion) [Ha]')
plt.xlabel('basis')
plt.xticks(x,B)
plt.xticks(rotation=45)
plt.plot(x,CI_R-CI_A,label='CASCI')
plt.plot(x,HF_R-HF_A,label='HF')
plt.axhline(0,c='black',ls=':',lw=0.75)
for i in list(x):
    plt.axvline(i,c='black',ls=':',lw=0.75)
plt.title('OH radical versus OH anion')
plt.legend()
plt.show()

