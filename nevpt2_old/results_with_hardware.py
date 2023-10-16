import numpy as np

# location of data aug-cc-pv*z/*/R_1.00_*
# cc-pvtz/cc-pvqz
# anion/radical
# statevector/qasm/nm/nm_em/nm_em_re

import matplotlib.pyplot as plt

v = {}
for b in ['aug-cc-pvtz','aug-cc-pvqz']:
    for s in ['anion','radical']:
        for p in ['statevector','qasm','nm','nm_em','nm_em_re']:
            f = open('%s/%s/R_1.00_%s/results.txt' % (b,s,p),'r')
            f = [x.split() for x in f.readlines()]
            f = f[len(f)-1]
            ave = float(f[9])
            std = float(f[11])
            v['%s/%s/%s' % (b,s,p)] = [ave,std]

L = 8.0
fig,ax = plt.subplots(1,1,figsize=(1*L,1*0.66*L))
ax1 = ax

kcalmol = 627.5
for p in ['statevector','qasm','nm','nm_em','nm_em_re']: 
    x   = [3,4]
    y   = [v['%s/radical/%s' % (b,p)][0]-v['%s/anion/%s' % (b,p)][0]                for b in ['aug-cc-pvtz','aug-cc-pvqz']]
    dy  = [np.sqrt(v['%s/radical/%s' % (b,p)][1]**2+v['%s/anion/%s' % (b,p)][1]**2) for b in ['aug-cc-pvtz','aug-cc-pvqz']]
    (_, caps, _) = ax1.errorbar(x,kcalmol*np.array(y),yerr=kcalmol*np.array(dy),marker='o',capsize=5, elinewidth=3,label=p)
    for cap in caps:
        cap.set_markeredgewidth(3)

plt.legend()



plt.show()



