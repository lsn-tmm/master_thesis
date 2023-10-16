import numpy as np

# location of data R_1.00_*_em_re
# e-1 e-2 e-3 e-4

import matplotlib.pyplot as plt

v = {}
hardware = ['bogota', 'manila']
for hw in hardware:
    for p in ['e-1','e-2','e-3','e-4']:
        f = open('R_1.00_%s_em_re/%s/results.txt' % (hw,p),'r')
        f = [x.split() for x in f.readlines()]
        f = f[len(f)-1]
        ave = float(f[9])
        std = float(f[11])
        v['%s_%s' % (hw,p)] = [ave,std]

print(v.keys())                
#exit()

kcalmol = 627.5


L = 8.0
fig,ax = plt.subplots(1,2,figsize=(1*L,1*0.66*L))
ax1 = ax[0]
ax2 = ax[1]
x = [0,1,2,3]
for hw in hardware:
    y   = [v['%s_%s' % (hw,p)][0]    for p in ['e-1','e-2','e-3','e-4']]
    dy  = [v['%s_%s' % (hw,p)][1]    for p in ['e-1','e-2','e-3','e-4']]
    #(_, caps, _) = ax1.errorbar(x,np.array(y),np.array(dy),marker='o',capsize=5, elinewidth=3,label=hw)
    ax1.plot(x,np.array(y),marker='o',label=hw)
    ax2.plot(x,np.array(dy),marker='o',label=hw)

ax1.set_xticks([0,1,2,3])
ax1.set_xticklabels(['e-1','e-2','e-3','e-4'])
ax2.set_xticks([0,1,2,3])
ax2.set_xticklabels(['e-1','e-2','e-3','e-4'])
ax2.set_yscale("log")
ax1.legend()
fig.savefig('reg2.pdf',format='pdf',bbox_inches='tight')
