import numpy as np
import scipy.optimize as opt
import sys
sys.path.append('../../nevpt2/src/')
from plot import make_empty_elist, find_minimum, fill_panel, c_list, kcal
from plot import morse_cbs, append_energies_cbs, compute_zpe, extrapolate, reduced_mass

import matplotlib.pyplot as plt

# ---------------------------------------

R_list    = ['0.80','0.82','0.84','0.86','0.88','0.90','0.92','0.94','0.96','0.98','1.00','1.02','1.04','1.06','1.08','1.10','1.12','1.14','1.16','1.18','1.20']
nR        = len(R_list)
E_anion_scf = np.zeros((nR,4)) # four components for dx,tz,qz,CBS
E_anion_pt2 = np.zeros((nR,4))
E_rad_scf   = np.zeros((nR,4))
E_rad_pt2   = np.zeros((nR,4))

for x, basis in zip([2,3,4],['aug-cc-pvdz','aug-cc-pvtz','aug-cc-pvqz']):
    E_anion   = make_empty_elist()
    E_radical = make_empty_elist()
    for R in R_list:
        E_anion   = append_energies_cbs('anion',basis,R,E_anion)
        E_radical = append_energies_cbs('radical',basis,R,E_radical) 
    E_anion_scf[:,x-2] = np.array(E_anion['full']['scf'])
    E_anion_pt2[:,x-2] = np.array(E_anion['iao']['nevpt2(Ry,qse)'])
    E_rad_scf[:,x-2] = np.array(E_radical['full']['scf'])
    E_rad_pt2[:,x-2] = np.array(E_radical['iao']['nevpt2(Ry,qse)'])

for jR,R in enumerate(R_list):
    res_anion = extrapolate([2,3,4],E_anion_scf[jR,:3],[3,4],E_anion_pt2[jR,1:3]-E_anion_scf[jR,1:3])
    res_rad   = extrapolate([2,3,4],E_rad_scf[jR,:3],[3,4],E_rad_pt2[jR,1:3]-E_rad_scf[jR,1:3])
    E_anion_scf[jR,3] = res_anion['E_hf_fit'][0]
    E_anion_pt2[jR,3] = res_anion['E_hf_fit'][0]+res_anion['E_c_fit'][0]
    E_rad_scf[jR,3] = res_rad['E_hf_fit'][0]
    E_rad_pt2[jR,3] = res_rad['E_hf_fit'][0]+res_rad['E_c_fit'][0]

R_list = [float(R) for R in R_list]

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['cmu serif'],'size':12})
rc('text', usetex=True)

L = 5.0
fig,ax = plt.subplots(2,2,figsize=(2*L,2*0.66*L))
plt.subplots_adjust(wspace=0.22,hspace=0.0)
ax1=ax[0,0]
ax2=ax[1,0]
ax3=ax[0,1]
ax4=ax[1,1]

outfile = open('results.txt','w')
outfile.write("basis method Req(anion) uncertainty Eeq(radical) uncertainty dE(Ha) error (Ha) \n")

m = 0
ms = 7
for x,Ehf_A,Ept_A,Ehf_R,Ept_R,b,c,ls in zip([R_list,R_list,R_list,R_list],
                                          [E_anion_scf[:,0],E_anion_scf[:,1],E_anion_scf[:,2],E_anion_scf[:,3]],
                                          [E_anion_pt2[:,0],E_anion_pt2[:,1],E_anion_pt2[:,2],E_anion_pt2[:,3]],
                                          [E_rad_scf[:,0],E_rad_scf[:,1],E_rad_scf[:,2],E_rad_scf[:,3]],
                                          [E_rad_pt2[:,0],E_rad_pt2[:,1],E_rad_pt2[:,2],E_rad_pt2[:,3]],
                                          ['aug-cc-pvdz','aug-cc-pvtz','aug-cc-pvqz','CBS'],
                                          [c_list['jacaranda'],c_list['purple'],c_list['palatinate'],c_list['red']],
                                          [':','-.','--','-']):
    ax1.plot(x,Ehf_A,c=c,ls=ls)
    ax2.plot(x,Ept_A,c=c,ls=ls)
    ax3.plot(x,Ehf_R,c=c,ls=ls)
    ax4.plot(x,Ept_R,c=c,ls=ls)
    x0,y0,dx0,dy0 = find_minimum(x,Ept_A); ax2.scatter(x0,y0,marker='X',c=c,edgecolor='black',linewidth=0.5,s=6*ms)
    x1,y1,dx1,dy1 = find_minimum(x,Ept_R); ax4.scatter(x1,y1,marker='X',c=c,edgecolor='black',linewidth=0.5,s=6*ms)
    ax1.plot(x,[np.nan]*len(Ehf_A),label=b,c=c,ls=ls,marker='X',mec='black',mew=0.5,ms=ms)
    outfile.write("%s NEVPT2, & %.5f(%d) & %.5f(%d) & %.3f(%d) \\\\ \n" % (b,x0,int(dx0*1e5),x1,int(dx1*1e5),kcal*(y1-y0),int(1e3*kcal*np.sqrt(dy0**2+dy1**2))))
    outfile.write("NEVPT2, ZPE (anion)   = %.12f \n" % compute_zpe(x,Ept_A,reduced_mass))
    outfile.write("NEVPT2, ZPE (radical) = %.12f \n" % compute_zpe(x,Ept_R,reduced_mass))

    x0,y0,dx0,dy0 = find_minimum(x,Ehf_A); ax1.scatter(x0,y0,marker='X',c=c,edgecolor='black',linewidth=0.5,s=6*ms)
    x1,y1,dx1,dy1 = find_minimum(x,Ehf_R); ax3.scatter(x1,y1,marker='X',c=c,edgecolor='black',linewidth=0.5,s=6*ms)
    outfile.write("%s SCF, & %.5f(%d) & %.5f(%d) & %.3f(%d) \\\\ \n" % (b,x0,int(dx0*1e5),x1,int(dx1*1e5),kcal*(y1-y0),int(1e3*kcal*np.sqrt(dy0**2+dy1**2))))

    m += 1

fill_panel(ax1,'$R_{\mathrm{OH}} [\mathrm{\AA}]$',[0.8,1.2],[0.8,0.9,1.0,1.1,1.2],['0.8','0.9','1.0','1.1','1.2'],
               '$E_{\mathrm{anion,SCF}} [E_h]$',[-75.44,-75.36],[-75.44,-75.42,-75.40,-75.38,-75.36],['-75.44','-75.42','-75.40','-75.38','-75.36'])

fill_panel(ax3,'$R_{\mathrm{OH}} [\mathrm{\AA}]$',[0.8,1.2],[0.8,0.9,1.0,1.1,1.2],['0.8','0.9','1.0','1.1','1.2'],
               '$E_{\mathrm{radical,SCF}} [E_h]$',[-75.44,-75.36],[-75.44,-75.42,-75.40,-75.38,-75.36],['-75.44','-75.42','-75.40','-75.38','-75.36'])

fill_panel(ax2,'$R_{\mathrm{OH}} [\mathrm{\AA}]$',[0.8,1.2],[0.8,0.9,1.0,1.1,1.2],['0.8','0.9','1.0','1.1','1.2'],
               '$E_{\mathrm{anion,NEVPT2(R_y,QSE)}} [E_h]$',[-75.60,-75.40],[-75.60,-75.55,-75.50,-75.45,-75.40],['-75.60','-75.55','-75.50','-75.45','-75.40'])

fill_panel(ax4,'$R_{\mathrm{OH}} [\mathrm{\AA}]$',[0.8,1.2],[0.8,0.9,1.0,1.1,1.2],['0.8','0.9','1.0','1.1','1.2'],
               '$E_{\mathrm{radical,NEVPT2(R_y,QSE)}} [E_h]$',[-75.60,-75.40],[-75.60,-75.55,-75.50,-75.45,-75.40],['-75.60','-75.55','-75.50','-75.45','-75.40'])

x0L,y0L,dxL,dyL = 0.00,1.05,2.22,0.20
handles,labels  = ax1.get_legend_handles_labels()
lgd = ax1.legend(handles,labels,ncol=5,fancybox=True,shadow=True,
                 bbox_to_anchor=(x0L,y0L,dxL,dyL),loc=3,mode='expand',
                 borderaxespad=0,handlelength=3)

fig.savefig('fig.pdf',format='pdf',bbox_inches='tight')

