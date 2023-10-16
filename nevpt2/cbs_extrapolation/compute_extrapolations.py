import numpy as np
import scipy.optimize as opt
import sys
sys.path.append('../src/')
from plot import make_empty_elist, find_minimum, fill_panel, c_list, kcal, morse
from plot import append_energies_cbs_f, compute_zpe, extrapolate, reduced_mass

import numpy as np
import matplotlib.pyplot as plt
from   scipy.optimize import curve_fit


# ---------------------------------------

R_list    = ['0.80','0.82','0.84','0.86','0.88','0.90','0.92','0.94','0.96','0.98','1.00','1.02','1.04','1.06','1.08','1.10','1.12','1.14','1.16','1.18','1.20']
nR        = len(R_list)

for flavor in ['nevpt2(fci,fci)','nevpt2(fci,qse)','nevpt2(qUCCSD,qse)','nevpt2(Ry,qse)']:

    E_anion_scf = np.zeros((nR,4))
    E_anion_pt2 = np.zeros((nR,4))
    E_rad_scf   = np.zeros((nR,4))
    E_rad_pt2   = np.zeros((nR,4))
    
    for x, basis in zip([2,3,4],['aug-cc-pvdz','aug-cc-pvtz','aug-cc-pvqz']):
        E_anion   = make_empty_elist()
        E_radical = make_empty_elist()
        for R in R_list:
            E_anion   = append_energies_cbs_f('anion',basis,R,E_anion)
            E_radical = append_energies_cbs_f('radical',basis,R,E_radical) 
        E_anion_scf[:,x-2] = np.array(E_anion['full']['scf'])
        E_anion_pt2[:,x-2] = np.array(E_anion['iao'][flavor])
        E_rad_scf[:,x-2] = np.array(E_radical['full']['scf'])
        E_rad_pt2[:,x-2] = np.array(E_radical['iao'][flavor])
 
    for jR,R in enumerate(R_list):
        res_anion = extrapolate([2,3,4],E_anion_scf[jR,:3],[3,4],E_anion_pt2[jR,1:3]-E_anion_scf[jR,1:3])
        res_rad   = extrapolate([2,3,4],E_rad_scf[jR,:3],[3,4],E_rad_pt2[jR,1:3]-E_rad_scf[jR,1:3])
        E_anion_scf[jR,3] = res_anion['E_hf_fit'][0]
        E_anion_pt2[jR,3] = res_anion['E_hf_fit'][0]+res_anion['E_c_fit'][0]
        E_rad_scf[jR,3] = res_rad['E_hf_fit'][0]
        E_rad_pt2[jR,3] = res_rad['E_hf_fit'][0]+res_rad['E_c_fit'][0]

    Ehf_A = E_anion_scf[:,3]
    Ept_A = E_anion_pt2[:,3]
    Ehf_R = E_rad_scf[:,3]
    Ept_R = E_rad_pt2[:,3]
    x     = [float(R) for R in R_list]

    x0,y0,dx0,dy0 = find_minimum(x,Ept_A)
    x1,y1,dx1,dy1 = find_minimum(x,Ept_R)
    print("NEVPT2, %s & %.5f(%d) & %.5f(%d) & %.3f(%d) \\\\" % (flavor,x0,int(dx0*1e5),x1,int(dx1*1e5),kcal*(y1-y0),int(1e3*kcal*np.sqrt(dy0**2+dy1**2))))
    x0,y0,dx0,dy0 = find_minimum(x,Ehf_A)
    x1,y1,dx1,dy1 = find_minimum(x,Ehf_R)
    print("SCF,       & %.5f(%d) & %.5f(%d) & %.3f(%d) \\\\" % (       x0,int(dx0*1e5),x1,int(dx1*1e5),kcal*(y1-y0),int(1e3*kcal*np.sqrt(dy0**2+dy1**2))))

# --------------------------------------------------------------------------------------------

for col,method in zip([2,3,4,5],['MP2','CCSD','CCSD(T)','CASSCF']):
    E_anion_scf = np.zeros((nR,4))
    E_anion_met = np.zeros((nR,4))
    E_rad_scf   = np.zeros((nR,4))
    E_rad_met   = np.zeros((nR,4))
    V = np.loadtxt('../aug_cc-pvxz_classical/E_anion_aug-cc-pvxz.txt')
    W = np.loadtxt('../aug_cc-pvxz_classical/E_radical_aug-cc-pvxz.txt')
    m = 0
    for jR,R in enumerate(R_list):
        for x,b in zip([2,3,4],['aug-cc-pvdz','aug-cc-pvtz','aug-cc-pvqz']):
            E_anion_scf[jR,x-2] = V[m,1]
            E_anion_met[jR,x-2] = V[m,col]
            E_rad_scf[jR,x-2]   = W[m,1]
            E_rad_met[jR,x-2]   = W[m,col]
            m += 1
        res_anion = extrapolate([2,3,4],E_anion_scf[jR,:3],[3,4],E_anion_met[jR,1:3])
        res_rad   = extrapolate([2,3,4],E_rad_scf[jR,:3],[3,4],E_rad_met[jR,1:3])
        E_anion_scf[jR,3] = res_anion['E_hf_fit'][0]
        E_anion_met[jR,3] = res_anion['E_hf_fit'][0]+res_anion['E_c_fit'][0]
        E_rad_scf[jR,3] = res_rad['E_hf_fit'][0]
        E_rad_met[jR,3] = res_rad['E_hf_fit'][0]+res_rad['E_c_fit'][0]

    Ehf_A = E_anion_scf[:,3]
    Emt_A = E_anion_met[:,3]
    Ehf_R = E_rad_scf[:,3]
    Emt_R = E_rad_met[:,3]
    x     = [float(R) for R in R_list]

    x0,y0,dx0,dy0 = find_minimum(x,Emt_A)
    x1,y1,dx1,dy1 = find_minimum(x,Emt_R)
    print("%s & %.5f(%d) & %.5f(%d) & %.3f(%d) \\\\" % (method,x0,int(dx0*1e5),x1,int(dx1*1e5),kcal*(y1-y0),int(1e3*kcal*np.sqrt(dy0**2+dy1**2))))
    x0,y0,dx0,dy0 = find_minimum(x,Ehf_A)
    x1,y1,dx1,dy1 = find_minimum(x,Ehf_R)






