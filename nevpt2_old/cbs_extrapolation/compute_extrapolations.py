import numpy as np
import scipy.optimize as opt
import sys
sys.path.append('../')

kcal   = 627.509
c_list = {'purple'       : '#B284BE',
          'jacaranda'    : '#888FC7',
          'light_yellow' : '#EEEA62',
          'gold'         : '#FFD300',
          'earwax'       : '#DAA520',
          'brown'        : '#7B3F00',
          'light_blue'   : '#bcd9ea',
          'cerulean'     : '#5ba4cf',
          'cobalt'       : '#0079bf',
          'dark_blue'    : '#055a8c',
          'light_green'  : '#acdf87',
          'yellow_green' : '#B9D146',
          'mid_green'    : '#68bb59',
          'dark_green'   : '#1e5631',
          'orange'       : '#FF8856',
          'red'          : '#DC343B',
          'light-gray'   : '#C0C0C0',
          'palatinate'   : '#72246C'}

def make_empty_elist():
    E_dict = {'iao'  : {'Ry':[],'qUCCSD':[],'nevpt2(fci,fci)':[],'nevpt2(fci,qse)':[],'nevpt2(qUCCSD,qse)':[],'nevpt2(Ry,qse)':[],'fci':[]},
              'full' : {'scf':[],'fci':[]}}
    return E_dict

def morse(x,Re,E0,De,ke):
    return E0+De*( 1-np.exp(-np.sqrt(ke/2.0/De)*(x-Re)) )**2

def find_minimum(x,y):
    x,y = np.array(x),np.array(y)
    p0 = [x[np.argmin(y)],np.min(y),0.1,0.1] 
    fit_params,pcov = opt.curve_fit(morse,x,y,p0=p0)
    return fit_params[0],fit_params[1],np.sqrt(pcov[0,0]),np.sqrt(pcov[1,1])

def fill_panel(pan,xlabel,xlim,xticks,xticklabels,ylabel,ylim,yticks,yticklabels,p=20.0,q=20.0):
    x0,x1 = xlim
    xlim  = [x0-(x1-x0)/p,x1+(x1-x0)/p]
    pan.set_xlabel(xlabel)
    pan.set_xlim(xlim)
    pan.set_xticks(xticks)
    pan.set_xticklabels(xticklabels)
    pan.set_ylabel(ylabel)
    y0,y1 = ylim
    ylim  = [y0-(y1-y0)/q,y1+(y1-y0)/q]
    pan.set_ylim(ylim)
    pan.set_yticks(yticks)
    pan.set_yticklabels(yticklabels)
    pan.tick_params(direction='in',which='both')

def append_energies(species,basis,R,E_list):
    f = open('../%s/%s/R_%s/results.txt'%(basis,species,R),'r')
    f = f.readlines()
    f = [x.split() for x in f]
    E_list['full']['scf'].append(float(f[0][5]))
    E_list['iao']['fci'].append(float(f[7][5]))
    E_list['full']['fci'].append(float(f[9][5]))
    E_list['iao']['Ry'].append(float(f[15][3]))
    E_list['iao']['qUCCSD'].append(float(f[13][2]))
    E_list['iao']['nevpt2(fci,fci)'].append(float(f[10][5]))
    E_list['iao']['nevpt2(fci,qse)'].append(float(f[11][5]))
    E_list['iao']['nevpt2(qUCCSD,qse)'].append(float(f[12][5]))
    E_list['iao']['nevpt2(Ry,qse)'].append(float(f[14][5]))
    return E_list

import numpy as np
import matplotlib.pyplot as plt
from   scipy.optimize import curve_fit

def expo(x,a,b,c):
    return a+b*np.exp(-c*x)

def corr(x,a,b):
    return a+b*x

def extrapolate(xhf,Ehf,xc,Ec):
    p0 = [Ehf[len(Ehf)-1]-0.1,(Ehf[0]-Ehf[len(Ehf)-1]),0.1]
    p_ave,p_cov = curve_fit(expo,xhf,Ehf,p0)
    p0 = [Ec[len(Ec)-1],1]
    q_ave,q_cov = curve_fit(corr,[1.0/xi**3 for xi in xc],Ec,p0)
    return {'E_hf_fit':p_ave,'E_hf_err':p_cov,'E_c_fit':q_ave,'E_c_err':q_cov}

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
            E_anion   = append_energies('anion',basis,R,E_anion)
            E_radical = append_energies('radical',basis,R,E_radical) 
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

for col,method in zip([2,3],['MP2','CCSD']):
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






