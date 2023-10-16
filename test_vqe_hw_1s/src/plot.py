import numpy as np
import scipy.optimize as opt
from   scipy.optimize import curve_fit

amu = 1822.89
mh  = 1.00784
mo  = 15.999
reduced_mass = mh*mo/(mh+mo)*amu

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

def morse_cbs(x,Re,E0,De,Ae):
    return E0+De*( 1-np.exp(-Ae*(x-Re)) )**2

def compute_zpe(x,y,reduced_mass):
    x,y = np.array(x),np.array(y)
    p0 = [x[np.argmin(y)],np.min(y),1.0,1.0]
    fit_params,pcov = opt.curve_fit(morse,x,y,p0=p0)
    abar = fit_params[3]
    dbar = fit_params[2]
    hnu0 = abar/(2.0*np.pi*1.88973)*np.sqrt(2*dbar/reduced_mass)
    return hnu0*(1.0/2.0)-(hnu0**2)/(4.0*dbar)*(1.0/2.0)**2

def find_minimum(x,y):
    x,y = np.array(x),np.array(y)
    p0 = [x[np.argmin(y)],np.min(y),0.1,0.1] 
    fit_params,pcov = opt.curve_fit(morse,x,y,p0=p0, maxfev=5000)
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
    
def append_energies(species,R,E_list, path='./'):
    f = open('%s/%s/R_%s/results.txt'%(path,species,R),'r')
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

def append_energies_f(species,R,E_list):
    f = open('%s/R_%s/results.txt'%(species,R),'r')
    f = f.readlines()
    f = [x.split() for x in f]
    E_list['iao']['nevpt2(qUCCSD,qse)'].append(float(f[1][5])) 
    E_list['iao']['nevpt2(Ry,qse)'].append(float(f[3][5])) 
    return E_list

def append_energies_cbs(species,basis,R,E_list):
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