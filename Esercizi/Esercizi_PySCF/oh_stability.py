import numpy as np
import pyscf
from   pyscf import gto,scf,mcscf

def get_oh_molecule(basis='6-31g',species='anion'):
    sym    = 'Coov'
    if(species=='anion'):
       R      = 0.9640
       charge = -1
       spin   = 0
       IRREP  = {'A1':6,'E1x':2,'E1y':2}
       ACTIVE = {'A1':3,'E1x':1,'E1y':1}
    else:
       R      = 0.9697
       charge = 0
       spin   = 1
       IRREP  = {'A1':(3,3),'E1x':(1,1),'E1y':(1,0)}
       ACTIVE = {'A1':3,'E1x':1,'E1y':1}
    # ----- 
    mol = gto.Mole()
    mol.build(atom     = [['O',(0,0,0)],['H',(0,0,R)]],
              basis    = basis,
              symmetry = sym,
              charge   = charge,
              spin     = spin,
              verbose  = 4)
    return mol,IRREP,ACTIVE

def simulate_oh_molecule(basis,species):
    mol,IRREP,ACTIVE = get_oh_molecule(basis,species)
    if(mol.spin==0):
       mf = scf.RHF(mol)
    else:
       mf = scf.ROHF(mol)
    mf.irrep_nelec = IRREP
    mf             = scf.newton(mf)
    E_HF           = mf.kernel()
    a = mf.stability()[0]
    E              = mf.kernel(a,mf.mo_occ)
    mf.stability()
    mf.analyze()
    # ----- 
    na = (mol.nelectron+mol.spin)//2-1
    nb = (mol.nelectron-mol.spin)//2-1
    no = 5
    mymc                  = mf.CASCI(nelecas=(na,nb),ncas=no)
    cas_space_symmetry    = ACTIVE
    mo                    = mcscf.sort_mo_by_irrep(mymc,mf.mo_coeff,cas_space_symmetry)
    mymc.fcisolver.wfnsym = 'A1'
    E_CASCI               = mymc.kernel(mo)[0]
    mymc.analyze()
    return E_HF,E_CASCI

res = {'basis'           : [],
       'E_hf_anion'      : [],
       'E_hf_radical'    : [],
       'E_casci_anion'   : [],
       'E_casci_radical' : []}

for basis in ['sto-6g',
              '6-31g','6-31++g','6-31g**','6-31++g**',
              'cc-pvdz','cc-pvtz','cc-pvqz',
              'aug-cc-pvdz','aug-cc-pvtz']:
    E_HF_anion,E_CASCI_anion     = simulate_oh_molecule(basis,'anion')
    E_HF_radical,E_CASCI_radical = simulate_oh_molecule(basis,'radical')
    res['basis'].append(basis)
    res['E_hf_anion'].append(E_HF_anion)
    res['E_hf_radical'].append(E_HF_radical)
    res['E_casci_anion'].append(E_CASCI_anion)
    res['E_casci_radical'].append(E_CASCI_radical)

np.save('results.npy',res,allow_pickle=True)

