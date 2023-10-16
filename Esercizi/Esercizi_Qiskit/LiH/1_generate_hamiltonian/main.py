import numpy as np
from   pyscf import gto,scf,ao2mo
from   scipy import linalg as LA
import sys
sys.path.append('../../molecular_active_space/')
from subroutines import build_active_space,get_integrals,print_dictionary,do_scf

mol = gto.M(verbose=4,atom=[['Li',(0,0,0)],['H',(0,0,1.593)]],basis='sto-6g',symmetry=True)
mf  = scf.RHF(mol)
E   = mf.kernel()

# get Hamiltonian coefficients in the mo basis
mol_info = get_integrals(mol,mf,C=mf.mo_coeff,n=mf.mo_occ)
print_dictionary(mol_info)

E2  = do_scf(mol_info)
print("reconstruction of HF energy ",E,E2)

# construct an active space freezing/downfolding certain orbitals
mol_info = build_active_space(mol_info,mol,mf,to_keep=[1,2,5])
print_dictionary(mol_info)
E3  = do_scf(mol_info)
print("reconstruction of HF energy ",E,E3)

mol_info['n']  = mol_info['no']
mol_info['na'] = mol_info['ne'][0]
mol_info['nb'] = mol_info['ne'][1]
mol_info['E0'] = mol_info['h0']

np.save('h_dict.npy',mol_info,allow_pickle=True)

