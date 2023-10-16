import numpy as np
from   pyscf import gto,scf,ao2mo,lo
from   scipy import linalg as LA
import sys
sys.path.append('molecular_active_space/')
from subroutines import build_active_space,get_integrals,print_dictionary,do_scf,do_electronic_structure

# STO-6G < IAO < Alessandro < 6-31G

mol = gto.M(verbose=4,atom=[['O',(0,0,0)],['H',(0,0,0.964)]],charge=0,spin=1,basis='6-31g',symmetry=True)
if(mol.spin==0): mf = scf.RHF(mol)
else:            mf = scf.ROHF(mol) 
E   = mf.kernel()
iao = lo.iao.iao(mol,mf.mo_coeff[:,mf.mo_occ>0.0]) # data una molecola e un insieme di orbitali occupati a livello RHF,
                                                   # costruzione degli IAO

# ---------------------------------------------------------------------------------------------------------------------

# get Hamiltonian coefficients in the mo basis
mol_info = get_integrals(mol,mf,C=iao,n=mf.mo_occ)
print_dictionary(mol_info)

E2  = do_scf(mol_info)[0]
print("reconstruction of HF energy ",E,E2)

# construct an active space freezing/downfolding certain orbitals
# la base minimale ha 6 orbitali, con indici 0,1,2,3,4,5
# noi vogliamo eliminare 0 (che e' l'1s dell'ossigeno, quello a piu' bassa energia)
# quindi tenere [1,2,3,4,5]
# a volte puo' essere utile rimuovere gli orbitali non cilindro-simmetrici ['A1', 'A1', 'A1', 'E1x', 'E1y', 'A1']
# qui, E1x ed E1y sono ai posti 3 e 4
mol_info = build_active_space(mol_info,mol,mf,to_keep=[1,2,3,4,5])
print_dictionary(mol_info)
E3  = do_scf(mol_info)[0]
#print("reconstruction of HF energy ",E,E3)
#exit()

mol_info['n']  = mol_info['no']
mol_info['na'] = mol_info['ne'][0]
mol_info['nb'] = mol_info['ne'][1]
mol_info['E0'] = mol_info['h0']

np.save('h_dict.npy',mol_info,allow_pickle=True)

do_electronic_structure(mol_info)


