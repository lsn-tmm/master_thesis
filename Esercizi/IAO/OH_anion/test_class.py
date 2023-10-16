import numpy as np
from   pyscf import gto,scf,ao2mo,lo
from   scipy import linalg as LA
import sys
sys.path.append('../source/')
from BO_class import Born_Oppenheimer

# STO-6G < IAO < Alessandro < 6-31G

mol = gto.M(verbose=4,atom=[['O',(0,0,0)],['H',(0,0,0.964)]],charge=-1,spin=0,basis='6-31++g**',symmetry=True)
if(mol.spin==0): mf = scf.RHF(mol)
else:            mf = scf.ROHF(mol) 
E   = mf.kernel()
iao = lo.iao.iao(mol,mf.mo_coeff[:,mf.mo_occ>0.0]) # data una molecola e un insieme di orbitali occupati a livello 
                                                   # RHF costruzione degli IAO

# ---------------------------------------------------------------------------------------------------------------------

# get Hamiltonian coefficients in the mo basis
#mol_info = get_integrals(mol,mf,C=iao,n=mf.mo_occ)
#print_dictionary(mol_info)
BO = Born_Oppenheimer(mol,mf,C=iao,n=mf.mo_occ)


E2 = BO.do_scf()[0]
BO.print_mol_data()
print("reconstruction of HF energy ",E,E2)


# construct an active space freezing/downfolding certain orbitals
# la base minimale ha 6 orbitali, con indici 0,1,2,3,4,5
# noi vogliamo eliminare 0 (che e' l'1s dell'ossigeno, quello a piu' bassa energia)
# quindi tenere [1,2,3,4,5]
# a volte puo' essere utile rimuovere gli orbitali non cilindro-simmetrici ['A1', 'A1', 'A1', 'E1x', 'E1y', 'A1']
# qui, E1x ed E1y sono ai posti 3 e 4

BO.build_active_space(to_keep=[1,2,3,4,5])
BO.print_mol_data()
E3 = BO.do_scf()[0]
print("reconstruction of HF energy ",E,E3)


BO.do_electronic_structure()
