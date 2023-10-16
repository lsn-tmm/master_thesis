from pyscf       import gto,scf
from subroutines import *

mol = gto.M(atom=[['O',(0.0000,0.0000,0.0000)],
                  ['H',(1.0000,0.0000,0.0000)],
                  ['H',(0.0000,1.0000,0.0000)]],
            charge=0,spin=0,basis='6-31g',symmetry=True,verbose=0)
mf    = scf.RHF(mol)
Ehf_1 = mf.kernel()

# get Hamiltonian coefficients in the mo basis
mol_info = get_integrals(mol,mf,C=mf.mo_coeff,n=mf.mo_occ)
print_dictionary(mol_info)

# construct an active space freezing/downfolding certain orbitals
mol_info = build_active_space(mol_info,mol,mf,to_keep=[4,5])
print_dictionary(mol_info,verbose=True)
dump_on_file(mol_info,fname='active_space.npy')

Ehf_2    = do_scf(mol_info)
print("reconstruction of HF energy ",Ehf_1,Ehf_2)

mol_info_reconstructed = read_from_file(fname='active_space.npy')
Ehf_3    = do_scf(mol_info_reconstructed)
print("reconstruction of HF energy ",Ehf_1,Ehf_2,Ehf_3)


