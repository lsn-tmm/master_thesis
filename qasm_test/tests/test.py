import numpy as np
import sys
sys.path.append('../src/')
from basis_constructor import Basis_Constructor
from bo_class import BO_class
from bo_solver import BO_solver
from fci_qse_solver import FCI_QSE_solver
from fci_solver import FCI_solver
from vqe_qse_solver import VQE_QSE_solver
from pyscf import gto,scf
from VQE_types import vqe_data

# 1. si definisce una molecola (in questo caso OH-) e si fa un conto di campo medio (RHF/ROHF)
mol = gto.M(atom=[['O',(0,0,0)],['H',(0,0,1)]],basis='6-31++g**',spin=1,charge=0,
            symmetry=True,verbose=4).build()
mf  = scf.ROHF(mol)
mf  = scf.newton(mf)
mf.kernel()
if(not mf.converged):
   mf.kernel(mf.make_rdm1())
mf.analyze()

# 2. si costruisce una base di orbitali core, valenza, esterni
BC_object = Basis_Constructor(mol,mf)
BC_object.compute_core_valence()
BC_object.compute_external()

# 3. si costruisce un oggetto BO_class che contenga l'Hamiltoniano proiettato nell'insieme degli orbitali core+valence,
#    congelamento degli orbitali di core 
BO_IAO = BO_class(mol,mf)
BO_IAO.transform_integrals(BC_object.valence)
BO_IAO = BO_IAO.freeze_orbitals([0,1])
BO_IAO_solver = BO_solver(BO_IAO)
print("IAO, SCF ",BO_IAO_solver.solve_with_scf()[1])
print("IAO, FCI ",BO_IAO_solver.solve_with_fci()[0])

# 4. si costruisce un oggetto BO_class che contenga l'Hamiltoniano della molecola, nell'intera base (core+valence+external)
BO_IAO_external = BO_class(mol,mf)
BO_IAO_external.transform_integrals(BC_object.return_basis())
BO_IAO_external = BO_IAO_external.freeze_orbitals([0])
BO_IAO_external_solver = BO_solver(BO_IAO_external)
print("FULL, SCF ",BO_IAO_external_solver.solve_with_scf()[1])
#print("FULL, FCI ",BO_IAO_external_solver.solve_with_fci()[0])

#exit()

# 5. partendo da BO_IAO (Hamiltoniano core+valence) e BO_IAO_external (Hamiltoniano core+valence+external) implementare NEVPT2
#    secondo lo schema "tutto esatto"
SOLVER = FCI_solver(BO_IAO,BO_IAO_external)
print("FCI+FCI NEVPT2 energy ",SOLVER.compute_nevpt2_energy())
#    secondo lo schema "ground state esatto, stati eccitati approssimati con QSE"
SOLVER = FCI_QSE_solver(BO_IAO,BO_IAO_external)
print("FCI+QSE NEVPT2 energy ", SOLVER.compute_nevpt2_energy())
#    secondo lo schema "ground state con VQE, stati eccitati con QSE"
print("="*53)

VQE_settings = vqe_data() #Default ---------- 'target_sector':None,'optimizer':'bfgs','max_iter':1000,'instance':'statevector_simulator','shots':1000, 'ansatz':'q_uccsd','initial_point':None
if(mol.spin!=0): VQE_settings.set_target_sector(2)

SOLVER = VQE_QSE_solver(BO_IAO,BO_IAO_external)
SOLVER.set_quantum_variables(VQE_settings)
print("VQE+QSE NEVPT2 energy ", SOLVER.compute_nevpt2_energy())

