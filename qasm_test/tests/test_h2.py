import numpy as np
import sys
sys.path.append('../../nevpt2/src/')
from basis_constructor import Basis_Constructor
from bo_class import BO_class
from bo_solver import BO_solver
from fci_qse_solver import FCI_QSE_solver
from fci_solver import FCI_solver
from vqe_qse_solver import VQE_QSE_solver
from pyscf import gto,scf,cc,mp
from VQE_types import vqe_data

# 1. si definisce una molecola (in questo caso OH-) e si fa un conto di campo medio (RHF/ROHF)
mol = gto.M(atom=[['H',(0,0,0)],['H',(0,0,0.75)]],basis='6-31g',spin=0,charge=0,
            symmetry=True,verbose=0).build()
mf  = scf.ROHF(mol)
mf  = scf.newton(mf)
Ehf = mf.kernel()
if(not mf.converged):
   Ehf = mf.kernel(mf.make_rdm1())
mf.analyze()

mc  = cc.CCSD(mf)
Ec  = mc.kernel()[0]
mm  = mp.MP2(mf)
Em  = mm.kernel()[0]

# 2. si costruisce una base di orbitali core, valenza, esterni
BC_object = Basis_Constructor(mol,mf)
BC_object.compute_core_valence()
BC_object.compute_external()

# 3. si costruisce un oggetto BO_class che contenga l'Hamiltoniano proiettato nell'insieme degli orbitali core+valence,
#    congelamento degli orbitali di core 
BO_IAO = BO_class(mol,mf)
BO_IAO.transform_integrals(BC_object.valence)
BO_IAO = BO_IAO.freeze_orbitals([])

# 4. si costruisce un oggetto BO_class che contenga l'Hamiltoniano della molecola, nell'intera base (core+valence+external)
BO_IAO_external = BO_class(mol,mf)
BO_IAO_external.transform_integrals(BC_object.return_basis())
BO_IAO_external = BO_IAO_external.freeze_orbitals([])

# 5. partendo da BO_IAO (Hamiltoniano core+valence) e BO_IAO_external (Hamiltoniano core+valence+external) implementare NEVPT2
#    secondo lo schema "tutto esatto"
print("*************************************************** \ntotal energy ",Ehf,Ehf+Em,Ehf+Ec)

SOLVER = FCI_solver(BO_IAO,BO_IAO_external)
print("FCI+FCI NEVPT2 energy ",SOLVER.compute_nevpt2_energy())

SOLVER = FCI_QSE_solver(BO_IAO,BO_IAO_external)
print("FCI+QSE NEVPT2 energy ", SOLVER.compute_nevpt2_energy())


print("="*53)
VQE_settings = vqe_data()
VQE_settings.ansatz = 'su2'
VQE_settings.reps = 0
SOLVER = VQE_QSE_solver(BO_IAO,BO_IAO_external)
SOLVER.set_quantum_variables(VQE_settings)
print("statevector --- VQE+QSE NEVPT2 energy ", SOLVER.compute_nevpt2_energy())
x = np.load('vqe_su2_output.npy',allow_pickle=True).item()
print("statevector --- VQE(Ry) energy %.12f \n" % x['energy'])
print("statevector --- VQE(Ry) angle %s \n" % str(x['opt_params']))


#print("="*53)
#VQE_settings = vqe_data()
#VQE_settings.ansatz = 'su2'
#VQE_settings.reps = 0
#VQE_settings.instance = 'qasm_simulator'
#VQE_settings.optimizer = 'adam'
#VQE_settings.shots = 100000
#SOLVER = VQE_QSE_solver(BO_IAO,BO_IAO_external)
#SOLVER.set_quantum_variables(VQE_settings)
#print("qasm, ideal --- VQE+QSE NEVPT2 energy ", SOLVER.compute_nevpt2_energy())
#x = np.load('vqe_su2_output.npy',allow_pickle=True).item()
#print("qasm, ideal --- VQE(Ry) energy %.12f \n" % x['energy'])
#print("qasm, ideal --- VQE(Ry) angle %s \n" % str(x['opt_params']))

x = np.load('vqe_su2_output.npy',allow_pickle=True).item()
VQE_settings = vqe_data()
VQE_settings.ansatz = 'su2'
VQE_settings.reps = 0
VQE_settings.instance = 'noise_model ibmq_manila'
VQE_settings.optimizer = 'cobyla'
VQE_settings.max_iter = 0
VQE_settings.shots = 8000
VQE_settings.initial_point = x['opt_params']
SOLVER = VQE_QSE_solver(BO_IAO,BO_IAO_external)   # <--------
SOLVER.set_quantum_variables(VQE_settings)
print("qasm, ideal --- VQE+QSE NEVPT2 energy ", SOLVER.compute_nevpt2_energy())
x = np.load('vqe_su2_output.npy',allow_pickle=True).item()
print("qasm, ideal --- VQE(Ry) energy %.12f \n" % x['energy'])
print("qasm, ideal --- VQE(Ry) angle %s \n" % str(x['opt_params']))

