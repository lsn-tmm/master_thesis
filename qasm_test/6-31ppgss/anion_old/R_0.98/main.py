import numpy as np
import sys
sys.path.append('../../../../nevpt2/src/')
import time
from basis_constructor import Basis_Constructor
from bo_class          import BO_class
from bo_solver         import BO_solver
from fci_qse_solver    import FCI_QSE_solver
from fci_solver        import FCI_solver
from vqe_qse_solver    import VQE_QSE_solver
from pyscf             import gto,scf
from VQE_types         import vqe_data

start = time.time()

outfile = open('results.txt','w')
mol = gto.M(atom=[['O',(0,0,0)],['H',(0,0,0.98)]],basis='6-31++g**',spin=0,charge=-1,
            symmetry=True,verbose=4).build()
mf  = scf.ROHF(mol)
mf  = scf.newton(mf)
mf.kernel()
if(not mf.converged):
   mf.kernel(mf.make_rdm1())
mf.analyze()

BC_object = Basis_Constructor(mol,mf)
BC_object.compute_core_valence()
BC_object.compute_external()

BO_IAO = BO_class(mol,mf)
BO_IAO.transform_integrals(BC_object.valence)
BO_IAO = BO_IAO.freeze_orbitals([0,1])
BO_IAO_solver = BO_solver(BO_IAO)
#outfile.write("IAO, [1s2s frozen] SCF energy %.12f \n" % BO_IAO_solver.solve_with_scf()[1])
#outfile.write("IAO, [1s2s frozen] FCI energy %.12f \n" % BO_IAO_solver.solve_with_fci()[0])

BO_IAO_external = BO_class(mol,mf)
BO_IAO_external.transform_integrals(BC_object.return_basis())
BO_IAO_external = BO_IAO_external.freeze_orbitals([0,1])
BO_IAO_external_solver = BO_solver(BO_IAO_external)
#outfile.write("FULL, [1s2s frozen] SCF energy %.12f \n" % BO_IAO_external_solver.solve_with_scf()[1])
#outfile.write("FULL, [1s2s frozen] FCI energy %.12f \n" % BO_IAO_external_solver.solve_with_fci()[0])

SOLVER = FCI_solver(BO_IAO,BO_IAO_external)
#outfile.write("FCI+FCI [1s2s frozen] NEVPT2 energy %.12f \n" % SOLVER.compute_nevpt2_energy())
SOLVER = FCI_QSE_solver(BO_IAO,BO_IAO_external)
outfile.write("FCI+QSE [1s2s frozen] NEVPT2 energy  %.12f \n" % SOLVER.compute_nevpt2_energy())

#  -----------------------------------------------------------------------------

VQE_settings = vqe_data()
SOLVER = VQE_QSE_solver(BO_IAO,BO_IAO_external)
SOLVER.set_quantum_variables(VQE_settings)
outfile.write("VQE(q-UCCSD)+QSE [1s2s frozen] NEVPT2 energy %.12f \n" % SOLVER.compute_nevpt2_energy())

#  -----------------------------------------------------------------------------

x = np.load('vqe_q_uccsd_output.npy',allow_pickle=True).item()
outfile.write("VQE(q-UCCSD) energy %.12f \n" % x['energy'])

VQE_settings.ansatz = 'su2'
VQE_settings.optimizer = 'cg'
VQE_settings.reps = 3
SOLVER.set_quantum_variables(VQE_settings)
outfile.write("statevector --- VQE(Ry)+QSE [1s2s frozen] NEVPT2 energy %.12f \n" % SOLVER.compute_nevpt2_energy())


#  -----------------------------------------------------------------------------

x = np.load('vqe_su2_output.npy',allow_pickle=True).item()
outfile.write("statevector --- VQE(Ry) energy (I) %.12f \n" % x['energy'])
outfile.write("statevector --- VQE(Ry) angle %s \n" % str(x['opt_params']))

#  -----------------------------------------------------------------------------

VQE_settings.instance = 'statevector_simulator'
VQE_settings.optimizer = 'cobyla'
VQE_settings.max_iter = 0
VQE_settings.initial_point = x['opt_params']
SOLVER.set_quantum_variables(VQE_settings)
outfile.write("statevector --- VQE(Ry)+QSE [1s2s frozen] NEVPT2 energy %.12f \n" % SOLVER.compute_nevpt2_energy())


#  -----------------------------------------------------------------------------

x = np.load('vqe_su2_output.npy',allow_pickle=True).item()
outfile.write("statevector --- VQE(Ry) energy (II) %.12f \n" % x['energy'])
outfile.write("statevector --- VQE(Ry) angle %s \n" % str(x['opt_params']))

#  -----------------------------------------------------------------------------

VQE_settings = vqe_data()
VQE_settings.ansatz = 'su2'
VQE_settings.reps = 3
VQE_settings.instance = 'qasm_simulator'
VQE_settings.optimizer = 'spsa'
VQE_settings.shots = 8000
SOLVER = VQE_QSE_solver(BO_IAO,BO_IAO_external)
SOLVER.set_quantum_variables(VQE_settings)
outfile.write("qasm, ideal --- VQE+QSE [1s2s frozen] NEVPT2 energy %.12f \n" % SOLVER.compute_nevpt2_energy())
x = np.load('vqe_su2_output.npy',allow_pickle=True).item()
outfile.write("qasm, ideal --- VQE(Ry) energy %.12f \n" % x['energy'])
outfile.write("qasm, ideal --- VQE(Ry) angle %s \n" % str(x['opt_params']))

end = time.time()

outfile.write("Total time execution %.2f s\n" % (end-start))
