import numpy as np
import sys
sys.path.append('../../../../../bloch/src/')
from basis_constructor import Basis_Constructor
from bo_class          import BO_class
from bo_solver         import BO_solver
from fci_qse_solver    import FCI_QSE_solver
from fci_solver        import FCI_solver
from vqe_qse_solver    import VQE_QSE_solver
from pyscf             import gto,scf
from VQE_types         import vqe_data

outfile = open('results.txt','w')
mol = gto.M(atom=[['O',(0,0,0)],['H',(0,0,0.90)]],basis='aug-cc-pvqz',spin=1,charge=0,
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
BO_IAO = BO_IAO.freeze_orbitals([0])
BO_IAO_solver = BO_solver(BO_IAO)
outfile.write("IAO, [1s frozen] SCF energy %.12f \n" % BO_IAO_solver.solve_with_scf()[1])
outfile.write("IAO, [1s frozen] FCI energy %.12f \n" % BO_IAO_solver.solve_with_fci()[0])

BO_IAO_external = BO_class(mol,mf)
BO_IAO_external.transform_integrals(BC_object.return_basis())
BO_IAO_external = BO_IAO_external.freeze_orbitals([0])
BO_IAO_external_solver = BO_solver(BO_IAO_external)
outfile.write("FULL, [1s frozen] SCF energy %.12f \n" % BO_IAO_external_solver.solve_with_scf()[1])
outfile.write("FULL, [1s frozen] FCI energy %.12f \n" % np.nan) 

SOLVER = FCI_solver(BO_IAO,BO_IAO_external)
outfile.write("FCI+FCI [1s frozen] NEVPT2 energy %.12f \n" % SOLVER.compute_nevpt2_energy())
SOLVER = FCI_QSE_solver(BO_IAO,BO_IAO_external)
outfile.write("FCI+QSE [1s frozen] NEVPT2 energy  %.12f \n" % SOLVER.compute_nevpt2_energy())

#  -----------------------------------------------------------------------------
'''

BO_IAO = BO_class(mol,mf)
BO_IAO.transform_integrals(BC_object.valence)
BO_IAO = BO_IAO.freeze_orbitals([0,1])
BO_IAO_solver = BO_solver(BO_IAO)
outfile.write("IAO, [1s,2s frozen] SCF energy %.12f \n" % BO_IAO_solver.solve_with_scf()[1])
outfile.write("IAO, [1s,2s frozen] FCI energy %.12f \n" % BO_IAO_solver.solve_with_fci()[0])

BO_IAO_external = BO_class(mol,mf)
BO_IAO_external.transform_integrals(BC_object.return_basis())
BO_IAO_external = BO_IAO_external.freeze_orbitals([0,1])
BO_IAO_external_solver = BO_solver(BO_IAO_external)
outfile.write("FULL, [1s,2s frozen] SCF energy %.12f \n" % BO_IAO_external_solver.solve_with_scf()[1])
outfile.write("FULL, [1s,2s frozen] FCI energy %.12f \n" % np.nan)

SOLVER = FCI_solver(BO_IAO,BO_IAO_external)
outfile.write("FCI+FCI [1s,2s frozen] NEVPT2 energy %.12f \n" % SOLVER.compute_nevpt2_energy())
SOLVER = FCI_QSE_solver(BO_IAO,BO_IAO_external)
outfile.write("FCI+QSE [1s,2s frozen] NEVPT2 energy %.12f \n" % SOLVER.compute_nevpt2_energy())
'''
#  -----------------------------------------------------------------------------

VQE_settings = vqe_data()
SOLVER = VQE_QSE_solver(BO_IAO,BO_IAO_external)
x = np.load('../R_0.90/vqe_su2_output.npy',allow_pickle=True).item()
outfile.write("VQE(Ry) energy (I) %.12f \n" % x['energy'])
VQE_settings.ansatz = 'su2'
VQE_settings.reps = 3
VQE_settings.instance = 'statevector_simulator'
VQE_settings.optimizer = 'cobyla'
VQE_settings.target_sector = 11
VQE_settings.max_iter = 0
VQE_settings.initial_point = x['opt_params']
y = np.load('../../../../../tomography/tests_oh_1s_frozen/auckland_nm_bloch.npy',allow_pickle=True).item()['radical_aug-cc-pvqz_0.90']
nsample = 100
E = np.zeros(nsample)
for i in range(nsample):
    yi = np.zeros(y.shape)
    for r in range(yi.shape[0]):
        yi[r,0] = np.random.normal(y[r,0],y[r,1])
    SOLVER.set_quantum_variables(VQE_settings,with_bloch_vector=True,bloch_vector=yi)
    Ei = SOLVER.compute_nevpt2_energy()
    outfile.write("VQE(Ry)+QSE [1s frozen] NEVPT2 energy %.12f \n" % Ei)
    E[i] = Ei
outfile.write("VQE(Ry)+QSE [1s frozen] NEVPT2 energy ---- FINAL AVERAGE ---- %.12f +/- %.12f \n" % (np.mean(E),np.std(E)))

