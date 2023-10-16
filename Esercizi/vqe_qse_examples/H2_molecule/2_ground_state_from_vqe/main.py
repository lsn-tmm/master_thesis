import numpy as np
import sys
sys.path.append('../../commons/')
sys.path.append('../../vqe_subroutines')
from   subroutines import map_to_qubits,produce_variational_form,run_vqe
from   job_details import write_job_details

mol_data = np.load('../1_generate_hamiltonian/iao_6-31g.npy',allow_pickle=True).item()
outfile  = open('vqe.txt','w')
write_job_details(outfile)
molecule,operators = map_to_qubits(mol_data,mapping='jordan_wigner',two_qubit_reduction=False,tapering=True)
var_form           = produce_variational_form(mol_data,operators,ansatz={'type':'q_uccsd','reps':1})
results            = run_vqe(mol_data,operators,var_form,optimizer_dict={'name':'bfgs','max_iter':1000},
                             instance_dict={'instance':'statevector_simulator','shots':1},fname_prefix='vqe_q_uccsd',outfile=outfile)

