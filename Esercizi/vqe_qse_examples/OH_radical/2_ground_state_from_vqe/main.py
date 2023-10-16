import numpy as np
import sys
sys.path.append('../../commons/')
sys.path.append('../../vqe_subroutines')
from   subroutines import map_to_qubits,produce_variational_form,run_vqe
from   job_details import write_job_details

import logging
from   qiskit.chemistry import set_qiskit_chemistry_logging
from   qiskit.aqua      import set_qiskit_aqua_logging
set_qiskit_chemistry_logging(logging.INFO)
set_qiskit_aqua_logging(logging.INFO)

mol_data = np.load('../1_generate_hamiltonian/OH_radical_iao_6-31++g**.npy',allow_pickle=True).item()
outfile  = open('vqe.txt','w')
write_job_details(outfile)
molecule,operators = map_to_qubits(mol_data,mapping='jordan_wigner',two_qubit_reduction=False,tapering=True,tapering_sector=11)
var_form           = produce_variational_form(mol_data,operators,ansatz={'type':'q_uccsd','reps':1})
results            = run_vqe(mol_data,operators,var_form,optimizer_dict={'name':'bfgs','max_iter':1000},
                             instance_dict={'instance':'statevector_simulator','shots':1},fname_prefix='vqe_q_uccsd',
                             outfile=outfile,penalty=[1.0,mol_data['ne'][0]+mol_data['ne'][1]])

