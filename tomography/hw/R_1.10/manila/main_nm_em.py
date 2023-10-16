import numpy as np
import sys,os
from qiskit import IBMQ
sys.path.append('../../../')
from tomography import QASM_Tomography

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q',group='open',project='main')

circuits = np.load('../circuits.npy',allow_pickle=True).item()
T = QASM_Tomography(circuits=circuits,shots=100000,optimization_level=0,
                    device_name='ibmq_manila',initial_layout=[1,2,3,4],provider=provider,ro_error_mitigation=True)

T.compute_probability_distributions()
T.compute_bloch_vector()
T.save_calculation('../tomo_nm_em')

