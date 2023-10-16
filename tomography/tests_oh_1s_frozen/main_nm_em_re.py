import numpy as np
import sys,os
from qiskit import IBMQ
sys.path.append('../')
from tomography import QASM_Tomography

for device,device_name in zip(['kolkata','auckland'],['ibmq_kolkata','ibm_auckland']):
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-internal',group='performance',project='paper-priority')
    
    circuits = np.load('circuits.npy',allow_pickle=True).item()
    T = QASM_Tomography(circuits=circuits,shots=100000,optimization_level=0,
                        device_name=device_name,initial_layout=[0,1,2,3,5,8],provider=provider,ro_error_mitigation=True)
    T.noise_amplification(3)
    T.compute_probability_distributions()
    T.compute_bloch_vector()
    T.save_calculation(device+'_nm_em_re')

