import numpy as np
import sys,os
from qiskit import IBMQ
sys.path.append('../')
from tomography import Hardware_Tomography
import math

task = 'submit'

#device='kolkata'; device_name='ibmq_'+device
#device='auckland'; device_name='ibm_'+device

for device,device_name in zip(['kolkata','ibmq_kolkata'],['aukland','ibm_auckland']):
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-internal',group='performance',project='paper-priority')
    circuits = np.load('circuits.npy',allow_pickle=True).item()
    T = Hardware_Tomography(circuits=circuits,shots=100000,optimization_level=0,
                         	device_name=device_name,initial_layout=[0,1,2,3,5,8],provider=provider,
                        	ro_error_mitigation=True)

    num_circuits = len(circuits.keys())
    k0 = list(circuits.keys())[0]
    num_qubits = circuits[k0].num_qubits
    if(device=='kolkata'):  max_circuits_per_job = 150
    if(device=='auckland'): max_circuits_per_job = 300
    n_jobs = float(num_circuits*(3**num_qubits))/float(max_circuits_per_job)
    n_jobs = math.ceil(n_jobs)
    print("number of circuits ............. ",num_circuits)
    print("number of qubits ............... ",num_qubits)
    print("number of tomography circuits .. ",num_circuits*(3**num_qubits))
    print("number of jobs ................. ",n_jobs)

    if (task=='submit'):
        T.submit_probability_distribution_jobs(circuits_per_job=max_circuits_per_job,logfile_name='logfile_'+device+'_em.txt')
    elif(task=='retrieve'):
        jobs = [f.split()[2] for f in open('logfile_'+device+'_em.txt','r').readlines()]
        T.compute_probability_distributions(jobs,max_circuits_per_job,mitigated_counts=False)
        T.compute_bloch_vector()
        T.save_calculation(device+'_raw')
        T.compute_probability_distributions(jobs,max_circuits_per_job,mitigated_counts=True)
        T.compute_bloch_vector()
        T.save_calculation(device+'_em')
