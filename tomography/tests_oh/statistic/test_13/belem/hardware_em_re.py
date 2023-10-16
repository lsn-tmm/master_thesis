import numpy as np
import sys,os
from qiskit import IBMQ
sys.path.append('../../../../')
from tomography import Hardware_Tomography
import math

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q',group='open',project='main')
circuits = np.load('../../circuits.npy',allow_pickle=True).item()
T = Hardware_Tomography(circuits=circuits,shots=20000,optimization_level=0,
                        device_name='ibmq_belem',initial_layout=[0,1,3,4],provider=provider,
                        ro_error_mitigation=True)
T.noise_amplification(3)

num_circuits = len(circuits.keys())
k0 = list(circuits.keys())[0]
num_qubits = circuits[k0].num_qubits
max_circuits_per_job = 100 
n_jobs = float(num_circuits*(3**num_qubits))/float(max_circuits_per_job)
n_jobs = math.ceil(n_jobs)
print("number of circuits ............. ",num_circuits)
print("number of qubits ............... ",num_qubits)
print("number of tomography circuits .. ",num_circuits*(3**num_qubits))
print("number of jobs ................. ",n_jobs)

task = 'retrieve'

if(task=='submit'):
   T.submit_probability_distribution_jobs(circuits_per_job=max_circuits_per_job,logfile_name='logfile_belem_em_re.txt')
elif(task=='retrieve'):
   jobs = [f.split()[2] for f in open('logfile_belem_em_re.txt','r').readlines()]
   T.compute_probability_distributions(jobs,max_circuits_per_job,mitigated_counts=True)
   T.compute_bloch_vector()
   T.save_calculation('tomo_HW_em_re')  
