#!/bin/bash

for HW in lima; #manila lima quito belem santiago bogota;
do
  task='retrieve'
  if [ ${task}=='submit' ]
  then
      mkdir ${HW}
  fi
  qubit=0
  if [ ${HW} == manila ] || [ ${HW} == bogota ] || [ ${HW} == santiago ]
  then
      qubit=[1,2,3,4]
      
  elif [ ${HW} == belem ] || [ ${HW} == lima ]
  then
      qubit=[0,1,3,4]
  elif [ ${HW} == quito ] 
  then
      qubit=[2,1,3,4]
  fi 
  cat > ${HW}/hardware_em.py  <<***
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
                        device_name='ibmq_${HW}',initial_layout=${qubit},provider=provider,
                        ro_error_mitigation=True)

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

task = '${task}'

if(task=='submit'):
   T.submit_probability_distribution_jobs(circuits_per_job=max_circuits_per_job,logfile_name='logfile_${HW}_em.txt')
elif(task=='retrieve'):
   jobs = [f.split()[2] for f in open('logfile_${HW}_em.txt','r').readlines()]
   T.compute_probability_distributions(jobs,max_circuits_per_job,mitigated_counts=False)
   T.compute_bloch_vector()
   T.save_calculation('tomo_HW_raw')
   T.compute_probability_distributions(jobs,max_circuits_per_job,mitigated_counts=True)
   T.compute_bloch_vector()
   T.save_calculation('tomo_HW_em')
***
  cat > ${HW}/hardware_em_re.py  <<***
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
                        device_name='ibmq_${HW}',initial_layout=${qubit},provider=provider,
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

task = '${task}'

if(task=='submit'):
   T.submit_probability_distribution_jobs(circuits_per_job=max_circuits_per_job,logfile_name='logfile_${HW}_em_re.txt')
elif(task=='retrieve'):
   jobs = [f.split()[2] for f in open('logfile_${HW}_em_re.txt','r').readlines()]
   T.compute_probability_distributions(jobs,max_circuits_per_job,mitigated_counts=True)
   T.compute_bloch_vector()
   T.save_calculation('tomo_HW_em_re')  
***
  cd ${HW}
  python hardware_em.py
  python hardware_em_re.py
  echo "${HW} finished"
  cd ../
done
