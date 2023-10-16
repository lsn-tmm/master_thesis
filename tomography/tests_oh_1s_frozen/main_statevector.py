import numpy as np
import sys,os
from qiskit import IBMQ
sys.path.append('../')
from tomography import Statevector_Tomography

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q',group='open',project='main')

circuits = np.load('circuits.npy',allow_pickle=True).item()
T = Statevector_Tomography(circuits)
T.compute_bloch_vector()
T.save_calculation('tomo_statevector')

