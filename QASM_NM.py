import sys
sys.path.append('../../../../../multi_unitary_thc/multi_unitary_thc_EC/')
from hamiltonian import *
from decomposer  import *
from circuits    import *
from mapper      import *
H = Hamiltonian('../../../hamiltonian_ethylene-0.npy')
decomposer_dict = {'tolerance'       : 1e-4,
                   'echo_pulses'     : 0,
                   'split_givens'    : False,
                   'post_selection'  : True,
                   'n_layer_circuit' : 1}
D = Decomposer(H,decomposer_dict)
D.generate_unitaries()
operator_dict = {'qubit_mapping'      : 'jordan_wigner',
                'two_qubit_reduction' : False,
                'chop'                : 1e-5}
M = Mapper(operator_dict,H,D)
from qiskit      import *
from qiskit.aqua import QuantumInstance,aqua_globals
from qiskit.providers.aer.noise import NoiseModel
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)
seed = 0
aqua_globals.random_seed = seed
IBMQ.load_account()
backend      = Aer.get_backend('qasm_simulator')
provider     = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
real_device  = provider.get_backend('ibmq_paris')
properties   = real_device.properties()
coupling_map = real_device.configuration().coupling_map
from qiskit.providers.aer.noise import NoiseModel
noise_model = NoiseModel.from_backend(properties)
instance = QuantumInstance(backend=backend,seed_transpiler=seed,seed_simulator=seed,shots=10000,
                           coupling_map=coupling_map,basis_gates=noise_model.basis_gates)
circuit_dict = {'initial_circuit'         : ([[1,0,1,0]],None),
                'num_steps'               : 1,
                'time_step'               : 20.000000,
                'instance'                : instance,
                'dump_data'               : ('hamiltonian_ethylene-0_raw_result_post.pkl','write'),
                'post_selection_function' : 'spin'}
post_process = {'ntry':20000,'threshold':1e-2,'seed':0,'results_file':'hamiltonian_ethylene-0_post_processed_result_post.pkl'}
C = QFD_THC_circuits(H,D,M,circuit_dict)
C.run(post_process)
print(C.print_details())
