import numpy as np
import sys
from qiskit.chemistry.fermionic_operator import FermionicOperator
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.operators import Z2Symmetries

def adjoint(WPO):
    import numpy as np
    ADJ = WPO.copy()
    ADJ._paulis = [[np.conj(weight),pauli] for weight,pauli in WPO._paulis]
    return ADJ

def Identity(n):
    from qiskit.aqua.operators import WeightedPauliOperator
    from qiskit.quantum_info import Pauli
    import numpy as np
    zeros = [0]*n
    zmask = [0]*n
    a_x = np.asarray(zmask,dtype=np.bool)
    a_z = np.asarray(zeros,dtype=np.bool)
    return WeightedPauliOperator([(1.0,Pauli(a_x,a_z))])

def creators_destructors(F_op,n_spin_orbitals):
    a_list = F_op._jordan_wigner_mode(n_spin_orbitals)
    c_list = [WeightedPauliOperator([(0.5,a1),(-0.5j,a2)]) for a1,a2 in a_list]
    d_list = [WeightedPauliOperator([(0.5,a1),( 0.5j,a2)]) for a1,a2 in a_list]
    return c_list,d_list

def to_spin_orbitals(n,h1,h2):
    h1_so = np.zeros((2*n,2*n))
    h2_so = np.zeros((2*n,2*n,2*n,2*n))
    h1_so[:n,:n] = h1
    h1_so[n:,n:] = h1
    h2_so[:n,:n,:n,:n] = h2
    h2_so[n:,n:,:n,:n] = h2
    h2_so[:n,:n,n:,n:] = h2
    h2_so[n:,n:,n:,n:] = h2
    return h1_so,h2_so

def prepare_operators(SOLVER,quantum_variables):
    from tapering import taper_principal_operator,taper_auxiliary_operators
    from time     import time
    
    n = SOLVER.no
    aa_pairs = [(s,r) for s in range(n) for r in range(n) if s<r]
    ab_pairs = [(s,r) for s in range(n) for r in range(n)]
    oper = {}
    h1_so,h2_so = to_spin_orbitals(n,SOLVER.h1,SOLVER.h2)
    H_op = FermionicOperator(h1=h1_so,h2=0.5*h2_so)
    c_op,d_op = creators_destructors(H_op,2*n)
    H_op = H_op.mapping(map_type='jordan_wigner')
    H_op = H_op + SOLVER.h0*Identity(H_op.num_qubits)
    oper['hamiltonian'] = H_op
    E_operators = {}
    E_operators['u']  = [d_op[r]             for r in range(n)]      # distruttori di spin-orbitali up
    E_operators['d']  = [d_op[r+n]           for r in range(n)]      # distruttori di spin-orbitali down
    E_operators['uu'] = [d_op[s]*d_op[r]     for (s,r) in aa_pairs]  # a(s,up)*a(r,up)     s<r
    E_operators['dd'] = [d_op[s+n]*d_op[r+n] for (s,r) in aa_pairs]  # a(s,down)*a(r,down) s<r
    E_operators['ud'] = [d_op[s]*d_op[r+n]   for (s,r) in ab_pairs]  # a(s,up)*a(r,down) 
    
    # ----- Preparation time
    for k in ['u','d','uu','dd','ud']:
        start = time()
        oper['qse_s_'+k] = [(adjoint(El)*Em).chop(1e-8)      for l,El in enumerate(E_operators[k]) for m,Em in enumerate(E_operators[k]) if l<=m]
        oper['qse_h_'+k] = [(adjoint(El)*H_op*Em).chop(1e-8) for l,El in enumerate(E_operators[k]) for m,Em in enumerate(E_operators[k]) if l<=m]
        end = time()
        print("qse operators prepared for %s \t time: %.2f s" % (k,end-start))
        
    # ----- Tapering Time    
    start = time()
    H_op,z2syms,sqlist,target_sector = taper_principal_operator(H_op,target_sector=quantum_variables.target_sector)
    end = time()
    
    total = {}
    total['tot'] = 0
    tape_time = end-start
    for k in oper.keys(): 
        if k != 'hamiltonian':  
            total['tot'] += len(oper[k])
            total[k] = len(oper[k])
            
    print("\n Total number of operators: ", total['tot'])
    
    print("after tapering qubits(H) =  %i \t time: %.2f s" % (H_op.num_qubits,tape_time))
    
    oper['hamiltonian'] = H_op
    for k in oper.keys():
        if(k!='hamiltonian'):
           start = time()
           oper[k] = taper_auxiliary_operators(oper[k],z2syms,target_sector)
           end = time()
           print("qse operators tapered for %s %i of %i \t mean: %.2f s" % (k,len(oper[k]), total[k], (end-start)/len(oper[k])))
           tape_time += total[k] * (end-start)/len(oper[k])
            
    print('Taper time: %.2f s' % tape_time)
        
    quantum_variables.target_sector = target_sector
    quantum_variables.tapering_info = [z2syms,sqlist]
    return oper,quantum_variables

def produce_variational_form(solver,quantum_variables):
    from qiskit.chemistry.components.initial_states import HartreeFock
    from local_uccsd import UCCSD
    from CustomEfficientSU2 import CustomEfficientSU2

    initial_state = HartreeFock(num_orbitals  = 2*solver.no,
                                qubit_mapping = 'jordan_wigner',
                                num_particles = list(solver.ne),
                                sq_list       = quantum_variables.tapering_info[1])
    print(initial_state.construct_circuit())
    if(quantum_variables.ansatz=='q_uccsd'):
       var_form = UCCSD(num_orbitals        = 2*solver.no,
                        num_particles       = list(solver.ne),
                        active_occupied     = None,
                        active_unoccupied   = None,
                        initial_state       = initial_state,
                        qubit_mapping       = 'jordan_wigner',
                        two_qubit_reduction = False,
                        num_time_slices     = 1,
                        z2_symmetries       = quantum_variables.tapering_info[0],
                        target_sector       = quantum_variables.target_sector)
    elif(quantum_variables.ansatz=='su2'):
       var_form = CustomEfficientSU2(num_qubits=initial_state.construct_circuit().num_qubits,
                                     reps=quantum_variables.reps, 
                                     entanglement='linear',
                                     initial_state=initial_state.construct_circuit())
    else:
       assert(False)
    return var_form

class VQE_data:
    def __init__(self,energy,circuit,parameter,instance):
        self.energy    = energy
        self.circuit   = circuit
        self.parameter = parameter 
        self.instance  = instance

def run_vqe(solver,operators,quantum_variables):
    from qiskit.aqua.components.optimizers import L_BFGS_B,CG,SPSA,COBYLA,ADAM
    from qiskit.aqua.algorithms            import VQE
    from qiskit                            import Aer
    from qiskit.aqua                       import QuantumInstance   
    from time                              import time

    start = time()
    
    if(quantum_variables.optimizer=='cg'):       optimizer = CG(maxiter=quantum_variables.max_iter)
    elif(quantum_variables.optimizer=='adam'):   optimizer = ADAM(maxiter=quantum_variables.max_iter)
    elif(quantum_variables.optimizer=='spsa'):   optimizer = SPSA(maxiter=1000)
    elif(quantum_variables.optimizer=='bfgs'):   optimizer = L_BFGS_B(maxiter=quantum_variables.max_iter)
    elif(quantum_variables.optimizer=='cobyla'): optimizer = COBYLA(maxiter=quantum_variables.max_iter)
    else:                                        assert(False)

    if(quantum_variables.instance=='statevector_simulator' or quantum_variables.instance=='qasm_simulator'):
       backend          = Aer.get_backend(quantum_variables.instance)
       quantum_instance = QuantumInstance(backend=backend,shots=quantum_variables.shots)
    elif('noise_model' in quantum_variables.instance):
       nm,device = quantum_variables.instance.split()
       from qiskit import IBMQ
       from qiskit.providers.aer.noise import NoiseModel
       IBMQ.load_account()
       backend          = Aer.get_backend('qasm_simulator')
       provider = IBMQ.get_provider(hub='ibm-q')
       #provider         = IBMQ.get_provider(hub='ibm-q-internal',group='deployed',project='default')
       device           = provider.get_backend(device)
       properties       = device.properties()
       noise_model      = NoiseModel.from_backend(properties)
       coupling_map     = device.configuration().coupling_map
       quantum_instance = QuantumInstance(backend=backend,shots=quantum_variables.shots,
                                          coupling_map=coupling_map,basis_gates=noise_model.basis_gates)
    else:
       assert(False)

    var_form    = produce_variational_form(solver,quantum_variables)
    p0          = quantum_variables.initial_point
    algo        = VQE(operators['hamiltonian'],var_form,optimizer,initial_point=p0)
    algo_result = algo.run(quantum_instance)
    p = algo._ret['opt_params']
    E = algo._ret['energy']
    np.save('vqe_%s_output.npy'%quantum_variables.ansatz,algo._ret)
    end = time()
    
    print('VQE time: %.2f s' % (end-start))
    return VQE_data(E,var_form.construct_circuit(p),p,quantum_instance)

def measure_operators(operators,wfn_circuit,instance):
    import numpy as np
    import functools
    circuits = []
    for idx,oper in enumerate(operators):
        if(not oper.is_empty()):
           circuit = oper.construct_evaluation_circuit(
                     wave_function               = wfn_circuit,
                     statevector_mode            = instance.is_statevector,
                     use_simulator_snapshot_mode = instance.is_statevector,
                     circuit_name_prefix         = 'oper_'+str(idx))
           circuits.append(circuit)
    if circuits:
        to_be_simulated_circuits = \
            functools.reduce(lambda x, y: x + y, [c for c in circuits if c is not None])
        result = instance.execute(to_be_simulated_circuits)
    # ---
    results_list = []
    for idx,oper in enumerate(operators):
        if(not oper.is_empty()):
           mean,std = oper.evaluate_with_result(
                      result = result,statevector_mode = instance.is_statevector,
                      use_simulator_snapshot_mode = instance.is_statevector,
                      circuit_name_prefix         = 'oper_'+str(idx))
           if(np.abs(np.imag(mean))>1e-4): print("attention: IMAG",mean)
           results_list.append([np.real(mean),np.abs(std)])
        else:
           results_list.append([0,0])
    # ---
    return results_list

