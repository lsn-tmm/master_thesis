def make_qmolecule_from_data(mol_data):
    # dizionario trasformato in oggetto
    from qiskit.chemistry import QMolecule
    import numpy as np
    n,na,nb,h0,h1,h2 = mol_data['no'],mol_data['ne'][0],mol_data['ne'][1],mol_data['h0'],mol_data['h1'],mol_data['h2']
    m = QMolecule()
    m.nuclear_repulsion_energy = h0
    m.num_orbitals             = n
    m.num_alpha                = na
    m.num_beta                 = nb
    m.mo_coeff                 = np.eye(n)
    m.mo_onee_ints             = h1
    m.mo_eri_ints              = h2
    return m

def Identity(n):
    from qiskit.aqua.operators import WeightedPauliOperator
    from qiskit.quantum_info import Pauli
    import numpy as np
    zeros = [0]*n
    zmask = [0]*n
    a_x = np.asarray(zmask,dtype=np.bool)
    a_z = np.asarray(zeros,dtype=np.bool)
    return WeightedPauliOperator([(1.0,Pauli(a_x,a_z))])


def map_to_qubits(mol_data={},mapping='jordan_wigner',two_qubit_reduction=False,tapering=True,tapering_sector=None):
    import sys
    sys.path.append('../commons/')

    from qiskit.chemistry.core import Hamiltonian,TransformationType,QubitMappingType
    from tapering import taper_principal_operator,taper_auxiliary_operators

    if(mapping=='jordan_wigner'): qubit_mapping = QubitMappingType.JORDAN_WIGNER
    elif(mapping=='parity'):      qubit_mapping = QubitMappingType.PARITY
    else:                         assert(False)

    # in questa funzione, il tuo dizionario viene utilizzato per definire un oggetto qmolecule di qiskit
    molecule  = make_qmolecule_from_data(mol_data)
    # in questa funzione, l'Hamiltoniano e gli operatori N (numero di elettroni), Sz (spin-z totale) e S^2 (spin totale)
    # vengono convertiti da fermionici a qubit usando un certo qubit_mapping (tipicamente JW oppure parity)
    core      = Hamiltonian(transformation      = TransformationType.FULL,
                            qubit_mapping       = qubit_mapping,
                            two_qubit_reduction = two_qubit_reduction,
                            freeze_core         = False,
                            orbital_reduction   = [])
    H_op,A_op = core.run(molecule)
    dE        = core._energy_shift + core._ph_energy_shift + core._nuclear_repulsion_energy
    H_op      = H_op + dE * Identity(H_op.num_qubits)
    A_op      = A_op[:3]

    #print("Hamiltoniano trasformato")
    #print(H_op.print_details())
    #print("Number operator ")
    #print(A_op[0].print_details())
    #print("Spin-squared operator ")
    #print(A_op[1].print_details())
    #print("Spin-z operator ")
    #print(A_op[2].print_details())

    operators = {'orbitals'  : core._molecule_info['num_orbitals'],
                 'particles' : core._molecule_info['num_particles'],
                 'mapping'   : core._qubit_mapping,
                 '2qr'       : two_qubit_reduction,
                 'tapering'  : tapering}
    if(tapering):
       # TAPERING
       # dato un Hamiltoniano H, definiamo
       # S = { P operatori di Pauli : [P,H] = 0 }
       # costruiamo da S un sottogruppo Abeliano S' e ne troviamo i generatori T1 ... Tk
       # T1 ... Tk sono operatori di Pauli
       # [Ti,Tj] = 0 per ogni coppia
       # S' = <Ti ... Tj>
       # Il tapering esegue una trasformazione unitaria |Psi) -> U|Psi) nel gruppo di Clifford
       # tale che U Ti U* = Xi
       # esempio: l'oscillatore armonico V(x) = x^2/2
       # le autofunzioni dell'OA sono Psi(n,x) = Exp(-x^2) Hermite[n,x]
       # sono pari o dispari (1 o -1)
       # U|Psi) = |parity)|Phi)
       #  |Psi) = alpha |Psi,pari) + beta |Psi,dispari)
       # U|Psi) = alpha |0)|Psi') + beta |1)|Psi'')
       # beta = 0
       # U|Psi) = |0)|Psi')
       H_op_tapered,z2syms,sqlist,target_sector = taper_principal_operator(H_op,target_sector=tapering_sector)
       A_op_tapered                             = taper_auxiliary_operators(A_op,z2syms,target_sector)
       operators['h_op']           = H_op_tapered
       operators['a_op']           = A_op_tapered
       operators['names']          = ['number','spin-2','spin-z']
       operators['target_sector']  = target_sector
       operators['tapering_info']  = [z2syms,sqlist]
       operators['untapered_h_op'] = H_op
    else:
       operators['h_op']           = H_op
       operators['a_op']           = A_op
       operators['names']          = ['number','spin-2','spin-z']
       operators['target_sector']  = None
       operators['tapering_info']  = [None,None]

    #print("Hamiltoniano con TAPERING")
    #print(H_op_tapered.print_details())
    #exit()

    return molecule,operators

def produce_variational_form(molecule,operators,ansatz={}):
    import sys
    sys.path.append('../commons/')
    from qiskit.chemistry.components.initial_states import HartreeFock

    initial_state = HartreeFock(num_orbitals  = operators['orbitals'],
                                qubit_mapping = operators['mapping'],
                                num_particles = operators['particles'],
                                sq_list       = operators['tapering_info'][1])
    print("Hartree Fock state")
    print(initial_state.construct_circuit().draw())
    if(ansatz['type']=='q_uccsd'):
       from local_uccsd import UCCSD
       var_form = UCCSD(num_orbitals        = operators['orbitals'],
                        num_particles       = operators['particles'],
                        active_occupied     = None,
                        active_unoccupied   = None,
                        initial_state       = initial_state,
                        qubit_mapping       = operators['mapping'],
                        two_qubit_reduction = operators['2qr'],
                        num_time_slices     = ansatz['reps'],
                        z2_symmetries       = operators['tapering_info'][0],
                        target_sector       = operators['target_sector'])
    elif(ansatz['type']=='so4'):
       from SO4 import var_form_unitary
       var_form = var_form_unitary(nqubit        = initial_state.construct_circuit().num_qubits,
                                   depth         = ansatz['reps'],
                                   initial_state = initial_state,
                                   entanglement  = ansatz['entanglement'])
    elif(ansatz['type']=='evangelista'):
       from Evangelista import Evangelista
       var_form = Evangelista(instructions  = operators,
                              initial_state = initial_state.construct_circuit(),
                              reps          = ansatz['reps'])
    else:
       assert(False)
    return var_form

def run_vqe(mol_data,operators,var_form,optimizer_dict,instance_dict,fname_prefix=None,outfile=None,penalty=[0,0]):
    import sys
    sys.path.append('../commons/')
    from qiskit.aqua.components.optimizers import L_BFGS_B,CG,SPSA,COBYLA,ADAM
    from qiskit.aqua.algorithms            import VQE
    from qiskit                            import Aer
    from qiskit.aqua                       import QuantumInstance
    from harvest                           import measure_operators
    from qiskit.aqua.algorithms            import NumPyEigensolver
    from prettytable                       import PrettyTable
    import os
    import numpy as np

    if(optimizer_dict['name']=='cg'):
       optimizer = CG(maxiter=optimizer_dict['max_iter'])
    elif(optimizer_dict['name']=='adam'):
       optimizer = ADAM(maxiter=optimizer_dict['max_iter'])
    elif(optimizer_dict['name']=='spsa'):
       optimizer = SPSA(maxiter=1000)
    elif(optimizer_dict['name']=='bfgs'):
       optimizer = L_BFGS_B(maxiter=optimizer_dict['max_iter'],iprint=1001)
    else:
       assert(False)

    backend          = Aer.get_backend(instance_dict['instance'])
    quantum_instance = QuantumInstance(backend=backend,shots=instance_dict['shots'])

    if(os.path.isfile(fname_prefix+'_input_parameters.txt')):
       p0 = np.loadtxt(fname_prefix+'_input_parameters.txt')
    else:
       p0 = np.zeros(var_form.num_parameters)

    # H --> H + alpha (N-N0)**2
    ID = Identity(operators['h_op'].num_qubits)
    J  = operators['h_op']+penalty[0]*(operators['a_op'][0]-penalty[1]*ID)*(operators['a_op'][0]-penalty[1]*ID)

    algo             = VQE(J,var_form,optimizer,aux_operators=operators['a_op'],include_custom=True,initial_point=p0)
    algo_result      = algo.run(quantum_instance)

    p1      = algo._ret['opt_params']
    res_vqe = measure_operators([operators['h_op']]+operators['a_op'],var_form.construct_circuit(p1),quantum_instance)

    ee      = NumPyEigensolver(operator=operators['h_op'],k=1,aux_operators=operators['a_op']).run()
    res_ee  = [ee['eigenvalues'][0]]+[ee['aux_operator_eigenvalues'][0][i][0] for i in range(len(operators['a_op']))]

    t = PrettyTable(['method','energy']+operators['names'])
    t.add_row(['VQE']+[str(round(np.real(x),8))+' +/- '+str(round(np.real(y),6)) for x,y in res_vqe])
    t.add_row(['FCI']+[str(round(np.real(x),8)) for x in res_ee])
    outfile.write("\nVQE %s\n" % fname_prefix)
    outfile.write(str(t))
    outfile.write("\n")
    np.savetxt(fname_prefix+"_output_parameters.txt",p1)

    res = {'mol_data'            : mol_data,
           'operators'           : operators,
           'num_parameters'      : var_form.num_parameters,
           'results_vqe'         : res_vqe,
           'vqe_circuit'         : var_form.construct_circuit(p1),
           'results_eigensolver' : res_ee}

    np.save(fname_prefix+'_results.npy',res,allow_pickle=True)

    return res
