import numpy as np

from qiskit.chemistry.fermionic_operator import FermionicOperator
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.operators import Z2Symmetries

def creators_and_destructors(n_spin_orbitals,qubit_mapping,two_qubit_reduction,num_particles):
    if(type(qubit_mapping)==str):
       map_type = qubit_mapping
    else:
       map_type = qubit_mapping.value.lower()
    # -----
    h1 = np.zeros((n_spin_orbitals,n_spin_orbitals))
    if map_type == 'jordan_wigner':
        a_list = FermionicOperator(h1)._jordan_wigner_mode(n_spin_orbitals)
    elif map_type == 'parity':
        a_list = FermionicOperator(h1)._parity_mode(n_spin_orbitals)
    elif map_type == 'bravyi_kitaev':
        a_list = FermionicOperator(h1)._bravyi_kitaev_mode(n_spin_orbitals)
    elif map_type == 'bksf':
        return bksf_mapping(self)
    else:
        assert(False)
    # creators
    c_list = [WeightedPauliOperator([(0.5,a1),(-0.5j,a2)]) for a1,a2 in a_list]
    # destructors
    d_list = [WeightedPauliOperator([(0.5,a1),(+0.5j,a2)]) for a1,a2 in a_list]

    if map_type == 'parity' and two_qubit_reduction:
        c_list = [Z2Symmetries.two_qubit_reduction(c,num_particles) for c in c_list]
        d_list = [Z2Symmetries.two_qubit_reduction(d,num_particles) for d in d_list]

    return c_list,d_list

def Identity(n):
    from qiskit.aqua.operators import WeightedPauliOperator
    from qiskit.quantum_info import Pauli
    import numpy as np
    zeros = [0]*n
    zmask = [0]*n
    a_x = np.asarray(zmask,dtype=np.bool)
    a_z = np.asarray(zeros,dtype=np.bool)
    return WeightedPauliOperator([(1.0,Pauli(a_x,a_z))])

def to_fermi_op(mol_info,h_1,h_2=None):
    from qiskit.chemistry import FermionicOperator
    map_type = mol_info['operators']['mapping']
    tqr      = mol_info['operators']['2qr']
    nelec    = mol_info['operators']['particles']
    X        = FermionicOperator(h1=h_1,h2=h_2).mapping(map_type=map_type)
    if(map_type=='parity' and tqr):
       X = Z2Symmetries.two_qubit_reduction(X,nelec)
    return X

def excitation_numbers(mol_info):
    no = mol_info['operators']['orbitals']//2
    ne = max(mol_info['operators']['particles'])
    num_singles = 2*ne*(no-ne)
    num_cepa    =   ne*(no-ne) 
    return num_singles,num_cepa

def build_qse_operators(class_of_operators,mol_info):
    n_spin_orbitals     = mol_info['operators']['orbitals']
    qubit_mapping       = mol_info['operators']['mapping']
    two_qubit_reduction = mol_info['operators']['2qr']
    num_particles       = mol_info['operators']['particles']
    c_list,d_list = creators_and_destructors(n_spin_orbitals,qubit_mapping,two_qubit_reduction,num_particles)

    d_list_up   = d_list[:n_spin_orbitals//2]
    d_list_down = d_list[n_spin_orbitals//2:]

    if(class_of_operators=='u'):
        return d_list_up
    elif(class_of_operators=='d'):
        return d_list_down
    elif(class_of_operators=='uu'):
        ret = []
        for p in range(n_spin_orbitals//2):
            for q in range(p+1,n_spin_orbitals//2):
                ret.append(d_list_up[p]*d_list_up[q])
        return ret
    elif(class_of_operators=='dd'):
        ret = []
        for p in range(n_spin_orbitals//2):
            for q in range(p+1,n_spin_orbitals//2):
                ret.append(d_list_down[p]*d_list_down[q])
        return ret
    elif(class_of_operators=='ud'):
        ret = []
        for p in range(n_spin_orbitals//2):
            for q in range(n_spin_orbitals//2):
                ret.append(d_list_up[p]*d_list_down[q])
        return ret
    else:
        assert(False)

