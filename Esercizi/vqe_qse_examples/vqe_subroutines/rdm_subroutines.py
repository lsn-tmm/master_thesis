import numpy as np

def to_fermi_op(mol_info,h_1,h_2=None):
    from qiskit.chemistry import FermionicOperator
    import sys
    sys.path.append('../commons/')
    from tapering import taper_auxiliary_operators

    map_type = mol_info['operators']['mapping']
    tqr      = mol_info['operators']['2qr']
    nelec    = mol_info['operators']['particles']
    X        = FermionicOperator(h1=h_1,h2=h_2).mapping(map_type=map_type)
    if(map_type=='parity' and tqr):
       X = Z2Symmetries.two_qubit_reduction(X,nelec)
    if(mol_info['operators']['tapering']):
       z2syms        = mol_info['operators']['tapering_info'][0]
       target_sector = mol_info['operators']['target_sector']
       X = taper_auxiliary_operators([X],z2syms,target_sector)[0]
    return X

def build_rdm_operators(mol_info):
    no = mol_info['operators']['orbitals']//2
    ne = max(mol_info['operators']['particles'])
    operators_1 = []
    for p in range(no):
        for q in range(no):
            h_1            = np.zeros((2*no,2*no))
            h_1[p,q]       = 1.0
            h_1[p+no,q+no] = 1.0
            o_h_1          = to_fermi_op(mol_info,h_1,h_2=None)
            operators_1.append(o_h_1)
    operators_2 = []
    for p in range(no):
        for r in range(no):
            for q in range(no):
                for s in range(no):
                    h_1                      = np.zeros((2*no,2*no))
                    h_2                      = np.zeros((2*no,2*no,2*no,2*no))
                    h_2[p,r,q,s]             = 1.0
                    h_2[p+no,r+no,q,s]       = 1.0
                    h_2[p,r,q+no,s+no]       = 1.0
                    h_2[p+no,r+no,q+no,s+no] = 1.0
                    o_h_1_h_2                = to_fermi_op(mol_info,h_1,h_2)
                    operators_2.append(o_h_1_h_2)
    return operators_1,operators_2

def measure_rdm_operators(rdm_ops,mol_info,instance_dict):
    import sys
    sys.path.append('../commons/')
    from harvest import measure_operators
    from qiskit                            import Aer
    from qiskit.aqua                       import QuantumInstance

    backend  = Aer.get_backend(instance_dict['instance'])
    instance = QuantumInstance(backend=backend,shots=instance_dict['shots'])

    print(mol_info.keys())
    o1,o2   = rdm_ops
    circ    = mol_info['vqe_circuit']
    no      = mol_info['operators']['orbitals']//2
    res1    = measure_operators(o1,circ,instance)
    rdm1    = np.zeros((no,no,2))
    mu      = 0
    for p in range(no):
        for q in range(no):
            rdm1[p,q,:] = res1[mu]
            mu += 1     
    res2    = measure_operators(o2,circ,instance)
    rdm2    = np.zeros((no,no,no,no,2))
    mu      = 0
    for p in range(no):
        for r in range(no):
            for q in range(no):
                for s in range(no):
                    rdm2[p,r,q,s,:] = res2[mu]
                    mu += 1
    return res2,rdm2

