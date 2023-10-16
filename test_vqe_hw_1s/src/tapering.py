def diagonalize(operator):
    import numpy as np
    from   qiskit.aqua.algorithms import NumPyEigensolver

    res = NumPyEigensolver(operator,k=1).run()
    return np.real(res['eigenvalues'][0])

def taper_principal_operator(qubit_op,target_sector=None):
    # qubit_op = H
    from qiskit.aqua.operators import Z2Symmetries
    import numpy as np

    z2_symmetries = Z2Symmetries.find_Z2_symmetries(qubit_op) # cerca degli operatori [Si,H]=0 [Si,Sj]=0
    nsym        = len(z2_symmetries.sq_paulis)
    z2syms      = z2_symmetries
    sqlist      = z2_symmetries.sq_list
    tapered_ops = z2_symmetries.taper(qubit_op)               # U|Psi) = |Phi)|x) x=label symmetry; UHU* = \sum_i H_i <=== tapered_ops 

    if(target_sector is None):                                # se gia' so quale settore i mi interessa, sostituisco H con H_i
       e_min = 1e10
       i_min = -1
       for i in range(len(tapered_ops)):                      # altrimenti, ciclo su tutti gli operatori H_i
           O_i = get_matrix(tapered_ops[i])                   # cerco l'elemento E(i) = min[diag(H_i)]
           e_i = np.min(np.real(np.diag(O_i)))
           print("In the tapering subroutine, i=%d, E(i)=%.6f, i_min=%d" % (i,e_i,i_min))
           if(e_i<e_min):                                     # e seleziono il settore i con E(i) minimo; FUNZIONA SE MINIMO UNICO
              e_min = e_i
              i_min = i
       return tapered_ops[i_min],z2syms,sqlist,i_min
    else:
       return tapered_ops[target_sector],z2syms,sqlist,target_sector

def get_matrix(O):
    from qiskit.aqua.operators.legacy import op_converter
    return op_converter.to_matrix_operator(O)._matrix.todense()

def taper_auxiliary_operators(A_op,z2syms,target_sector):
    import numpy as np
    from qiskit.aqua.operators import WeightedPauliOperator
    from qiskit.quantum_info   import Pauli
    from qiskit.aqua.operators import commutator

    Ws_list = [ s for s in z2syms.symmetries]
    Ws_list = [ WeightedPauliOperator(paulis=[[1.0,s]]) for s in Ws_list] 
    Ws_list = [ get_matrix(s) for s in Ws_list]

    the_ancillas = []
    for A in A_op:
        if(not A.is_empty()):
           comm = True
           A_mat = get_matrix(A)
           for s_mat in Ws_list:
               comm_s = np.dot(A_mat,s_mat)-np.dot(s_mat,A_mat)
               comm_s = np.abs(comm_s).max()
               if(comm_s>1e-6): comm=False
           if(not comm):
              the_ancillas.append(WeightedPauliOperator(paulis=[[0.0,z2syms.symmetries[0]]]))
           else:
              A_taper = z2syms.taper(A)
              if(type(A_taper)==list): the_ancillas.append(A_taper[target_sector])
              else: the_ancillas.append(A_taper)
    return the_ancillas

