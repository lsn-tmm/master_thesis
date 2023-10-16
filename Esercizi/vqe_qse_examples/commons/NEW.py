import numpy as np
import itertools
from typing         import List,Optional,Union
from qiskit         import QuantumRegister,QuantumCircuit
from qiskit.circuit import ParameterVector,Parameter
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.components.initial_states    import InitialState

def is_xx(i,a,n):
    si = i//n
    sa = a//n
    if(si==sa): return True
    else:       return False

def is_xxxx(i,a,j,b,n):
    si = i//n
    sa = a//n
    sj = j//n
    sb = b//n
    if(si==sa and sj==sb and si==sj and i<j and a<b): return True
    else: return False

def is_xxyy(i,a,j,b,n):
    si = i//n
    sa = a//n
    sj = j//n
    sb = b//n
    if(si==sa and sj==sb and si!=sj): return True
    else: return False

def build_operators(occ,vrt,n):
    op      = []
    loop    = (len(occ)>0)
    while(loop):
       i   = occ[0]
       dop = [] 
       for a in vrt:
           if(is_xx(i,a,n)): dop.append((i,a))
       for j in occ[1:]:
          sj = j//n
          for a in vrt:
              sa = a//n
              for b in vrt: 
                  sb = b//n
                  if(is_xxxx(i,a,j,b,n) or is_xxyy(i,a,j,b,n)):
                     dop.append((i,a,j,b))
       occ = occ[1:]
       op += dop
       print(i)
       print(dop)
       loop = (len(occ)>0)
    #exit()
    return op

def to_qubit_op(mol_info,h_1,h_2=None):
    from qiskit.chemistry import FermionicOperator
    from qiskit.chemistry.core import Hamiltonian,TransformationType,QubitMappingType
    from tapering import taper_principal_operator,taper_auxiliary_operators
    map_type = mol_info['mapping']
    tqr      = mol_info['2qr']
    nelec    = mol_info['particles']
    tapering = mol_info['tapering']
    X        = FermionicOperator(h1=h_1,h2=h_2).mapping(map_type=map_type)
    if(map_type=='parity' and tqr):
       X = Z2Symmetries.two_qubit_reduction(X,nelec)
    if(tapering):
       z2syms,sqlist = mol_info['tapering_info']
       target_sector = mol_info['target_sector']
       X = taper_auxiliary_operators([X],z2syms,target_sector)[0]
    return X

def to_qubit_operators(instructions,operators):
    no = instructions['orbitals']
    op = []
    for j,oj in enumerate(operators):
        h_1 = np.zeros((no,no))
        h_2 = np.zeros((no,no,no,no))
        if(len(oj)==2):
           i,a = oj
           h_1[a,i] =  1.0
           h_1[i,a] = -1.0
        else:
           i,a,j,b = oj
           h_2[a,i,b,j] =  1.0
           h_2[i,a,j,b] = -1.0
        o_h_1_h_2 = to_qubit_op(instructions,h_1,h_2)
        if(not o_h_1_h_2.is_empty()):
           op.append((oj,o_h_1_h_2))
    return op 

def is_self_commuting(P):
    from qiskit.aqua.operators import WeightedPauliOperator
    from qiskit.aqua.operators import commutator
    comm = True
    for cj,Pj in P._paulis:
        Pj = WeightedPauliOperator(paulis=[[1.0,Pj]])
        for ck,Pk in P._paulis:
            Pk = WeightedPauliOperator(paulis=[[1.0,Pk]])
            comm_jk = commutator(Pj,Pk)
            comm_jk = comm_jk.chop(1e-6)
            comm_jk = comm_jk.is_empty()
            if(not comm_jk): comm=False
    return comm

def dec_fun(x,i):
    if(x<=i): return x
    else:     return x-1
 
def build_mask(oper,n):
    num_oper   = len(oper)
    num_par    = num_oper
    param_mask = [x for x in range(num_oper)]
    for i in range(num_oper):
        for j in range(i+1,num_oper):
            oi,oj = oper[i],oper[j]
            if(len(oi)==2):
               if(oi[0]==oj[0]+n and oi[1]==oj[1]+n):
                  num_par       -= 1
                  param_mask[j]  = param_mask[i]
                  param_mask     = [ dec_fun(x,i) for x in param_mask]
               if(oj[0]==oi[0]+n and oj[1]==oi[1]+n):
                  num_par       -= 1
                  param_mask[j]  = param_mask[i]
                  param_mask     = [ dec_fun(x,i) for x in param_mask]
    return num_par,param_mask

class Evangelista(VariationalForm):

    def __init__(self,instructions={},initial_state=None,reps=1):
        super().__init__()
        self._num_qubits     = initial_state.num_qubits
        self._instructions   = instructions
        self._initial_state  = initial_state
        no = instructions['orbitals']//2
        na = instructions['particles'][0]
        nb = instructions['particles'][1]
        self._occ          = [i for i in    range(na)]+[no+i for i in    range(nb)]
        self._vrt          = [a for a in range(na,no)]+[no+a for a in range(nb,no)]
        self._operators    = build_operators(self._occ.copy(),self._vrt.copy(),no)[::-1]
        num_par,param_mask = build_mask(self._operators,no)
        self._operators    = to_qubit_operators(instructions,self._operators)
        for j,(Ej,oj) in enumerate(self._operators):
            print("operator ",j," is ",Ej," is it term-commuting? ",is_self_commuting(oj)," par ",param_mask[j])
        exit()
        num_par,param_mask     = param_to_pauli(self._operators)
        self._num_par          = num_par
        self._param_to_pauli   = param_mask
        self._reps             = reps
        self._num_parameters   = len(num_par)*self._reps
        self._bounds           = [(-np.pi,np.pi)]*self._num_parameters

    def construct_circuit(self,parameters):
        circuit = self._initial_state.copy()
        m       = 0
        circuit.barrier()
        print(parameters,len(parameters),len(self._operators))
        m = 0 
        for r in range(self._reps):
            for j,(Ej,oj) in enumerate(self._operators):
                tj = parameters[m]
                circuit = (1j*oj).evolve(circuit,evo_time=tj,num_time_slices=1,expansion_mode='trotter',expansion_order=1)
                m += 1
        return circuit.copy()


