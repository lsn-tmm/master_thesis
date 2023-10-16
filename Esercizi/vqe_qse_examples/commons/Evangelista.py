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
       loop = (len(occ)>0)
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

def is_aa_bb(ox,oy,n):
    B = False
    if(len(ox)==len(oy) and len(ox)==2):
       i,a = ox
       j,b = oy
       s1  = i//n
       s2  = j//n
       if(s1!=s2 and i%n==j%n and a%n==b%n): B=True
    return B

def sort_pair(p,q):
    return min(p,q),max(p,q)

def is_aaaa_bbbb(ox,oy,n):
    B = False
    if(len(ox)==len(oy) and len(ox)==4):
       i,a,j,b = ox
       k,c,l,d = oy
       i,j = sort_pair(i,j)
       k,l = sort_pair(k,l)
       a,b = sort_pair(a,b)
       c,d = sort_pair(c,d)
       s1 = i//n
       t1 = j//n
       s2 = k//n
       t2 = l//n
       if(s1==t1 and s2==t2 and s1!=s2 and i%n==k%n and j%n==l%n and a%n==c%n and b%n==d%n): B=True
    return B

def remove_redundancies(oper,n):
    n_oper   = len(oper)
    num_par  = 0
    mask_par = [-1]*n_oper
    for x in range(n_oper):
        for y in range(x+1,n_oper):
            ox,Ex = oper[x]
            oy,Ey = oper[y]
            if(is_aa_bb(ox,oy,n) or is_aaaa_bbbb(ox,oy,n)):
               mask_par[x] = num_par
               mask_par[y] = num_par
               num_par    += 1
    for k in range(n_oper):
        if(mask_par[k]==-1):
           mask_par[k] = num_par 
           num_par    += 1
    return num_par,mask_par

class Evangelista(VariationalForm):

    def __init__(self,instructions={},initial_state=None,reps=1):
        super().__init__()
        self._num_qubits     = initial_state.num_qubits
        self._instructions   = instructions
        self._initial_state  = initial_state
        no = instructions['orbitals']//2
        na = instructions['particles'][0]
        nb = instructions['particles'][1]
        self._occ       = [i for i in    range(na)]+[no+i for i in    range(nb)]
        self._vrt       = [a for a in range(na,no)]+[no+a for a in range(nb,no)]
        self._operators = build_operators(self._occ.copy(),self._vrt.copy(),no)[::-1]
        self._operators = to_qubit_operators(instructions,self._operators)
        #for j,(Ej,oj) in enumerate(self._operators):
        #    print("operator ",j," is ",Ej," is it term-commuting? ",is_self_commuting(oj))
        #num_par,mask_par     = remove_redundancies(self._operators,no)
        num_par  = len(self._operators)
        mask_par = [x for x in range(num_par)]
        self._num_par        = num_par
        self._mask_par       = mask_par
        self._reps           = reps
        self._num_parameters = self._num_par*self._reps
        self._bounds         = [(-np.pi,np.pi)]*self._num_parameters

    def construct_circuit(self,parameters):
        circuit = self._initial_state.copy()
        m       = 0
        circuit.barrier()
        for r in range(self._reps):
            for j,(Ej,oj) in enumerate(self._operators):
                m       = self._mask_par[j]+r*(self._num_par)
                print("reps, oper, par ",r,j,m)
                tj      = parameters[m]
                circuit = (1j*oj).evolve(circuit,evo_time=tj,num_time_slices=1,expansion_mode='trotter',expansion_order=1)
        return circuit.copy()

