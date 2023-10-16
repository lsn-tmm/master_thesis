import numpy as np

def int2bas(n, b, nbit=None):
    if n == 0:
        if nbit == None:
            return [0]
        else:
            return [0] * nbit
    x = []
    while n:
        x.append(int(n % b))
        n //= b
    if nbit == None:
        return x[::-1]
    else:
        return [0]*(nbit-len(x)) + x[::-1]

def bas2int(x, b):
    nbit = len(x)
    z = [b**(nbit-i-1) for i in range(nbit)]
    return np.dot(z, x)

def f(x):
    if(x==1 or x==2): return x
    return 0

# ---------------------------------------

def index_to_label(mu,n):
    return int2bas(mu,4,n)

def pauli_label_to_index(lab):
    x = {'I':0,'X':1,'Y':2,'Z':3}
    lab_int = [x[k] for k in lab]
    return bas2int(lab_int,4)

def label_to_pauli(x):
    from qiskit.opflow import I,X,Y,Z
    one_qubit_pauli = [I,X,Y,Z]
    P = one_qubit_pauli[x[0]]
    for i in range(1,len(x)):
        P = P^one_qubit_pauli[x[i]]
    return P

def label_to_basis(x):
    return [f(xi) for xi in x]

def basis_to_index(B):
    return bas2int(B,3)

def index_to_basis(nu,n):
    return int2bas(nu,3,n)

def construct_measurement_circuit(basis,qc):
    from qiskit import QuantumRegister,QuantumCircuit,ClassicalRegister
    qr = qc.qregs[0]
    cr = ClassicalRegister(qr.size)
    qc_M = QuantumCircuit(qr,cr)
    qc_M= qc_M.compose(qc)
    for i,bi in enumerate(basis[::-1]):
        if(bi==1):
           qc_M.h(i)
        if(bi==2):
           qc_M.sdg(i)
           qc_M.h(i)
    for i in range(qc_M.num_qubits):
        qc_M.measure(i,i)
    return qc_M

def replace(a,b,c):
    if(a==b): return c
    return a

def repeat(qc,x):
    from qiskit import QuantumRegister,QuantumCircuit,ClassicalRegister
    qr = qc.qregs[0]
    qc_B = QuantumCircuit(qc.num_qubits)
    for gate in qc:
        gate_name = gate[0].name
        qubits = [q.index for q in gate[1]]
        if(gate_name=='hopgate'):
           gate_scaled = gate[0].copy()
           gate_scaled.params = [a/x for a in gate_scaled.params]
           for i in range(x): 
               qc_B.append(gate_scaled,qubits)
        elif(gate_name=='cx'):
           for i in range(x):
               qc_B.append(gate[0],qubits); qc_B.barrier()
        else:
           qc_B.append(gate[0],qubits)
    return qc_B

# ---------------------------------------

def post_select(data,value):
    data = {x:data[x] for x in data.keys() if sum([int(y) for y in str(x)])==value}
    return data

