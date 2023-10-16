import numpy as np

def generate_circuit(bs,theta,barriers=False):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(4)
    for i,xi in enumerate(bs):
        if(xi==1): qc.x(i)
    if(barriers): qc.barrier()
    for j,tj in enumerate(theta):
        layer = j//4
        qubit = j%4
        qc.ry(tj,qubit)
        if(qubit==3 and layer<3):
           if(barriers): qc.barrier()
           for i in range(3):
               qc.cx(i,i+1)
           if(barriers): qc.barrier()
    return qc

# angoli delle rotazioni y, presi con lo script "import_parameters.py"
par = np.load('parameters.npy',allow_pickle=True).item()
cir = {}
for species in ['radical','anion']:
    if(species=='anion'): bs = [0,1,1,0]
    else:                 bs = [0,1,0,0]
    for basis in ['aug-cc-pvtz','aug-cc-pvqz']:
        print("species: ",species," basis: ",basis)
        cir[species+'_'+basis] = generate_circuit(bs,par[species+'_'+basis],barriers=False)
        print(cir[species+'_'+basis].draw())

np.save('circuits.npy',cir,allow_pickle=True)

