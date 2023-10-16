import numpy as np

def generate_circuit(bs,theta,barriers=False):
    from qiskit import QuantumCircuit
    n = len(bs)
    qc = QuantumCircuit(n)
    depth = 3
    for i,xi in enumerate(bs):
        if(xi==1): qc.x(i)
    if(barriers): qc.barrier()
    for j,tj in enumerate(theta):
        layer = j//n
        qubit = j%n
        qc.ry(tj,qubit)
        if(qubit==n-1 and layer<depth):
           if(barriers): qc.barrier()
           for i in range(n-1):
               qc.cx(i,i+1)
           if(barriers): qc.barrier()
    return qc

# angoli delle rotazioni y, presi con lo script "import_parameters.py"
par = np.load('parameters.npy',allow_pickle=True).item()
cir = {}
for species in ['radical','anion']:
    if(species=='anion'): bs = [1,0,1,1,1,0]
    else:                 bs = [1,0,1,1,0,0]
    for basis in ['aug-cc-pvqz']:
        for R in ['0.94']:
            print("species: ",species," basis: ",basis)
            cir[species+'_'+basis+'_'+R] = generate_circuit(bs,par[species+'_'+basis+'_'+R],barriers=False)
            print(cir[species+'_'+basis+'_'+R].draw())
         
np.save('circuits.npy',cir,allow_pickle=True)

#anion bs 101110
#radical bs 101100
