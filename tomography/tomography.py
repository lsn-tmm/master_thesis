from abc import ABC, abstractmethod
import numpy as np
from utilities import index_to_label,label_to_pauli,label_to_basis,basis_to_index,index_to_basis,index_to_basis
from utilities import construct_measurement_circuit,replace,repeat,post_select

class Tomography(ABC):

    def __init__(self,circuits):
        self.circuits  = circuits
        self.n_qubits  = [ck.num_qubits for k,ck in self.circuits.items()]
        self.n_circuit = len(self.n_qubits)
        self.n_qubits  = self.n_qubits[0]
        self.bloch     = None
        self.construct_paulis()
        super().__init__()

    def noise_amplification(self,repetition_index):
        for k in self.circuits.keys():
            self.circuits[k] = repeat(self.circuits[k],repetition_index)

    def construct_paulis(self):
        self.pauli = [ x                                        for x         in range(4**self.n_qubits)]
        self.pauli = [(x,index_to_label(x,self.n_qubits))       for x         in self.pauli]
        self.pauli = [(x,label_to_pauli(Px),label_to_basis(Px)) for (x,Px)    in self.pauli]
        self.pauli = [(x,Px,Bx,basis_to_index(Bx))              for (x,Px,Bx) in self.pauli]

    def print_paulis(self):
        for (x,Px,Bx,y) in self.pauli:
            print(x,Px,Px._primitive.to_label(),Bx,y,index_to_basis(y,self.n_qubits))

    def print_bloch(self):
        for k in self.circuits.keys():
            for (x,Px,Bx,y) in self.pauli:
                print("<%s|%s|%s> = %.12f +/- %.12f " %  (k,Px._primitive.to_label(),k,self.bloch[k][x,0],self.bloch[k][x,1]))

    def save_calculation(self,filename):
        np.save(filename+'_pauli.npy',self.pauli,allow_pickle=True)
        np.save(filename+'_bloch.npy',self.bloch,allow_pickle=True)

    @abstractmethod
    def compute_bloch_vector(self):
        pass

# -----------------------------------------------------------------------

class Statevector_Tomography(Tomography):

    def get_expectation_value(self,psi,P):
        return np.einsum('i,ij,j->',np.conj(psi),P.to_matrix(),psi).real

    def compute_bloch_vector(self):
        import qiskit
        from qiskit.circuit import Parameter, QuantumCircuit
        from qiskit import Aer

        n_pauli = len(self.pauli)
        self.bloch = {}
        for k in self.circuits.keys():
            self.bloch[k] = np.zeros((n_pauli,2))
            ck = self.circuits[k]
            backend = Aer.get_backend('statevector_simulator')
            result  = qiskit.execute(ck,backend=backend).result()
            ck      = result.get_statevector(ck)
            for (x,Px,Bx,y) in self.pauli:
                self.bloch[k][x,0] = self.get_expectation_value(ck,Px)

# -----------------------------------------------------------------------

class Random_Tomography(Tomography):

    def __init__(self,circuits,shots,optimization_level,device_name=None,initial_layout=None,
                 provider=None,ro_error_mitigation=False,dynamical_decoupling=False,post_select_value=None):
        from qiskit import Aer
        self.shots = shots
        self.optimization_level = optimization_level
        self.device_name = device_name
        self.initial_layout = initial_layout
        self.provider = provider
        self.ro_error_mitigation = ro_error_mitigation
        self.dynamical_decoupling = dynamical_decoupling
        self.post_select_value = post_select_value
        super().__init__(circuits)

    @abstractmethod
    def execute(self,circuits):
        pass

    @abstractmethod
    def compute_probability_distributions(self):
        pass

    def call_post_selection(self):
        # post-selection in the Z basis
        i = basis_to_index([0]*self.n_qubits)
        for k in self.circuits.keys():
            self.p[k][i]=post_select(self.p[k][i],self.post_select_value)

    def compute_bloch_vector(self):
        if(self.post_select_value is not None):
           self.call_post_selection()
        n_pauli = len(self.pauli)
        self.bloch = {}
        for k in self.circuits.keys():
            self.bloch[k] = np.zeros((n_pauli,2))
            for (x,Px,Bx,y) in self.pauli:
                self.bloch[k][x,:] = self.get_expectation_value(index_to_label(x,self.n_qubits),self.p[k][y])

    def get_expectation_value(self,x,counts):
        ave = 0.0
        for (bitstring,count) in counts.items():
            n = sum([int(c) for k,c in enumerate(list(bitstring)) if x[k]!=0])
            ave += count*(-1)**(n%2)
        cnt = sum(counts.values())
        ave = ave/cnt
        var = np.abs(1-ave*ave)/cnt
        return ave,np.sqrt(var)

# -----------------------------------------------------------------------

class QASM_Tomography(Random_Tomography):

    def execute(self,circuits):
        from qiskit import Aer
        from qiskit import execute

        # pure QASM
        if(self.device_name is None):
           job = execute(circuits,Aer.get_backend('qasm_simulator'),shots=self.shots,optimization_level=self.optimization_level)
           return job.result().get_counts()
        else:
           from qiskit.providers.aer.noise import NoiseModel
           hardware = self.provider.get_backend(self.device_name)
           noise_model = NoiseModel.from_backend(hardware)
           coupling_map = hardware.configuration().coupling_map
           basis_gates = noise_model.basis_gates

           # DYN-DEC
           if(self.dynamical_decoupling):
              from qiskit.compiler import transpile
              from runtime_utilities import transpile_circuits_with_dd
              circuits = transpile_circuits_with_dd(circuits,self.shots,self.optimization_level,self.initial_layout,
                                                    self.ro_error_mitigation,self.provider,hardware)
              circuits = transpile(circuits,hardware,scheduling_method="alap")
              LAYOUT = [x for x in range(circuits[0].num_qubits)]
           else:
              LAYOUT = self.initial_layout

           # RO
           if(not self.ro_error_mitigation):
              job = execute(circuits,Aer.get_backend('qasm_simulator'),shots=self.shots,optimization_level=self.optimization_level,
                            initial_layout=LAYOUT,coupling_map=coupling_map,basis_gates=basis_gates,
                            noise_model=noise_model)
              return job.result().get_counts()
           else:
              from qiskit.utils import QuantumInstance
              from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)
              instance = QuantumInstance(backend = Aer.get_backend('qasm_simulator'),
                                         shots = self.shots,
                                         noise_model = noise_model,
                                         coupling_map = coupling_map,
                                         measurement_error_mitigation_cls = CompleteMeasFitter,
                                         optimization_level = self.optimization_level,
                                         initial_layout = LAYOUT)
              job = instance.execute(circuits)
              return job.get_counts()

    def compute_probability_distributions(self):
        n_basis = 3**self.n_qubits
        self.p = {k:[None]*n_basis for k in self.circuits.keys()}
        for k in self.circuits.keys():
            c_list_k  = [construct_measurement_circuit(index_to_basis(y,self.n_qubits),self.circuits[k]) for y in range(n_basis)]
            self.p[k] = self.execute(c_list_k)

# -----------------------------------------------------------------------

class Hardware_Tomography(Random_Tomography):

    def execute(self,circuits):
        from runtime_utilities import execute_with_runtime
        device = self.provider.get_backend(self.device_name)
        return execute_with_runtime(circuits,self.shots,self.optimization_level,self.initial_layout,
                                    self.ro_error_mitigation,self.provider,device,self.dynamical_decoupling)

    def retrieve(self, job_id):
        device = self.provider.get_backend(self.device_name)
        return device.retrieve_job(job_id)

    def max_shots_and_circuits(self):
        device = self.provider.get_backend(self.devname)
        return device.configuration().max_shots,device.configuration().max_experiments

    def construct_tomography_circuits(self,circuits_per_job):
        n_basis  = 3**self.n_qubits
        self.tomography_circuits = []
        index    = 0
        for y in range(n_basis):
            By = index_to_basis(y,self.n_qubits)
            for k in self.circuits.keys():
                ck  = self.circuits[k]
                cky = construct_measurement_circuit(By,ck)
                job_index = index//circuits_per_job
                res_index = index-job_index*circuits_per_job
                self.tomography_circuits.append((k,y,cky,job_index,res_index))
                index += 1

    def submit_probability_distribution_jobs(self,circuits_per_job,logfile_name):
        logfile = open(logfile_name,'w')
        self.construct_tomography_circuits(circuits_per_job)
        job_indices = [job_index for (k,y,cky,job_index,res_index) in self.tomography_circuits]
        job_indices = list(set(job_indices))
        for x in job_indices:
            circuits_x = [cky for (k,y,cky,job_index,res_index) in self.tomography_circuits if job_index==x]
            job = self.execute(circuits_x)
            print("submitted job %s " % job._job_id)
            logfile.write("submitted job %s \n" % job._job_id)
            self.tomography_circuits = [(k,y,cky,replace(name,x,job._job_id),res_index) for (k,y,cky,name,res_index) in self.tomography_circuits]
        for (k,y,cky,job_id,res_index) in self.tomography_circuits:
            print(k,y," ---> ",job_id,res_index)

    def compute_probability_distributions(self,job_id_list,circuits_per_job,mitigated_counts=False):
        from runtime_utilities import retrieve_with_runtime
        self.construct_tomography_circuits(circuits_per_job)
        self.tomography_circuits = [(k,y,cky,job_id_list[x],res_id) for (k,y,cky,x,res_id) in self.tomography_circuits]
        n_basis = 3**self.n_qubits
        self.p = {k:[None]*n_basis for k in self.circuits.keys()}
        job_id_list = [J for (k,y,cky,J,res_index) in self.tomography_circuits]
        job_id_set = []
        for i in job_id_list:
            if i not in job_id_set:
                job_id_set.append(i)
        retrieved_counts = {J:retrieve_with_runtime(J,self.provider,self.n_qubits,self.shots,mitigated_counts) for J in job_id_set}
        for index in range(len(self.tomography_circuits)):
            (k,y,cky,job_id,res_index) = self.tomography_circuits[index]
            self.p[k][y] = retrieved_counts[job_id][res_index]

