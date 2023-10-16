def execute_with_runtime(qc,shots,opt,layout,ro,provider,backend,dd):
    if(dd):
       return execute_with_runtime_with_dd(qc,shots,opt,layout,ro,provider,backend)
    else:
       return execute_with_runtime_without_dd(qc,shots,opt,layout,ro,provider,backend)

# -----

def execute_with_runtime_without_dd(qc,shots,opt,layout,ro,provider,backend):
    from qiskit.providers.ibmq import RunnerResult
    from qiskit import IBMQ, QuantumCircuit
    from qiskit.visualization import plot_histogram
    from qiskit.quantum_info import hellinger_fidelity
    from qiskit.ignis.mitigation.expval import expectation_value
    
    program_inputs = {
        'circuits': qc,
        'shots': shots,
        'optimization_level': opt,
        'initial_layout': layout,
        'measurement_error_mitigation': ro
    }
    options = {'backend_name': backend.name()}
    job = provider.runtime.run(program_id="circuit-runner",
                               options=options,
                               inputs=program_inputs,
                               result_decoder=RunnerResult)
    return job

# -----

def transpile_circuits_with_dd(qc,shots,opt,layout,ro,provider,backend):
    from qiskit.providers.ibmq import RunnerResult
    from qiskit import IBMQ, QuantumCircuit
    from qiskit.visualization import plot_histogram
    from qiskit.quantum_info import hellinger_fidelity
    from qiskit.ignis.mitigation.expval import expectation_value

    from qiskit.compiler import transpile
    from qiskit.transpiler import PassManager, InstructionDurations
    from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling
    from qiskit.circuit.library import XGate

    # ----------------- creation of pass manager
    def add_DD(circuits_transpiled, backend, seq=['xp', 'xm']):
        import copy
        from from_riddhi import InsertDD
        durations = InstructionDurations.from_backend(backend)
        xgate_duration = durations.get(XGate(),[0])
        new_durations = [['xp', None, xgate_duration],['xm', None, xgate_duration]]
        durations.update(new_durations)
        pm = PassManager(InsertDD(durations=durations, dd_sequence=seq))
        circuits_dd = pm.run(circuits_transpiled)
        circuits_dd = transpile(copy.deepcopy(circuits_dd), backend=backend, scheduling_method='alap')
        return circuits_dd

    # ----------------- transpilation of circuit with backend gates and layout
    qc = transpile(qc,basis_gates=backend.configuration().basis_gates,initial_layout=layout)

    # ----------------- running pass manager
    qc = add_DD(qc,backend)

    # ----------------- transpilation of circuit with backend gates and layout
    qc = transpile(qc,backend)
    return qc

def execute_with_runtime_with_dd(qc,shots,opt,layout,ro,provider,backend):
    from qiskit.providers.ibmq import RunnerResult
    from qiskit import IBMQ, QuantumCircuit
    from qiskit.visualization import plot_histogram
    from qiskit.quantum_info import hellinger_fidelity
    from qiskit.ignis.mitigation.expval import expectation_value
    
    from qiskit.compiler import transpile
    from qiskit.transpiler import PassManager, InstructionDurations
    from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling
    from qiskit.circuit.library import XGate
   
    qc = transpile_circuits_with_dd(qc,shots,opt,layout,ro,provider,backend)

    # ----------------- inputs of runtime program
    program_inputs = {'circuits': transpile(qc,backend,scheduling_method="alap"),
                      'shots': shots,
                      'optimization_level': 0,
                      'measurement_error_mitigation': ro}
    options = {'backend_name': backend.name()}
    job = provider.runtime.run(program_id="circuit-runner",
                               options=options,
                               inputs=program_inputs,
                               result_decoder=RunnerResult)
    return job

# -----

def retrieve_with_runtime(job_id,provider,n,shots=1,mitigated_counts=False):
    from qiskit.providers.ibmq import RunnerResult
    from qiskit import IBMQ, QuantumCircuit
    from qiskit.visualization import plot_histogram
    from qiskit.quantum_info import hellinger_fidelity
    from qiskit.ignis.mitigation.expval import expectation_value
    
    def _hex_to_bin(hexstring):
        return str(bin(int(hexstring, 16)))[2:]
    
    def _pad_zeros(bitstring, memory_slots):
        return format(int(bitstring, 2), '0{}b'.format(memory_slots))

    print("retrieving job ",job_id)
    
    job = provider.runtime.job(job_id)
    res = job.result()
    print('Risultato: ', res)
    #exit()

    if(mitigated_counts):
       res = [R['data']['quasiprobabilities'] for R in res['results']]
       res = [{c:shots*R[c] for c in R.keys()} for R in res]
    else:
       res = [R['data']['counts'] for R in res['results']]
    res = [{_pad_zeros(_hex_to_bin(k),n):R[k] for k in R.keys()} for R in res] 

    print("retrieved job ",job_id)
    return res

