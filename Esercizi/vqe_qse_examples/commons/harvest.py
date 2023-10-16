import numpy as np
import functools

def measure_operators(operators,wfn_circuit,instance):
    circuits = []
    for idx,oper in enumerate(operators):
        if(not oper.is_empty()):
           circuit = oper.construct_evaluation_circuit(
                     wave_function               = wfn_circuit,
                     statevector_mode            = instance.is_statevector,
                     use_simulator_snapshot_mode = instance.is_statevector,
                     circuit_name_prefix         = 'oper_'+str(idx))
           circuits.append(circuit)
    if circuits:
        to_be_simulated_circuits = \
            functools.reduce(lambda x, y: x + y, [c for c in circuits if c is not None])
        result = instance.execute(to_be_simulated_circuits)
    # ---
    results_list = []
    for idx,oper in enumerate(operators):
        if(not oper.is_empty()):
           mean,std = oper.evaluate_with_result(
                      result = result,statevector_mode = instance.is_statevector,
                      use_simulator_snapshot_mode = instance.is_statevector,
                      circuit_name_prefix         = 'oper_'+str(idx))
           if(np.abs(np.imag(mean))>1e-4): print("attention: IMAG",mean)
           results_list.append([np.real(mean),np.abs(std)])
        else:
           results_list.append([0,0])
    # ---
    return results_list

