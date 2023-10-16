# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Code developed by Youngseok Kim and modified by Riddhi Gupta


RES=0
# Youngseok: RES=10
# it used to be utilized for qiskit-direct, where singles are length of 10pts in most of the backends.
# The production user (IQX provider) actually sees it as 10*16 pts, so that hard-coded value may not be relevant to you if you are using IQX
# Riddhi: RES=0
# I don't think RES is being used a resolution or single lengths - it's being used as a qubit index number
# RES = 10*16 gives error:
# ~/opt/anaconda3/envs/ameba_m3/lib/python3.9/site-packages/qiskit/transpiler/instruction_durations.py in get(self, inst, qubits, unit)
#     177             return self._get(inst_name, qubits, unit)
#     178         except TranspilerError as ex:
# --> 179             raise TranspilerError(
#     180                 f"Duration of {inst_name} on qubits {qubits} is not found."
#     181             ) from ex

# TranspilerError: 'Duration of x on qubits [160] is not found.'
# Paul nation: RES= 10*16
# RES is resolution of wave in analogue waveform generator 

from collections import defaultdict
from typing import List
import numpy as np



from qiskit import QuantumCircuit, QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import Delay
from qiskit.circuit.library import U3Gate, XGate
from qiskit.transpiler.passes import Optimize1qGates
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.circuit.gate import Gate


class PulseAlignment(TransformationPass):

    def __init__(self):
        """ASAPSchedule initializer.
        Hack made by Riddhi gupta
        """
        super().__init__()

    def run(self, dag):
        """ Run and check delays
        """
        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('DD runs on physical circuits only')

        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        for nd in dag.topological_op_nodes():
            if nd.op.duration is not None:
                if np.mod(int(nd.op.duration),16) != 0.:
                    print("Instr type :: ", type(nd.op), np.mod(int(nd.op.duration),16))
                    nd.op.duration = 16*int(round(nd.op.duration/16.))
            new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs)
        return new_dag

def apply_pulse_alignment_condition(duration_float, istype=None, mod=16):    
    '''
    Hack made by Riddhi Gupta
    '''
    if isinstance(duration_float, U3Gate) or istype==None:
        return duration_float #TODO: CNOT
    
    if isinstance(duration_float, Delay):
        duration_float = duration_float.duration
    
    remainder = np.mod(int(duration_float),mod)
    if remainder > 0:
        duration = duration_float - remainder
    else:
        duration = duration_float
    if np.mod(duration,mod) == 0:
        if istype == 'int':
            return duration
        else:
            return Delay(duration)
    else:
        raise RuntimeError


"""Dynamical Decoupling insertion pass."""

class InsertDD(TransformationPass):
    """DD insertion pass."""

    def __init__(self, durations, dd_sequence, blacklist=[]):
        """ASAPSchedule initializer.
        Args:
            durations (InstructionDurations): Durations of instructions to be used in scheduling
            dd_sequence (list[str]): sequence of gates to apply in idle spots
        """
        super().__init__()
        self._durations = durations
        self._dd_sequence = dd_sequence
        self._blacklist = blacklist

    def run(self, dag):
        """Run the InsertDD pass on `dag`.
        Args:
            dag (DAGCircuit): a scheduled DAG.
        Returns:
            DAGCircuit: equivalent circuit with no extra delays but DD where possible.
        Raises:
            TranspilerError: if the circuit is not mapped on physical qubits.
        """
        # print(type(dag))
        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('DD runs on physical circuits only')

        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        for nd in dag.topological_op_nodes():
            if isinstance(nd.op, Delay):
                qubit = nd.qargs[0]
                pred = next(dag.predecessors(nd))
                succ = next(dag.successors(nd))
                # print("Checking delay instances")
                unchanged_delay = apply_pulse_alignment_condition(nd.op, istype='Delay')
                if pred.type == 'in':  # discount initial delays
                    new_dag.apply_operation_back(unchanged_delay, nd.qargs, nd.cargs) 
                elif qubit.index in self._blacklist:
                    new_dag.apply_operation_back(unchanged_delay, nd.qargs, nd.cargs) 
                else:
                    _dd_sequence, slack = self.estimate_sequence(nd.op.duration)
                    if len(_dd_sequence) != len(self._dd_sequence):
                        print('Q{} dd seq is modified to {}'.format(qubit, ','.join(_dd_sequence)))
                    #dd_sequence_duration = sum([self._durations.get(g, qubit) for g in self._dd_sequence])
                    #slack = nd.op.duration - dd_sequence_duration
                    # absorb any double counted delays
                    #if isinstance(succ.op, Delay):
                    #    if succ.op.duration >= 0:
                    #        slack +=  succ.op.duration
                    #        succ.op.duration = 0 
                    if slack <= 0:     # dd doesn't fit
                        # print("DD doesnt fit")
                        unchanged_delay = apply_pulse_alignment_condition(nd.op, istype='Delay')
                        new_dag.apply_operation_back(unchanged_delay, nd.qargs, nd.cargs)
                    else:
                        if np.mod(len(_dd_sequence), 2) == 0:
                            nseq=len(_dd_sequence)
                            interval = apply_pulse_alignment_condition(slack/nseq, istype='int', mod=32) #int(slack/nseq)
                            begin =  apply_pulse_alignment_condition(interval/2, istype='int', mod=16) # int(interval/2)
                            new_dag.apply_operation_back(Delay(begin), [qubit])
                            for i in range(nseq-1):
                                insert_gate(new_dag, qubit, _dd_sequence[i])
                                new_dag.apply_operation_back(Delay(interval), [qubit])
                            insert_gate(new_dag, qubit, _dd_sequence[-1])
                            new_dag.apply_operation_back(Delay(begin), [qubit])
                            #print('{} applied'.format(','.join(_dd_sequence)))
                        #if len(self._dd_sequence) == 2:
                        #    begin = int(slack/4)
                        #    new_dag.apply_operation_back(Delay(begin), [qubit])
                        #    insert_gate(new_dag, qubit, _dd_sequence[0])
                        #    new_dag.apply_operation_back(Delay(slack-2*begin), [qubit])
                        #    insert_gate(new_dag, qubit, _dd_sequence[1])
#                            new_dag.apply_operation_back(XmGate(), [qubit])
                        #    new_dag.apply_operation_back(Delay(begin), [qubit])
                        elif len(_dd_sequence) == 1: 
                            if str(succ.type) == 'op' and isinstance(succ.op, U3Gate): # only do it if the rest can absorb
                            #print(succ.op); print(succ.op.params)
                                begin = apply_pulse_alignment_condition(slack/2, istype='int') # int(slack/2)
                                new_dag.apply_operation_back(Delay(begin), [qubit])
                                insert_gate(new_dag, qubit, _dd_sequence[0])
#                            new_dag.apply_operation_back(XpGate(), [qubit])
                                diff = apply_pulse_alignment_condition(slack-begin, istype='int')
                                new_dag.apply_operation_back(Delay(diff), [qubit])
                                # absorb an X gate into the successor (from left in circuit)
                                theta, phi, lam = succ.op.params
                                succ.op.params = Optimize1qGates.compose_u3(theta, phi, lam, np.pi, 0, np.pi)
                            else:
                                unchangedop = apply_pulse_alignment_condition(nd.op)
                                new_dag.apply_operation_back(unchangedop, nd.qargs, nd.cargs)
                         
                        else:
                            raise TranspilerError('whats this sequence you tryna do?')
            else:
                unchangedop = apply_pulse_alignment_condition(nd.op)
                new_dag.apply_operation_back(unchangedop, nd.qargs, nd.cargs)

        #new_dag.qubit_time_available = dag.qubit_time_available
        new_dag.duration = dag.duration
        return new_dag
    
    def estimate_slack(self, sequence, op_duration):
        # print("self._durations", self._durations)
        sequence_duration = sum([self._durations.get(g, RES) for g in sequence])
        return op_duration - sequence_duration
        
    def estimate_sequence(self, op_duration):
        """ try n, n-2, n-4, ..., 1 """
        ndd=len(self._dd_sequence)
        if not (ndd == 1 or np.mod(ndd,2)==0):
            raise TranspilerError('dd sequence must be 1, or multiple of 2. But it is {}'.format(ndd))
        if ndd == 1 or ndd == 2:
            slack_duration = self.estimate_slack(self._dd_sequence, op_duration)
            return self._dd_sequence, slack_duration
        for i in range(int(ndd/2))[::-1]:
            k=i+1
            seq=[self._dd_sequence[j] for j in range(k)] # first few
            seq.extend([self._dd_sequence[-(j+1)] for j in range(k)][::-1]) # last few
            # print("self._durations", self._durations)
            sequence_duration = sum([self._durations.get(g, RES) for g in seq])
            slack_duration = op_duration - sequence_duration
            if slack_duration > sequence_duration:
                #print('sequence {} can be applied for op duraiton {}'.format(','.join(seq), op_duration))
                return seq, slack_duration
            #else:
            #    print('sequence {} is modified'.format(','.join(seq)))
        return seq, slack_duration
            
        
    
def insert_gate(new_dag, qubit, gate):
    if gate == 'x':
        new_dag.apply_operation_back(XGate(), [qubit])
    if gate == 'xp':
        new_dag.apply_operation_back(XpGate(), [qubit])
    if gate == 'xm':
        new_dag.apply_operation_back(XmGate(), [qubit])
    if gate == 'yp':
        new_dag.apply_operation_back(YpGate(), [qubit])
    if gate == 'ym':
        new_dag.apply_operation_back(YmGate(), [qubit])

class XpGate(Gate):
    r"""The single-qubit Pauli-X gate (:math:`\sigma_x`).
             ┌────┐
        q_0: ┤ Xp ├
             └────┘
    """

    def __init__(self, label=None):
        """Create new X gate."""
        super().__init__('xp', 1, [], label=label)

    def _define(self):
        """
        gate xp a { u3(pi,0,pi) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        # from .u3 import U3Gate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (U3Gate(np.pi, 0, np.pi), [q[0]], [])
        ]
        qc._data = rules
        self.definition = qc

    def inverse(self):
        r"""Return inverted X gate (itself)."""
        return XmGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the X gate."""
        return np.array([[0, 1],
                            [1, 0]], dtype=complex)


class XmGate(Gate):
    r"""The single-qubit Pauli-X gate (:math:`\sigma_x`).
             ┌────┐
        q_0: ┤ Xm ├
             └────┘
    """

    def __init__(self, label=None):
        """Create new X gate."""
        super().__init__('xm', 1, [], label=label)

    def _define(self):
        """
        gate xm a { u3(pi,0,pi) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        # from .u3 import U3Gate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (U3Gate(np.pi, 0, np.pi), [q[0]], [])
        ]
        qc._data = rules
        self.definition = qc

    def inverse(self):
        r"""Return inverted X gate (itself)."""
        return XpGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the X gate."""
        return np.array([[0, 1],
                            [1, 0]], dtype=complex)

class YpGate(Gate):
    r"""The single-qubit Pauli-Y gate (:math:`\sigma_x`).
             ┌────┐
        q_0: ┤ Yp ├
             └────┘
    """

    def __init__(self, label=None):
        """Create new Y gate."""
        super().__init__('yp', 1, [], label=label)

    def _define(self):
        """
        gate yp a { u3(pi,0,pi) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        # from .u3 import U3Gate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (U3Gate(np.pi, np.pi / 2, np.pi / 2), [q[0]], [])
        ]
        qc._data = rules
        self.definition = qc

    def inverse(self):
        r"""Return inverted Y gate (itself)."""
        return YmGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the Y gate."""
        return np.array([[0, -1j],
                            [1j, 0]], dtype=complex)


class YmGate(Gate):
    r"""The single-qubit Pauli-Y gate (:math:`\sigma_y`).
             ┌────┐
        q_0: ┤ Ym ├
             └────┘
    """

    def __init__(self, label=None):
        """Create new Y gate."""
        super().__init__('ym', 1, [], label=label)

    def _define(self):
        """
        gate ym a { u3(pi,0,pi) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        # from .u3 import U3Gate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (U3Gate(np.pi, np.pi / 2, np.pi / 2), [q[0]], [])
        ]
        qc._data = rules
        self.definition = qc

    def inverse(self):
        r"""Return inverted Y gate (itself)."""
        return YpGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the Y gate."""
        return np.array([[0, -1j],
                            [1j, 0]], dtype=complex)



# class CheckDelays(TransformationPass):

#     def __init__(self):
#         """ASAPSchedule initializer.
#         """
#         super().__init__()

#     def run(self, dag):
#         """ Run and check delays
#         """
#         if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
#             raise TranspilerError('DD runs on physical circuits only')

#         new_dag = DAGCircuit()
#         for qreg in dag.qregs.values():
#             new_dag.add_qreg(qreg)
#         for creg in dag.cregs.values():
#             new_dag.add_creg(creg)

#         for nd in dag.topological_op_nodes():
#             if isinstance(nd.op, Delay):
#                 print("Delay instance")
#                 unchanged_delay = apply_pulse_alignment_condition(nd.op, istype='Delay')
#                 new_dag.apply_operation_back(unchanged_delay, nd.qargs, nd.cargs) 
                
#             else:
#                 print("Op instance")
#                 unchangedop = apply_pulse_alignment_condition(nd.op)
#                 new_dag.apply_operation_back(unchangedop, nd.qargs, nd.cargs)

#         #new_dag.qubit_time_available = dag.qubit_time_available
#         # new_dag.duration = dag.duration
#         return new_dag
