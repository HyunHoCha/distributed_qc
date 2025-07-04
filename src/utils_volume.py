import random
import math
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_unitary, Operator
from qiskit.transpiler import PassManager
import numpy as np
# from qiskit.transpiler.passes import Optimize1qGatesDecomposition
# from qiskit.quantum_info import OneQubitEulerDecomposer
# from qiskit.circuit.library import UGate
from qiskit.circuit.library import DraperQFTAdder, QFT, RGQFTMultiplier
import sys
import warnings

# python3 utils_volume.py

warnings.filterwarnings("ignore", category=DeprecationWarning)


def su4():
    U = random_unitary(4).data
    det = np.linalg.det(U)
    root_4th = np.power(det, 1 / 4)
    su4_mat = U / root_4th
    return su4_mat


def quantum_volume(n, d):
    # n: number of qubits
    # d: depth

    qubit_indices = list(range(n))

    num_pairs = math.floor(n / 2)

    qc = QuantumCircuit(n)

    for l in range(d):
        random.shuffle(qubit_indices)

        for pair_idx in range(num_pairs):
            q1 = qubit_indices[2 * pair_idx]
            q2 = qubit_indices[2 * pair_idx + 1]

            u_gate = Operator(su4())
            qc.unitary(u_gate, [q1, q2])

    return qc


def merge_unary_gates(circuit):
    _circuit = [circuit[0] + (0,)]

    t = 1
    for qubit_set in circuit:
        if qubit_set == _circuit[-1][:-1]:
            pass
        else:
            _circuit.append(qubit_set + (t,))
            t += 1

    return _circuit


def draper_qft_adder(n):
    # Unary gate: (q, t)
    # Binary (CZ) gate: (q1, q2, t)
    
    circuit = []
    t = 0

    draper_circuit = DraperQFTAdder(n)
    decomposed_draper_circuit = draper_circuit.decompose()
    for instruction, qargs, cargs in decomposed_draper_circuit.data:
        if instruction.name == 'QFT' or instruction.name == 'IQFT':
            qft_circuit = QFT(len(qargs)).decompose()
            for sub_instruction, sub_qargs, sub_cargs in qft_circuit.data:
                if sub_instruction.name == 'cp':
                    circuit.append((sub_qargs[0]._index + n, sub_qargs[1]._index + n, t))
                    t += 1
                elif sub_instruction.name == 'h':
                    circuit.append((sub_qargs[0]._index + n, t))
                    t += 1
        else:
            circuit.append((qargs[0]._index, 2 * n - 1 - qargs[1]._index, t))
            t += 1

    return circuit


def rgqft_adder(n):
    # Unary gate: (q, t)
    # Binary (CZ) gate: (q1, q2, t)

    rgqft_circuit = RGQFTMultiplier(n)
    decomposed_rgqft_circuit = rgqft_circuit.decompose()

    basis_gates = ['u1', 'u2', 'u3', 'cp']

    transpiled_rgqft_circuit = transpile(decomposed_rgqft_circuit, basis_gates=basis_gates, optimization_level=3)

    circuit = []
    t = 0
    reg_dict = {'a': 0, 'b': n, 'out': 2 * n}

    for instruction, qargs, cargs in transpiled_rgqft_circuit.data:
        if instruction.name == 'cp':
            cp_indices = sorted([reg_dict[qargs[0]._register.name] + qargs[0]._index,
                                reg_dict[qargs[1]._register.name] + qargs[1]._index])
            circuit.append((cp_indices[0], cp_indices[1], t))
            t += 1
        else:  # instruction.name in ['u1', 'u2', 'u3']
            u_index = reg_dict[qargs[0]._register.name] + qargs[0]._index
            circuit.append((u_index, t))
            t += 1

    return circuit
