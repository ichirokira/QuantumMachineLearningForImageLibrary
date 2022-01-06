import cirq
import numpy as np
import itertools

def entangler(bits, entangling_arrangement="chain", type_entangles="cnot"):
        circuit = cirq.Circuit()
        if entangling_arrangement == "chain":
            index = [[2 * j, 2 * j + 1] for j in range(len(bits) // 2)] + [
                [2 * j + 1, 2 * j + 2] for j in range((len(bits) - 1) // 2)]
        elif entangling_arrangement == "all":
            index = list(itertools.chain(
                *[[np.random.permutation([i, j]) for j in range(i + 1, len(bits))] for i
                  in range(len(bits) - 1)]))
        for this_bits, next_bits in index:
            if this_bits < next_bits:
                a = this_bits
                b = next_bits
            else:
                a = next_bits
                b = this_bits
            if type_entangles == 'cnot':
                circuit.append(cirq.CNOT(bits[a], bits[b]))
            elif type_entangles == 'cphase':
                circuit.append(cirq.cphase(np.pi)(bits[a], bits[b]))
            elif type_entangles == "sqrtiswap":
                circuit.append(cirq.ISwapPowGate(exponent=1 / 2)(bits[a], bits[b]))
        return circuit

