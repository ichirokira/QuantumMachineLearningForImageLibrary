import cirq
from modelling.transformation.entangle import entangler
from modelling.transformation.rotation import layerX, layerY, layerZ

def Farhi(bits,readout, gen_params=None):
    circuit = cirq.Circuit()
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))
    for i, qubit in enumerate(bits):
        circuit.append(cirq.XX(qubit, readout)**gen_params())
    for i, qubit in enumerate(bits):
        circuit.append(cirq.ZZ(qubit, readout)**gen_params())
    circuit.append((cirq.H(readout)))

    return circuit