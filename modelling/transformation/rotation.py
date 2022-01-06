import cirq
def layerX(bits, params=None, gen_params=None):
        """
        if params = None, a gen_params function should be used
        """
        circuit = cirq.Circuit()
        for i, qubit in enumerate(bits):
            if params is None:
                if gen_params == None:
                    raise ValueError("gen_params function is None. Should pass a param generating function")
                circuit.append(cirq.rx(gen_params())(qubit))
            else:
                circuit.append(cirq.rx(params[i])(qubit))
        return circuit

def layerY(bits, params=None, gen_params=None):
        """
        if params = None, a gen_params function should be used
        """
        circuit = cirq.Circuit()
        for i, qubit in enumerate(bits):
            if params is None:
                if gen_params == None:
                    raise ValueError("gen_params function is None. Should pass a param generating function")
                circuit.append(cirq.ry(gen_params())(qubit))
            else:
                circuit.append(cirq.ry(params[i])(qubit))
        return circuit


def layerZ(bits, params=None, gen_params=None):
    """
    if params = None, a gen_params function should be used
    """
    circuit = cirq.Circuit()
    for i, qubit in enumerate(bits):
        if params is None:
            if gen_params == None:
                raise ValueError("gen_params function is None. Should pass a param generating function")
            circuit.append(cirq.rz(gen_params())(qubit))
        else:
            circuit.append(cirq.rz(params[i])(qubit))
    return circuit