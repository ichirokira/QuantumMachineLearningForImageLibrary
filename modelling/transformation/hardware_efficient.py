import cirq
from modelling.transformation.entangle import entangler
from modelling.transformation.rotation import layerX, layerY, layerZ

def HE(bits,entangling_arrangement="chain", type_entangles='cnot', gen_params=None):

        circuit = cirq.Circuit()
        ent = entangler(bits, entangling_arrangement=entangling_arrangement,type_entangles=type_entangles)
        layerx = layerX(bits, gen_params=gen_params)
        layery = layerY(bits, gen_params=gen_params)
        layerz = layerZ(bits, gen_params=gen_params)
        circuit.append(layerx)
        circuit.append(layery)
        circuit.append(layerz)
        circuit.append(ent)
        return circuit
def HE_Color_Independence(position_bits, color_bits, entangling_arrangement="chain", type_entangles='cnot', gen_params=None):
        circuit = cirq.Circuit()
        ent = entangler(position_bits, entangling_arrangement=entangling_arrangement, type_entangles=type_entangles)
        layerx = layerX(position_bits, gen_params=gen_params)
        layery = layerY(position_bits, gen_params=gen_params)
        layerz = layerZ(position_bits, gen_params=gen_params)
        circuit.append(layerx)
        circuit.append(layery)
        circuit.append(layerz)
        circuit.append(ent)
        for c in color_bits:
            circuit.append(
                cirq.ry(gen_params()).on(c).controlled_by(*position_bits))
        return circuit