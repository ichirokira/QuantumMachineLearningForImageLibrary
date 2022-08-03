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

def HE_Superposed(strip_bits, info_bits, entangling_arrangement="chain", type_entangles='cnot', gen_params=None):
        circuit = cirq.Circuit()
        pre_strip_binary = ""
        for k in range(len(strip_bits)):
                pre_strip_binary += "0"

        for n in range(2**len(strip_bits)):
                cur_index_binary = format(n, "b").zfill(len(strip_bits))
                for index_bit in range(len(strip_bits)):
                        if cur_index_binary[index_bit] != pre_strip_binary[index_bit]:
                                circuit.append(cirq.X(strip_bits[index_bit]))

                ent = entangler(info_bits, entangling_arrangement=entangling_arrangement, type_entangles=type_entangles)
                layerx = layerX(info_bits, gen_params=gen_params)
                layery = layerY(info_bits, gen_params=gen_params)
                layerz = layerZ(info_bits, gen_params=gen_params)
                circuit.append(layerx)
                circuit.append(layery)
                circuit.append(layerz)
                circuit.append(ent)

                pre_strip_binary = cur_index_binary

        return circuit

