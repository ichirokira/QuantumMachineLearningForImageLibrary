import cirq

def Pyramid_Transform(position_bits, color_bits, num_row_bits, num_col_bits, gen_params=None):

    circuit = cirq.Circuit()
    num_block_bits = min(num_row_bits, num_col_bits)
    for block_bit in range(num_block_bits):
        for r in range(2**(num_row_bits-block_bit)):
            for c in range(2**(num_col_bits-block_bit)):
                r_binary = format(r, "b").zfill(num_row_bits-1)
                c_binary = format(c, "b").zfill(num_col_bits-1)
                controlled_bits = []
                for i, b in enumerate(r_binary):
                    controlled_bits.append(position_bits[i])
                    if b == "0":
                        circuit.append(cirq.X(position_bits[i]))
                for j, b in enumerate(c_binary):
                    controlled_bits.append(position_bits[num_row_bits+j])
                    if b == "0":
                        circuit.append(cirq.X(position_bits[num_row_bits+j]))

                circuit.append(
                    cirq.ry(2*gen_params()).on(*color_bits).controlled_by(*controlled_bits)
                )

                for i, b in enumerate(r_binary):
                    if b == "0":
                        circuit.append(cirq.X(position_bits[i]))
                for j, b in enumerate(c_binary):
                    if b == "0":
                        circuit.append(cirq.X(position_bits[num_row_bits+j]))

    circuit.append(
        cirq.ry(2 * gen_params()).on(*color_bits)
    )
    return circuit