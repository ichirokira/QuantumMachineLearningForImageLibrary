import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import math
import sympy
import numpy as np
import seaborn as sns
import collections

class QuanvolutionFRQI(tf.keras.layers.Layer):
    def __init__(self, config, filter_size, **kwangs):
        super(QuanvolutionFRQI, self).__init__(**kwangs)
        self.filter_size = filter_size
        self.block_count = config.NUM_BLOCKS
        self.encoder_type = config.ENCODER
        self.config = config

        self.num_qubits_row = (math.ceil(math.log2(self.filter_size[0])))
        self.num_qubits_col = (math.ceil(math.log2(self.filter_size[1])))
        self.image_color_base = 1
        self.learning_params = []
        self.layer_gen()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filter_size': self.filter_size,
            'block_count': self.block_count,
        })
        return config

    def FRQI(self, bits, params):

        num_qubits = len(bits)

        pre_position_binary = ""
        for i in range(num_qubits - self.image_color_base):
            pre_position_binary += "0"
        circuit = cirq.Circuit()
        for i in range(num_qubits - self.image_color_base):
            circuit.append(cirq.H(bits[i]))
        for i in range((self.filter_size[0])):
            for j in range((self.filter_size[1])):
                cur_position_binary = format(i, "b").zfill(self.num_qubits_row) + format(j, "b").zfill(
                    self.num_qubits_col)
                for b in range(num_qubits - self.image_color_base):
                    if cur_position_binary[b] != pre_position_binary[b]:
                        circuit.append(cirq.X(bits[b]))

                circuit.append(cirq.ry(2 * params[self.image_color_base * (i * self.filter_size[1] + j)]).on(
                        bits[-1]).controlled_by(*bits[:-self.image_color_base]))

                pre_position_binary = cur_position_binary
        return circuit

    def pqc_block(self, qubits):
        circuit1 = cirq.Circuit()

        index = [[i, ((i+1) % len(qubits))] for i in range(len(qubits))]
        for this_bits, next_bits in index:
            #print(this_bits, next_bits, len(qubits))
            circuit1.append(cirq.rz(self._get_new_param()).on(qubits[next_bits]).controlled_by(qubits[this_bits]))

        circuit2 = cirq.Circuit()
        for qubit in qubits:
            circuit2.append(cirq.ry(self._get_new_param())(qubit))

        circuit = cirq.Circuit()
        circuit.append(circuit1)
        circuit.append(circuit2)
        return circuit

    def convolution_circuit(self, qubits):
        circuit = cirq.Circuit()

        for i in range(self.block_count):
            circuit.append(self.pqc_block(qubits))

        return circuit

    def layer_gen(self):
        if self.encoder_type == "FRQI":
            qubits = cirq.GridQubit.rect(1, self.num_qubits_col+self.num_qubits_row+1)
        input_params = [sympy.symbols("a%d" % i) for i in range(self.filter_size[0] * self.filter_size[1])]
        if self.encoder_type == "FRQI":
            cir_e = self.FRQI(qubits, input_params)
        #cir_e = self.encoder_circuit(qubits, input_params, self.encoder_type)
        cir_c = self.convolution_circuit(qubits)

        full_circuit = cirq.Circuit()
        full_circuit.append(cir_e)
        full_circuit.append(cir_c)

        self.circuit = full_circuit
        self.params = input_params + self.learning_params

        self.ops = []
        for qubit in qubits:
            self.ops.append(cirq.Z(qubit))

    def _get_new_param(self):
        new_param = sympy.symbols("p" + str(len(self.learning_params)))
        self.learning_params.append(new_param)
        return new_param

    def unrolled_convolution_inputs(self, inputs):
        conv_inputs = []

        for i in range(self.out_height):
            for j in range(self.out_width):
                conv_input = tf.slice(inputs, [0, i, j], [-1, self.filter_size[0], self.filter_size[1]])
                conv_inputs.append(conv_input)

        # stack -> N, out_width*out_height, self.filter_size, self.filter_size
        stack = tf.stack(conv_inputs, axis=1)
        # reshape -> N*out_width*out_height, filter_size^2
        return tf.reshape(stack, shape=[-1, self.filter_size[0] * self.filter_size[1]])

    def build(self, input_shape):
        self.in_width = input_shape[2]
        self.in_height = input_shape[1]
        self.out_width = self.in_width - self.filter_size[1] + 1
        self.out_height = self.in_height - self.filter_size[0] + 1

        self.kernel = self.add_weight(name='kernel', shape=[len(self.learning_params), ],
                                      initializer=tf.keras.initializers.glorot_normal())
        self.circuit_tensor = tfq.convert_to_tensor([self.circuit] * self.out_width * self.out_height)

    def call(self, inputs):
        # inputs shapes: N, width, height

        # unroll convolution inputs -> N*out_width*out_height, filter_size^2
        conv_inputs = self.unrolled_convolution_inputs(inputs)

        circuit_inputs = tf.tile([self.circuit_tensor], [tf.shape(inputs)[0], 1])
        circuit_inputs = tf.reshape(circuit_inputs, shape=[-1])

        controller = tf.tile(self.kernel, [tf.shape(inputs)[0] * self.out_width * self.out_height])
        controller = tf.reshape(controller, shape=[tf.shape(inputs)[0] * self.out_width * self.out_height, -1])

        input_data = tf.concat([conv_inputs, controller], 1)

        output = tfq.layers.Expectation()(circuit_inputs, symbol_names=self.params,
                                          symbol_values=input_data, operators=self.ops)
        output = tf.reshape(output, shape=[-1, self.out_height, self.out_width, self.num_qubits_col+self.num_qubits_row+self.image_color_base])
        return output






