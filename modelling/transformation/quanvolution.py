import tensorflow as tf
tf.config.run_functions_eagerly(True)
import tensorflow_quantum as tfq

import cirq
import math
import sympy
import numpy as np
import seaborn as sns
import collections

class Quanvolution(tf.keras.layers.Layer):
    def __init__(self, config, filter_size, **kwangs):
        super(Quanvolution, self).__init__(**kwangs)
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
                for b in range(self.num_qubits - self.image_color_base):
                    if cur_position_binary[b] != pre_position_binary[b]:
                        circuit.append(cirq.X(bits[b]))

                circuit.append(cirq.ry(2 * params[self.image_color_base * (i * self.filter_size[1] + j)]).on(
                        bits[-1]).controlled_by(*bits[:-self.image_color_base]))

                pre_position_binary = cur_position_binary
        return circuit

    def pqc_block(self, qubits):
        circuit1 = cirq.Circuit()
        index = [[i, i+1 % len(qubits)] for i in range(len(qubits))]
        for this_bits, next_bits in index:
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
        input_params = [sympy.symbols("a%d" % i) for i in range(self.filter_size ** 2)]
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
        for i in range(self.out_width):
            for j in range(self.out_height):
                conv_input = tf.slice(inputs, [0, i, j], [-1, self.filter_size, self.filter_size])
                conv_inputs.append(conv_input)

        # stack -> N, out_width*out_height, self.filter_size, self.filter_size
        stack = tf.stack(conv_inputs, axis=1)
        # reshape -> N*out_width*out_height, filter_size^2
        return tf.reshape(stack, shape=[-1, self.filter_size ** 2])

    def build(self, input_shape):
        self.in_width = input_shape[1]
        self.in_height = input_shape[2]
        self.out_width = self.in_width - self.filter_size + 1
        self.out_height = self.in_height - self.filter_size + 1

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
        output = tf.reshape(output, shape=[-1, self.out_width, self.out_height, self.num_qubits_col+self.num_qubits_row+self.image_color_base])
        return output


class QuanvolutionNEQR(tf.keras.layers.Layer):
    def __init__(self, config, filter_size, color_qubits=8, **kwangs):
        super(QuanvolutionNEQR, self).__init__(**kwangs)
        self.filter_size = filter_size
        self.block_count = config.NUM_BLOCKS
        self.encoder_type = config.ENCODER

        self.config = config

        self.num_qubits_row = (math.ceil(math.log2(self.filter_size[0])))
        self.num_qubits_col = (math.ceil(math.log2(self.filter_size[1])))
        self.num_qubits_color = color_qubits
        self.num_qubits = self.num_qubits_row + self.num_qubits_col + self.num_qubits_color
        self.bits = cirq.GridQubit.rect(1, self.num_qubits)

        self.learning_params = []
        self.transformation_circuit = self.layer_gen()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filter_size': self.filter_size,
            'block_count': self.block_count,
        })
        return config

    def NEQR(self, bits, params):
        # with tf.compat.v1.Session() as sess:
        #     sess.run(tf.compat.v1.global_variables_initializer())
        #     params = params.eval()
        # print(params)
        a = tf.constant(params)
        proto_tensor = tf.make_tensor_proto(a)
        params_numpy = tf.make_ndarray(proto_tensor)
        pre_position_binary = ""

        for i in range(self.num_qubits - self.num_qubits_color):
            pre_position_binary += "0"
        circuit = cirq.Circuit()
        for i in range(self.num_qubits - self.num_qubits_color):
            circuit.append(cirq.H(bits[i]))
        for i in range((self.filter_size[0])):
            for j in range((self.filter_size[1])):
                cur_position_binary = format(i, "b").zfill(self.num_qubits_row) + format(j, "b").zfill(
                    self.num_qubits_col)
                for b in range(self.num_qubits - self.num_qubits_color):
                    if cur_position_binary[b] != pre_position_binary[b]:
                        circuit.append(cirq.X(bits[b]))
                # color_bin_string = format(params[i*self.image_shape[1]+j], "b").zfill(self.num_qubits_color)
                color_bin_string = format(params_numpy[i * self.filter_size[1] + j], "b").zfill(self.num_qubits_color)
                # print(color_bin_string)
                for indx, cb in enumerate(color_bin_string[::-1]):
                    # print(indx)
                    if cb == '1':
                        circuit.append(cirq.X(bits[self.num_qubits_row + self.num_qubits_col + indx]).controlled_by(
                            *bits[:(self.num_qubits - self.num_qubits_color)]))
                pre_position_binary = cur_position_binary
        return circuit

    def pqc_block(self, qubits):
        circuit1 = cirq.Circuit()
        index = [[i, i + 1 % len(qubits)] for i in range(len(qubits))]
        for this_bits, next_bits in index:
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
        cir_c = self.convolution_circuit(self.bits)
        return cir_c

    def _get_new_param(self):
        new_param = sympy.symbols("p" + str(len(self.learning_params)))
        self.learning_params.append(new_param)
        return new_param

    def unrolled_convolution_inputs(self, inputs):
        conv_inputs = []
        for i in range(self.out_width):
            for j in range(self.out_height):
                conv_input = tf.slice(inputs, [0, i, j], [-1, self.filter_size, self.filter_size])
                conv_inputs.append(conv_input)

        # stack -> N, out_width*out_height, self.filter_size, self.filter_size
        stack = tf.stack(conv_inputs, axis=1)
        # reshape -> N*out_width*out_height, filter_size^2
        return tf.reshape(stack, shape=[-1, self.filter_size ** 2])

    def build(self, input_shape):
        self.in_width = input_shape[1]
        self.in_height = input_shape[2]
        self.out_width = self.in_width - self.filter_size + 1
        self.out_height = self.in_height - self.filter_size + 1

        self.kernel = self.add_weight(name='kernel', shape=[len(self.learning_params), ],
                                      initializer=tf.keras.initializers.glorot_normal())
        # self.circuit_tensor = tfq.convert_to_tensor([self.circuit] * self.out_width * self.out_height)

    def call(self, inputs):
        # inputs shapes: N, width, height

        # unroll convolution inputs -> N*out_width*out_height, filter_size^2
        conv_inputs = self.unrolled_convolution_inputs(inputs)

        encoder_circuits = []
        for conv_input in conv_inputs:
            encoder_circuits.append(tfq.convert_to_tensor([self.NEQR(self.bits, conv_input)]))

        encoder_circuits_tensor = tf.stack(encoder_circuits)

        transformation_circuits_tensor = tfq.convert_to_tensor([self.transformation_circuit]*self.out_width * self.out_height)
        transformation_circuits_tensor = tf.tile([transformation_circuits_tensor], [tf.shape(inputs)[0], 1])

        full_circuits = []
        for encoder, transformation in zip(encoder_circuits_tensor, transformation_circuits_tensor):
            e = tfq.from_tensor(encoder)
            t = tfq.from_tensor(transformation)
            full_circuit = e+t
            full_circuits.append((tfq.convert_to_tensor([full_circuit])))

        full_circuits_tensor = tf.concat(full_circuits, 0)
        full_circuits_tensor = tf.reshape(full_circuits_tensor, shape=[-1])
        #circuit_inputs = tf.tile([self.circuit_tensor], [tf.shape(inputs)[0], 1])
        #circuit_inputs = tf.reshape(circuit_inputs, shape=[-1])

        controller = tf.tile(self.kernel, [tf.shape(inputs)[0] * self.out_width * self.out_height])
        controller = tf.reshape(controller, shape=[tf.shape(inputs)[0] * self.out_width * self.out_height, -1])

        # input_data = tf.concat([conv_inputs, controller], 1)
        self.ops = []
        for i in self.bits:
            self.ops.append(cirq.Z(i))

        output = tfq.layers.Expectation()(full_circuits_tensor, symbol_names=self.learning_params,
                                          symbol_values=controller, operators=self.ops)
        output = tf.reshape(output, shape=[-1, self.out_width, self.out_height,
                                           self.num_qubits])
        return output



