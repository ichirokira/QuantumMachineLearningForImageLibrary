import tensorflow as tf
tf.config.run_functions_eagerly(True)
import tensorflow_quantum as tfq
from tensorflow.keras.utils import to_categorical

import cirq
import sympy
import numpy as np
import seaborn as sns
import collections
import math
import itertools
from modelling.transformation import *

from cirq.contrib.svg import SVGCircuit


class BRQI_Basis(tf.keras.layers.Layer):
    def __init__(self, config, image_shape=(28,28,1), color_qubits=8, name=None, **kwangs):
        super(BRQI_Basis, self).__init__(name, **kwangs)
        self.learning_params = []
        self.config = config
        self.num_blocks = config.NUM_BLOCKS
        self.type_entangles = config.TYPE_ENTANGLES
        self.entangling_arrangement = config.ENTANGLING_ARR
        self.image_shape = image_shape
        self.num_qubits_row = (math.ceil(math.log2(self.image_shape[0])))
        self.num_qubits_col = (math.ceil(math.log2(self.image_shape[1])))
        self.num_qubits_color = (math.ceil(math.log2(color_qubits)))
        self.num_qubits = self.num_qubits_row + self.num_qubits_col + self.num_qubits_color + 1
        self.transformation = config.TRANSFORMATION
        self.bits = cirq.GridQubit.rect(1, self.num_qubits)
        if self.transformation == "Farhi":
            self.readout = cirq.GridQubit(-1, -1)
            assert len(self.config.CLASSES) == 2, "Farhi Design only supports for binary classification"
        # self.bs = bs
        self.transform_circuit = self.QNNL_layer_gen(self.bits)

    def _print_circuit(self):
        return SVGCircuit(self.circuit)

    def _get_new_param(self):
        new_param = sympy.symbols("p" + str(len(self.learning_params)))
        self.learning_params.append(new_param)
        return new_param

    def BRQI(self, bits, params):
        # with tf.compat.v1.Session() as sess:
        #     sess.run(tf.compat.v1.global_variables_initializer())
        #     params = params.eval()
        #print(params)
        a = tf.constant(params)
        proto_tensor = tf.make_tensor_proto(a)
        params_numpy = tf.make_ndarray(proto_tensor)
        pre_position_binary = ""
        for i in range(self.num_qubits - self.num_qubits_color):
            pre_position_binary += "0"



        circuit = cirq.Circuit()
        for i in range(self.num_qubits_row + self.num_qubits_col):
            circuit.append(cirq.H(bits[i]))
        for i in range(self.num_qubits_color):
            circuit.append(cirq.H(bits[self.num_qubits_row + self.num_qubits_col+i]))

        for i in range((self.image_shape[0])):
            for j in range((self.image_shape[1])):
                cur_position_binary = format(i, "b").zfill(self.num_qubits_row) + format(j, "b").zfill(
                    self.num_qubits_col)
                for b in range(self.num_qubits_row + self.num_qubits_col):
                    if cur_position_binary[b] != pre_position_binary[b]:
                        circuit.append(cirq.X(bits[b]))
                # color_bin_string = format(params[i*self.image_shape[1]+j], "b").zfill(self.num_qubits_color)
                color_bin_string = format(params_numpy[i * self.image_shape[1] + j], "b").zfill(self.num_qubits_color)
                #print(color_bin_string)
                start_color_index = ""
                for i in range(self.num_qubits_color):
                    start_color_index += "0"
                pre_color_index = start_color_index
                for indx, cb in enumerate(color_bin_string[::-1]):
                    #print(indx)
                    cur_color_index = format(indx, "b").zfill(self.num_qubits_color)
                    for bit_color_indx in range(self.num_qubits_color):
                        if cur_color_index[bit_color_indx] != pre_color_index[bit_color_indx]:
                            circuit.append(cirq.X(bits[self.num_qubits_row + self.num_qubits_col+bit_color_indx]))
                    if cb == '1':
                        circuit.append(cirq.X(bits[-1]).controlled_by(
                            *bits[:(-1)]))
                    pre_color_index = cur_color_index
                for bit_color_indx in range(self.num_qubits_color):
                    if pre_color_index[bit_color_indx] != start_color_index[bit_color_indx]:
                        circuit.append(cirq.X(bits[self.num_qubits_row + self.num_qubits_col + bit_color_indx]))
                pre_position_binary = cur_position_binary
        return circuit



    def QNNL_layer_gen(self, bits):
        circuit = cirq.Circuit()
        for i in range(self.num_blocks):
            if self.transformation == "HE":
                block = HE(bits, entangling_arrangement=self.entangling_arrangement, type_entangles=self.type_entangles,
                           gen_params=self._get_new_param)
            elif self.transformation == "HE_color_indenpendence":
                block = HE_Color_Independence(bits[:-self.num_qubits_color], bits[-self.num_qubits_color:],
                                              entangling_arrangement=self.entangling_arrangement,
                                              type_entangles=self.type_entangles,
                                              gen_params=self._get_new_param)
            elif self.transformation == "Farhi":
                block = Farhi(bits, self.readout, gen_params=self._get_new_param)
            elif self.transformation == "Pyramid":
                block = Pyramid_Transform(position_bits=bits[:-self.image_color_base],
                                          color_bits=bits[-self.image_color_base:],
                                          num_col_bits=self.num_qubits_col, num_row_bits=self.num_qubits_row,
                                          gen_params=self._get_new_param)
            circuit.append(block)
        return circuit

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=[len(self.learning_params), ],
                                      initializer=tf.keras.initializers.glorot_normal())

        # self.circuit_tensor = tfq.convert_to_tensor([self.circuit])

    def call(self, inputs):

        # inputs shapes: N, H, W, C
        inputs = tf.reshape(inputs, shape=[tf.shape(inputs)[0], -1])
        encoder_circuits = []

        for input in inputs:
            #print(input.shape)
            encoder_circuits.append(tfq.convert_to_tensor([self.BRQI(self.bits, input)]))
        encoder_circuits_tensor = tf.stack(encoder_circuits)
        # circuit = cirq.Circuit()
        # for i in range(self.num_blocks):
        #     block = self.basic_blocks(bits)
        #     circuit.append(block)

        circuit_tensor = tfq.convert_to_tensor([self.transform_circuit])

        circuit_inputs = tf.tile([circuit_tensor], [tf.shape(inputs)[0], 1])
        full_circuits = []
        for encoder, circuit_input in zip(encoder_circuits_tensor, circuit_inputs):
            e = tfq.from_tensor(encoder)
            c = tfq.from_tensor(circuit_input)
            full_circuit = e + c
            full_circuits.append(tfq.convert_to_tensor([full_circuit]))

        full_circuits_tensor = tf.concat(full_circuits, 0)
        full_circuits_tensor = tf.reshape(full_circuits_tensor, shape=[-1])
        # print(tf.shape(inputs)[0])
        controller = tf.tile(self.kernel, [tf.shape(inputs)[0]])
        controller = tf.reshape(controller, shape=[tf.shape(inputs)[0], -1])
        # print(controller.shape)
        # input_data = tf.concat([inputs, controller], 1)
        if self.transformation == "Farhi":
            self.ops = cirq.Z(self.readout)
        else:
            if self.config.MEASUREMENT == 'single':
                self.ops = 0
                for i in range(self.num_qubits_color):
                    self.ops += cirq.X(self.bits[i])
            elif self.config.MEASUREMENT == 'selection':
                self.ops = []
                self.ops.append(cirq.X(self.bits[0]))
                self.ops.append(cirq.X(self.bits[1]))
            else:
                self.ops = []
                for i in range(len(self.bits)):
                    self.ops.append(cirq.X(self.bits[i]))
        QNNL_output = tfq.layers.Expectation()(full_circuits_tensor, symbol_names=self.learning_params,
                                                   symbol_values=controller, operators=self.ops)

        # QNNL_output = tf.keras.layers.Dense(self.num_classes)(QNNL_output)
        # QNNL_output = tf.keras.activations.softmax(QNNL_output, axis=1)
        return QNNL_output