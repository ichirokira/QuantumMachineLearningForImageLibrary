import tensorflow as tf
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



class Multimodal_FRQI(tf.keras.layers.Layer):
    def __init__(self, config, name=None, **kwangs):
        super(Multimodal_FRQI, self).__init__(name, **kwangs)
        self.learning_params = []
        self.config=config
        self.num_blocks = config.NUM_BLOCKS
        self.type_entangles = config.TYPE_ENTANGLES
        self.entangling_arrangement = config.ENTANGLING_ARR
        self.num_views_qubits = (math.ceil(math.log2(len(config.VIEWS))))
        self.num_feature_qubits = (math.ceil(math.log2(config.MAX_LENGTH)))
        # self.num_qubits_row = (math.ceil(math.log2(self.image_shape[0])))
        # self.num_qubits_col = (math.ceil(math.log2(self.image_shape[1])))
        # if self.image_shape[2] == 1:
        #     self.image_color_base = 1
        # elif self.image_shape[2] == 3:
        #     self.image_color_base = 3
        self.num_qubits = self.num_views_qubits + self.num_feature_qubits + 1
        self.transformation = config.TRANSFORMATION

        input_dim = (len(config.VIEWS), config.MAX_LENGTH)
        self.QNNL_layer_gen(input_dim)

    def _print_circuit(self):
        return SVGCircuit(self.circuit)

    def _get_new_param(self):
        new_param = sympy.symbols("p" + str(len(self.learning_params)))
        self.learning_params.append(new_param)
        return new_param

    def Multiview_FRQI(self, bits, params):
        pre_index_binary = ""
        for m in range(self.num_views_qubits):
            pre_index_binary += "0"

        circuit = cirq.Circuit()
        for i in range(self.num_views_qubits):
            circuit.append(cirq.H(bits[i]))

        for i in range(self.num_feature_qubits):
            circuit.append(cirq.H(bits[self.num_views_qubits+i]))
        for n in range(len(self.config.VIEWS)):

            pre_position_binary = ""
            for k in range(self.num_feature_qubits):
                pre_position_binary += "0"

            cur_index_binary = format(n, "b").zfill(self.num_views_qubits)
            for index_bit in range(self.num_views_qubits):
                if cur_index_binary[index_bit] != pre_index_binary[index_bit]:
                    circuit.append(cirq.X(bits[index_bit]))
            for i in range((self.config.MAX_LENGTH)):
                # for j in range((self.image_shape[1])):
                cur_position_binary = format(i, "b").zfill(self.num_feature_qubits)
                for b in range(self.num_feature_qubits):
                    if cur_position_binary[b] != pre_position_binary[b]:
                        circuit.append(cirq.X(bits[self.num_views_qubits+b]))

                circuit.append(cirq.ry(2 * params[n*self.config.MAX_LENGTH+i]).on(
                    bits[-1]).controlled_by(*bits[:-1]))

                pre_position_binary = cur_position_binary
            pre_index_binary = cur_index_binary
            cur_position_binary = ""
            for k in range(self.num_feature_qubits):
                cur_position_binary += "0"
            for b in range(self.num_feature_qubits):
                if cur_position_binary[b] != pre_position_binary[b]:
                    circuit.append(cirq.X(bits[self.num_views_qubits+b]))
        return circuit
    def QNNL_layer_gen(self, input_dim):
        bits = cirq.GridQubit.rect(1, self.num_qubits)
        if self.transformation == "Farhi":
            readout = cirq.GridQubit(-1, -1)
            num_classes = len(self.config.CLASSES)
            assert num_classes == 2, "Farhi Design only supports for binary classification"
        input_params = []
        for n in range(input_dim[0]):
            for i in range(input_dim[1]):
                input_params.append(sympy.symbols("a{}-{}".format(n, i)))

        full_circuit = cirq.Circuit()
        encoder = self.Multiview_FRQI(bits, input_params)
        full_circuit.append(encoder)

        for i in range(self.num_blocks):
            if self.transformation == "HE":
                block = HE(bits, entangling_arrangement=self.entangling_arrangement, type_entangles=self.type_entangles,
                           gen_params=self._get_new_param)


            full_circuit.append(block)

        self.circuit = full_circuit
        # print(self.circuit)
        self.params = input_params + self.learning_params
        if self.transformation == "Farhi":
            self.ops = cirq.Z(readout)
        else:
            if self.config.MEASUREMENT == 'single':
                self.ops = cirq.X(bits[-1])
                # self.ops = 0
                # for i in range(len(bits)):
                #     self.ops += cirq.X(bits[i]) * 1 / len(bits)
            elif self.config.MEASUREMENT == 'selection':
                self.ops = []
                for i in range(20):
                    self.ops.append(cirq.X(bits[-1]))

            else:
                self.ops = []
                for i in range(len(bits)):
                    self.ops.append(cirq.X(bits[i]))
                # self.ops = []
                # for i in range(20):
                #     self.ops.append(cirq.X(bits[-1]))
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=[len(self.learning_params), ],
                                      initializer=tf.keras.initializers.glorot_normal())

        self.circuit_tensor = tfq.convert_to_tensor([self.circuit])

    def call(self, inputs):
        # inputs shapes: N, V, C
        inputs = tf.reshape(inputs, shape=[tf.shape(inputs)[0], -1])
        circuit_inputs = tf.tile([self.circuit_tensor], [tf.shape(inputs)[0], 1])

        circuit_inputs = tf.reshape(circuit_inputs, shape=[-1])
        # print(tf.shape(inputs)[0])
        controller = tf.tile(self.kernel, [tf.shape(inputs)[0]])
        controller = tf.reshape(controller, shape=[tf.shape(inputs)[0], -1])
        # print(controller.shape)
        input_data = tf.concat([inputs, controller], 1)

        QNNL_output = tfq.layers.Expectation()(circuit_inputs, symbol_names=self.params,
                                               symbol_values=input_data, operators=self.ops)

        # QNNL_output = tf.keras.layers.Dense(self.num_classes)(QNNL_output)
        # QNNL_output = tf.keras.activations.softmax(QNNL_output, axis=1)
        return QNNL_output





