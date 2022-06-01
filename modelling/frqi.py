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


class MCQI_Gate(cirq.Gate):
    """
    This multichannel design are currently not supported by tfq
    We cannot serialize MCQI Gate to use in tf.keras.layers
    """
    def __init__(self, bits, theta_r, theta_g, theta_b):
        super(MCQI_Gate, self)
        self.bits = bits
        self.theta_r = theta_r
        self.theta_g = theta_g
        self.theta_b = theta_b

    def _num_qubits_(self):
        return 3

    def _unitary_(self):
        uni_r = cirq.ry(2 * self.theta_r).on(self.bits[-3]).controlled_by(*self.bits[-3:])._unitary_()
        uni_g = cirq.ry(2 * self.theta_g).on(self.bits[-2]).controlled_by(*self.bits[-3:])._unitary_()
        uni_b = cirq.ry(2 * self.theta_b).on(self.bits[-1]).controlled_by(*self.bits[-3:])._unitary_()

        return np.matmul(uni_b, np.matmul(uni_g, uni_r))

    def _circuit_diagram_info_(self, args):
        return "MCQI_R", "MCQI_G", "MCQI_B"

class FRQI_Basis(tf.keras.layers.Layer):
    def __init__(self, config, image_shape=(28,28,1), name=None, **kwangs):
        super(FRQI_Basis, self).__init__(name, **kwangs)
        self.learning_params = []
        self.config=config
        self.num_blocks = config.NUM_BLOCKS
        self.type_entangles = config.TYPE_ENTANGLES
        self.entangling_arrangement = config.ENTANGLING_ARR
        self.image_shape = image_shape
        self.num_qubits_row = (math.ceil(math.log2(self.image_shape[0])))
        self.num_qubits_col = (math.ceil(math.log2(self.image_shape[1])))
        if self.image_shape[2] == 1:
            self.image_color_base = 1
        elif self.image_shape[2] == 3:
            self.image_color_base = 3
        self.num_qubits = self.num_qubits_row + self.num_qubits_col + self.image_color_base  # position qubits and color intensity qubit
        self.transformation = config.TRANSFORMATION

        self.QNNL_layer_gen(self.image_shape)

    def _print_circuit(self):
        return SVGCircuit(self.circuit)

    def _get_new_param(self):
        new_param = sympy.symbols("p" + str(len(self.learning_params)))
        self.learning_params.append(new_param)
        return new_param

    def FRQI(self, bits, params):

        pre_position_binary = ""
        for i in range(self.num_qubits - self.image_color_base):
            pre_position_binary += "0"
        circuit = cirq.Circuit()
        for i in range(self.num_qubits - self.image_color_base):
            circuit.append(cirq.H(bits[i]))
        for i in range((self.image_shape[0])):
            for j in range((self.image_shape[1])):
                cur_position_binary = format(i, "b").zfill(self.num_qubits_row) + format(j, "b").zfill(
                    self.num_qubits_col)
                for b in range(self.num_qubits - self.image_color_base):
                    if cur_position_binary[b] != pre_position_binary[b]:
                        circuit.append(cirq.X(bits[b]))
                if self.image_color_base == 1:
                    circuit.append(cirq.ry(2 * params[self.image_color_base * (i * self.image_shape[1] + j)]).on(
                        bits[-1]).controlled_by(*bits[:-self.image_color_base]))
                elif self.image_color_base == 3:
                    theta_r = params[self.image_color_base * (i * self.image_shape[1] + j)]
                    theta_g = params[self.image_color_base * (i * self.image_shape[1] + j) + 1]
                    theta_b = params[self.image_color_base * (i * self.image_shape[1] + j) + 2]
                    circuit.append(
                        MCQI_Gate(bits, theta_r, theta_g, theta_b).on(*bits[-self.image_color_base:]).controlled_by(
                            *bits[:-self.image_color_base]))
                pre_position_binary = cur_position_binary
        return circuit
    def QNNL_layer_gen(self, input_dim):
        bits = cirq.GridQubit.rect(1, self.num_qubits)
        if self.transformation == "Farhi":
            readout = cirq.GridQubit(-1, -1)
            num_classes = len(self.config.CLASSES)
            assert num_classes == 2, "Farhi Design only supports for binary classification"
        input_params = []
        for i in range(input_dim[0]):
            for j in range(input_dim[1]):
                for c in range(input_dim[2]):
                    input_params.append(sympy.symbols("a{}-{}-{}".format(i, j, c)))

        full_circuit = cirq.Circuit()
        encoder = self.FRQI(bits, input_params)
        full_circuit.append(encoder)

        for i in range(self.num_blocks):
            if self.transformation == "HE":
                block = HE(bits, entangling_arrangement=self.entangling_arrangement, type_entangles=self.type_entangles,
                           gen_params=self._get_new_param)
            elif self.transformation == "HE_color_indenpendence":
                block = HE_Color_Independence(bits[:-self.image_color_base], bits[-self.image_color_base: ],
                                              entangling_arrangement=self.entangling_arrangement, type_entangles=self.type_entangles,
                                              gen_params=self._get_new_param)
            elif self.transformation == "Farhi":
                block = Farhi(bits, readout, gen_params=self._get_new_param)
            elif self.transformation == "Pyramid":
                block = Pyramid_Transform(position_bits=bits[:-self.image_color_base], color_bits=bits[-self.image_color_base: ],
                                          num_col_bits=self.num_qubits_col, num_row_bits=self.num_qubits_row, gen_params=self._get_new_param)

            elif "PQC" in self.transformation:
                block = PQCs(bits, gen_params=self._get_new_param, pqc=int(self.transformation.split("_")[1])).circuit

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
        # inputs shapes: N, H, W, C
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





