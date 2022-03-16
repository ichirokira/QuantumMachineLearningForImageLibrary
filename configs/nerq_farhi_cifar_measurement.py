from modelling import *
NAME = 'nerq_farhi_cifar'

#-------------MODEL-----------------------
ENCODER = "NERQ"
TRANSFORMATION = "Farhi"
NUM_BLOCKS = 1
ENTANGLING_ARR = 'chain'
TYPE_ENTANGLES = 'cnot'
MEASUREMENT = "full"
#-----------------DATASET--------------------
DATASET = 'CIFAR10'
CLASSES = [4, 5]
#-----------------TRAINING CONFIGURATION---------------------
MAX_NUM_QUBITS = 12
MIN_COLOR_QUBITS = 4
MIN_POS_QUBITS = 8
NUM_EPOCHES = 50
BATCH_SIZE = 240
LR = 0.001
LOG_DIR = "./results/{}_{}_{}_{}_{}/".format(ENCODER, TRANSFORMATION, DATASET, NUM_BLOCKS, MEASUREMENT)
LOG_GRADIENTS = False
LOG_OUTPUT_VALUES = False
