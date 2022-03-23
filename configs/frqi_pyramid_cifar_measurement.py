from modelling import *
NAME = 'frqi_pyramid_cifar_measurement'

#-------------MODEL-----------------------
ENCODER = "FRQI"
TRANSFORMATION = "Pyramid"
NUM_BLOCKS = 3
ENTANGLING_ARR = 'chain'
TYPE_ENTANGLES = 'cnot'
MEASUREMENT = 'full'

#-----------------DATASET--------------------
DATASET = 'CIFAR10'
CLASSES = [0, 1]
#-----------------TRAINING CONFIGURATION---------------------
MAX_NUM_QUBITS = 12
MIN_COLOR_QUBITS = 1
MIN_POS_QUBITS = 6
NUM_EPOCHES = 20
BATCH_SIZE = 240
LR = 0.001
LOG_DIR = "./results/{}_{}_{}_{}_{}/".format(ENCODER, TRANSFORMATION, DATASET, NUM_BLOCKS, MEASUREMENT)
LOG_GRADIENTS = False
LOG_OUTPUT_VALUES = False
