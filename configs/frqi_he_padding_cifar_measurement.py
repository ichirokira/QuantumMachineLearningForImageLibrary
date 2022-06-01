from modelling import *
NAME = 'frqi_he_padding_cifar'

#-------------MODEL-----------------------
ENCODER = "FRQI"
TRANSFORMATION = "HE"
NUM_BLOCKS = 3

ENTANGLING_ARR = 'chain'
TYPE_ENTANGLES = 'cnot'
MEASUREMENT = "full"

#-----------------DATASET--------------------
DATASET = 'CIFAR10'
NEW_SIZE = [16,16]
PAD_TEST = True
CLASSES = [0, 1]
#-----------------TRAINING CONFIGURATION---------------------
MAX_NUM_QUBITS = 12
MIN_COLOR_QUBITS = 1
MIN_POS_QUBITS = 6
NUM_EPOCHES = 200
BATCH_SIZE = 240
LR = 0.001
LOG_DIR = "./results/{}_{}_Scale_{}_padTest_{}_{}_{}_{}/".format(ENCODER, TRANSFORMATION, IMG_SCALE, PAD_TEST, DATASET, NUM_BLOCKS, MEASUREMENT)
LOG_GRADIENTS = True
LOG_OUTPUT_VALUES = True
