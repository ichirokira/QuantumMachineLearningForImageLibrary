from modelling import *
NAME = 'frqi_he_foldupmnist'

#-------------MODEL-----------------------
ENCODER = "FRQI"
TRANSFORMATION = "PQC_7"
NUM_BLOCKS = 3

ENTANGLING_ARR = 'chain'
TYPE_ENTANGLES = 'cnot'
MEASUREMENT = "full"

#-----------------DATASET--------------------
DATASET = 'MNIST'
NUM_FOLD = 8
CLASSES = [0, 1]
#-----------------TRAINING CONFIGURATION---------------------
MAX_NUM_QUBITS = 12
MIN_COLOR_QUBITS = 1
MIN_POS_QUBITS = 6
NUM_EPOCHES = 100
BATCH_SIZE = 240
LR = 0.001
LOG_DIR = "./results/{}_{}_Scalable_{}_{}_{}_{}_01/".format(ENCODER, TRANSFORMATION, NUM_FOLD, DATASET, NUM_BLOCKS, MEASUREMENT)
LOG_GRADIENTS = False
LOG_OUTPUT_VALUES = False
