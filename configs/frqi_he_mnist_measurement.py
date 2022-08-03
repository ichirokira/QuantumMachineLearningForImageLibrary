from modelling import *
NAME = 'frqi_he_mnist_measurement'

#-------------MODEL-----------------------
ENCODER = "FRQI"
TRANSFORMATION = "PQC_7"
NUM_BLOCKS = 3
ENTANGLING_ARR = 'chain'
TYPE_ENTANGLES = 'cnot'
MEASUREMENT = 'full'


#-----------------DATASET--------------------
DATASET = 'MNIST'
CLASSES = [2, 3]
#-----------------TRAINING CONFIGURATION---------------------
MAX_NUM_QUBITS = 12
MIN_COLOR_QUBITS = 1
MIN_POS_QUBITS = 6
NUM_EPOCHES = 100
BATCH_SIZE = 32
LR = 0.001
LOG_DIR = "./results/{}_{}_{}_{}_{}_{}/".format(ENCODER, TRANSFORMATION, DATASET, NUM_BLOCKS, MEASUREMENT, CLASSES)
LOG_GRADIENTS = False
LOG_OUTPUT_VALUES = False
