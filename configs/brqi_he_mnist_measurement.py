from modelling import *
NAME = 'brqi_he_mnist_measurement'

#-------------MODEL-----------------------
ENCODER = "BRQI"
TRANSFORMATION = "HE"
NUM_BLOCKS = 3
ENTANGLING_ARR = 'chain'
TYPE_ENTANGLES = 'cnot'
MEASUREMENT = 'full'

#-----------------DATASET--------------------
DATASET = 'MNIST'
CLASSES = [0, 1]
#-----------------TRAINING CONFIGURATION---------------------
MAX_NUM_QUBITS = 10
MIN_COLOR_QUBITS = 4
MIN_POS_QUBITS = 6
NUM_EPOCHES = 50
BATCH_SIZE = 8
LR = 0.001
LOG_DIR = "./results/{}_{}_{}_{}_{}/".format(ENCODER, TRANSFORMATION, DATASET, NUM_BLOCKS, MEASUREMENT)
LOG_GRADIENTS = False
LOG_OUTPUT_VALUES = False

