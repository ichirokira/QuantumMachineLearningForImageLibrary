from modelling import *
NAME = 'frqi_he_multimnist'

#-------------MODEL-----------------------
ENCODER = "FRQI"
TRANSFORMATION = "HE"
NUM_BLOCKS = 5
ENTANGLING_ARR = 'chain'
TYPE_ENTANGLES = 'cnot'
MEASUREMENT = "full"

#-----------------DATASET--------------------
DATASET = 'FashionMNIST'
NUM_IMAGES = 4
LENGTH_DATA = 3000
ARGUMENTED_TIMES = 4
CLASSES = [0, 2]
#-----------------TRAINING CONFIGURATION---------------------
MAX_NUM_QUBITS = 12
MIN_COLOR_QUBITS = 1
MIN_POS_QUBITS = 6
NUM_EPOCHES = 50
BATCH_SIZE = 240
LR = 0.001
LOG_DIR = "./results/{}_{}_Multi_{}_{}_{}_{}_02/".format(ENCODER, TRANSFORMATION, NUM_IMAGES, DATASET, NUM_BLOCKS, MEASUREMENT)
LOG_GRADIENTS = True
LOG_OUTPUT_VALUES = True
