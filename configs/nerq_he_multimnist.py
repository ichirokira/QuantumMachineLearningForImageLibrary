from modelling import *
NAME = 'neqr_he_multimnist'

#-------------MODEL-----------------------
ENCODER = "NERQ"
TRANSFORMATION = "HE"
NUM_BLOCKS = 3

ENTANGLING_ARR = 'chain'
TYPE_ENTANGLES = 'cnot'
MEASUREMENT = "full"

#-----------------DATASET--------------------
DATASET = 'MNIST'
NUM_IMAGES = 2
CLASSES = [0, 1]
#-----------------TRAINING CONFIGURATION---------------------
MAX_NUM_QUBITS = 12
MIN_COLOR_QUBITS = 4
MIN_POS_QUBITS = 8
NUM_EPOCHES = 50
BATCH_SIZE = 240
LR = 0.001
LOG_DIR = "./results/{}_{}_Multi_{}_{}_{}_{}/".format(ENCODER, TRANSFORMATION, NUM_IMAGES, DATASET, NUM_BLOCKS, MEASUREMENT)
LOG_GRADIENTS = False
LOG_OUTPUT_VALUES = False
