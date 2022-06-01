from modelling import *
NAME = 'neqr_he_foldupmnist'

#-------------MODEL-----------------------
ENCODER = "NERQ"
TRANSFORMATION = "HE"
NUM_BLOCKS = 5

ENTANGLING_ARR = 'chain'
TYPE_ENTANGLES = 'cnot'
MEASUREMENT = "full"

#-----------------DATASET--------------------
DATASET = 'MNIST'
NUM_FOLD = 2
CLASSES = [0, 1]
#-----------------TRAINING CONFIGURATION---------------------
MAX_NUM_QUBITS = 12
MIN_COLOR_QUBITS = 4
MIN_POS_QUBITS = 8
NUM_EPOCHES = 50
BATCH_SIZE = 240
LR = 0.001
LOG_DIR = "./results/{}_{}_Foldup_{}_{}_{}_{}/".format(ENCODER, TRANSFORMATION, NUM_FOLD, DATASET, NUM_BLOCKS, MEASUREMENT)
LOG_GRADIENTS = False
LOG_OUTPUT_VALUES = False
