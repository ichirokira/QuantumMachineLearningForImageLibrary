from modelling import *
NAME = 'frqi_he_cifar'

#-------------MODEL-----------------------
ENCODER = "FRQI"
TRANSFORMATION = "HE"
NUM_BLOCKS = 5
ENTANGLING_ARR = 'chain'
TYPE_ENTANGLES = 'cnot'

#-----------------DATASET--------------------
DATASET = 'CIFAR10'
CLASSES = [0, 1]
#-----------------TRAINING CONFIGURATION---------------------
MAX_NUM_QUBITS = 10
MIN_COLOR_QUBITS = 1
MIN_POS_QUBITS = 6
NUM_EPOCHES = 200
BATCH_SIZE = 240
LR = 0.001
LOG_DIR = "./results/{}_{}_{}_{}/".format(ENCODER, TRANSFORMATION, DATASET, NUM_BLOCKS)
LOG_GRADIENTS = False

