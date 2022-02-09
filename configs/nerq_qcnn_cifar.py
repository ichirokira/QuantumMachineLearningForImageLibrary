from modelling import *
NAME = 'nerq_qcnn_cifar'

#-------------MODEL-----------------------
ENCODER = "NERQ"
TRANSFORMATION = "QCNN"
NUM_BLOCKS = 1
ENTANGLING_ARR = 'chain'
TYPE_ENTANGLES = 'cnot'

#-----------------DATASET--------------------
DATASET = 'CIFAR10'
CLASSES = [0, 1]
#-----------------TRAINING CONFIGURATION---------------------
MAX_NUM_QUBITS = 12
MIN_COLOR_QUBITS = 1
MIN_POS_QUBITS = 6
NUM_EPOCHES = 50
BATCH_SIZE = 16
LR = 0.001
LOG_DIR = "./results/{}_{}_{}_{}/".format(ENCODER, TRANSFORMATION, DATASET, NUM_BLOCKS)
LOG_GRADIENTS = False

