from modelling import *
NAME = 'nerq_he_audio'

#-------------MODEL-----------------------
ENCODER = "NERQ"
TRANSFORMATION = "HE"
NUM_BLOCKS = 3
ENTANGLING_ARR = 'chain'
TYPE_ENTANGLES = 'cnot'
MEASUREMENT = 'full'

#-----------------DATASET--------------------
DATASET = 'AUDIO_MNIST'

CLASSES = [0, 1]
#-----------------TRAINING CONFIGURATION---------------------
MAX_NUM_QUBITS = 12
MIN_COLOR_QUBITS = 4
MIN_POS_QUBITS = 6
NUM_EPOCHES = 200
BATCH_SIZE = 32
LR = 0.001
LOG_DIR = "./results/{}_{}_{}_{}_{}/".format(ENCODER, TRANSFORMATION, DATASET, NUM_BLOCKS, MEASUREMENT)
LOG_GRADIENTS = False
LOG_OUTPUT_VALUES = False
