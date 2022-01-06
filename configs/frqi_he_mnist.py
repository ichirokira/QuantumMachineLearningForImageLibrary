from modelling import *
NAME = 'frqi_he_mnist'

#-------------MODEL-----------------------
ENCODER = "FRQI"
TRANSFORMATION = "HE"
NUM_BLOCKS = 3
ENTANGLING_ARR = 'chain'
TYPE_ENTANGLES = 'cnot'

#-----------------DATASET--------------------
DATASET = 'MNIST'
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
IMAGE_SHAPE = (28, 28,1)
#-----------------TRAINING CONFIGURATION---------------------
NUM_EPOCHES = 200
BATCH_SIZE = 240
LR = 0.001
LOG_DIR = "./results/{}_{}_{}_{}/".format(ENCODER, TRANSFORMATION, DATASET, NUM_BLOCKS)
LOG_GRADIENTS = False

