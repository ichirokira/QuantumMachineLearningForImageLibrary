from modelling import *
NAME = 'frqi_multi_6_view_mnist'

#-------------MODEL-----------------------
ENCODER = "FRQI"
TRANSFORMATION = "HE"
NUM_BLOCKS = 5
ENTANGLING_ARR = 'chain'
TYPE_ENTANGLES = 'cnot'
MEASUREMENT = "full"

#-----------------DATASET--------------------
DATASET = '6_VIEWS'
DATA_PATH = "../MultiviewDataset/handwritten_6views.mat"
VIEWS = [0,1,2,3]
MAX_LENGTH = 256
CLASSES = [1, 2]
#-----------------TRAINING CONFIGURATION---------------------

NUM_EPOCHES = 200
BATCH_SIZE = 16
LR = 0.001
LOG_DIR = "./results/{}_{}_Multi_{}_{}_{}_{}_{}_v10/".format(ENCODER, TRANSFORMATION, VIEWS, MAX_LENGTH, DATASET, NUM_BLOCKS, MEASUREMENT)
LOG_GRADIENTS = False
LOG_OUTPUT_VALUES = False
