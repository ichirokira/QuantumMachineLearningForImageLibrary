# QuantumMachineLearningForImageLibrary

## Introduction
Quantum Machine Learning For Image Library is a collection of Quantum Machine Learning circuits,
representations that aim to generalize the train paradigm for image using Quantum Machine Learning.

The simple usage allows you try your own modification

```commandline
python train.py --config configs/your_configs.py
```
## Installation

```commandline
pip install tensorflow==2.4.1
pip install tensorflow_quantum==0.5.1
pip install seaborn
```

## Configuration
```doctest
from modelling import *
NAME = 'frqi_he_mnist' # Name of your config file

#-------------MODEL-----------------------
ENCODER = "FRQI" # Name of your Encoder methods (Please check Current Support for details)
TRANSFORMATION = "HE" # Name of your Encoder methods (Please check Current Support for details)
NUM_BLOCKS = 3 # Number of Transformation layers
ENTANGLING_ARR = 'chain' # Arrangement of entangle gate ("chain", "all")
TYPE_ENTANGLES = 'cnot' # Types of entangle gate ("cnot", "cphase", "sqrtiswap")
MEASUREMENT = 'full' # Type of measurement ("full"-use all qubits, "single"-use only color qubit)

#-----------------DATASET--------------------
DATASET = 'MNIST' # Name of Dataset (Please check Current Support for details)
CLASSES = [0, 1] # List Classes
#-----------------TRAINING CONFIGURATION---------------------
MAX_NUM_QUBITS = 12 # Maximum number of qubits in your circuits. If excess, the code will automatically modify size of 
                    # image or range of color
MIN_COLOR_QUBITS = 1 # Minimum number of qubits encode color
MIN_POS_QUBITS = 6 # Minimum number of qubits encode position
NUM_EPOCHES = 200 # Number of epochs for training process
BATCH_SIZE = 240 # Batch Size for training process
LR = 0.001 # Learning rate
LOG_DIR = "./results/{}_{}_{}_{}/".format(ENCODER, TRANSFORMATION, DATASET, NUM_BLOCKS) # log directory
LOG_GRADIENTS = False # Whether log the gradient details or not
LOG_OUTPUT_VALUES = True # Whether log the measurement output of each qubit or not

```
## Current Support 

### Encoder

- FRQI: [Flexible Representation of Quantum Images](https://doi.org/10.1007/s11128-010-0177-y)
- NERQ: [A novel enhanced quantum representation of digital images](https://doi.org/10.1007/s11128-013-0567-z) 
### Transformation
- HE: [Hard-efficient circuit Design](https://arxiv.org/abs/1704.05018)
- HE_color_indenpendence: a modified version of HE that separate color qubits
### Dataset
- MNIST
- CIFAR10