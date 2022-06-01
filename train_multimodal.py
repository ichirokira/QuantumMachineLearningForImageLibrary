import math
import os
import sys
import shutil
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from dataset.dataset import *

from configs.utils import parse_config
from modelling import *
import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = ArgumentParser(description="Implementation of QuantumML for Image")
    parser.add_argument('--config', type=str, help='Model config')
    parser.add_argument('--resume', type=str, help='Checkpoint to resume')
    args = parser.parse_args()

    return args

def train(config):

    assert config.DATASET == "6_VIEWS", "This only supports 6 Views dataset. Please use another training file"
    train_dataset = HandWrittenDataGenerator(config, train=True)
    val_dataset = HandWrittenDataGenerator(config, train=False)
    # if config.DATASET == 'CIFAR10':
    #     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    #
    #     x_train = tf.image.rgb_to_grayscale(x_train)
    #     x_test = tf.image.rgb_to_grayscale(x_test)
    # elif config.DATASET == 'MNIST':
    #     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #     x_train = x_train[..., np.newaxis]
    #     x_test = x_test[..., np.newaxis]

    # N, H, W, C = x_train.shape
    # x_train = tf.cast(x_train, tf.float32)
    # x_test = tf.cast(x_test, tf.float32)
    #
    # print("Number of original training examples", len(x_train))
    # print("Number of original test examples", len(x_test))

    if config.ENCODER == 'FRQI':
        qnn_layer = Multimodal_FRQI(config)
    elif config.ENCODER == 'NERQ':
        num_qubits_row = (math.ceil(math.log2(H)))
        num_qubits_col = (math.ceil(math.log2(W)))
        color_qubits = 8
        num_qubits = num_qubits_col + num_qubits_row + color_qubits
        new_scale = 255.0
        if num_qubits > config.MAX_NUM_QUBITS:
            print("[INFO] Require {} qubits excess {}".format(num_qubits, config.MAX_NUM_QUBITS))
            removed_qubits = (num_qubits - config.MAX_NUM_QUBITS) // 3
            if removed_qubits == 0:
                removed_qubits = 1
            if num_qubits_row+num_qubits_col - 2*removed_qubits < config.MIN_POS_QUBITS:
                """if number of position qubits is smaller than threshold, rescale color"""

                if (color_qubits - 3*removed_qubits) > config.MIN_COLOR_QUBITS:
                    color_qubits -= 3*removed_qubits
                    new_scale = 2 ** color_qubits - 1
                    print("[INFO] Rescale Color Range to {}".format(new_scale + 1))
                else:
                    color_qubits = config.MIN_COLOR_QUBITS
                    pos_qubits = (config.MAX_NUM_QUBITS - color_qubits) // 2
                    x_train = tf.image.resize(x_train[:], (2 ** (pos_qubits), 2 ** (pos_qubits)))
                    x_test = tf.image.resize(x_test[:], (2 ** (pos_qubits), 2 ** (pos_qubits)))
                    new_scale = 2 ** color_qubits - 1
                    print("[INTO] Resize image from {} to {}. Rescale Color Range to {}".format([H, W],
                                                                                                [2 ** (pos_qubits),
                                                                                                 2 ** (pos_qubits)],
                                                                                                new_scale + 1))
                N, H, W, C = x_train.shape
                # new_scale = 2**color_qubits-1
                # print("[INFO] Rescale Color Range to {}".format(new_scale+1))
            else:
                x_train = tf.image.resize(x_train[:], (2 ** (num_qubits_row - removed_qubits), 2 ** (num_qubits_col - removed_qubits)))
                x_test = tf.image.resize(x_test[:], (2 ** (num_qubits_row - removed_qubits), 2 ** (num_qubits_col - removed_qubits)))

                if (color_qubits - removed_qubits) > config.MIN_COLOR_QUBITS:
                    color_qubits -= removed_qubits
                else:
                    color_qubits = config.MIN_COLOR_QUBITS
                new_scale = 2**color_qubits-1
                print("[INFO] Resize image from {} to {}. Rescale Color Range to {}".format([H, W],
                                                                                         [2 ** (num_qubits_row - removed_qubits), 2 ** (num_qubits_col - removed_qubits)],
                                                                                         new_scale+1))
                N, H, W, C = x_train.shape
        x_train = preprocessNEQR(x_train, new_scale=new_scale)
        x_test = preprocessNEQR(x_test, new_scale=new_scale)
        qnn_layer = NEQR_Basis(config, image_shape=(H, W, C), color_qubits=color_qubits)

    num_classes = len(config.CLASSES)
    if config.MEASUREMENT:
        assert num_classes == 2, "Single Measurement only supports for binary classification"
    # x_train = x_train.numpy()
    # x_test = x_test.numpy()
    # if config.ENCODER == "NERQ":
    #     x_train_filtered, y_train_filtered = filter_nerq(x_train, y_train, config, 1000, train=True)
    #     x_test_filtered, y_test_filtered = filter_nerq(x_test, y_test, config, 500, train=False)
    # else:
    #     x_train_filtered, y_train_filtered = filter_class(x_train, y_train, config, train=True)
    #     x_test_filtered, y_test_filtered = filter_class(x_test, y_test, config, train=False)
    #
    # print("[INFO] Training image shape: ", x_train_filtered.shape)
    # print("[INFO] Test image shape: ", x_test_filtered.shape)
    # if config.TRANSFORMATION != "Farhi" and config.MEASUREMENT != 'single':
    #     y_train_filtered = to_categorical(y_train_filtered, num_classes)
    #     y_test_filtered = to_categorical(y_test_filtered, num_classes)
    #
    # print("[INFO] Training Label Shape", y_train_filtered.shape)
    # print("[INFO] Test Label Shape", y_test_filtered.shape)

    qdnn_model = tf.keras.models.Sequential()

    qdnn_model.add(qnn_layer)

    if config.TRANSFORMATION != "Farhi":
        if config.MEASUREMENT == 'selection':
            qdnn_model.add(tf.keras.layers.Activation('softmax'))

        elif config.MEASUREMENT == 'full':
            qdnn_model.add(tf.keras.layers.Dense(num_classes))
            qdnn_model.add(tf.keras.layers.Activation('softmax'))
    # trainableParams = np.sum([np.prod(v.get_shape()) for v in qdnn_model.trainable_weights])
    print("[INFO] Trainable Params: ", len(qnn_layer.learning_params))


    if config.TRANSFORMATION == "Farhi" or config.MEASUREMENT == 'single':
        qdnn_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(config.LR),
                           metrics=["accuracy"])
    else:
        qdnn_model.compile(optimizer=tf.keras.optimizers.Adam(config.LR), loss='categorical_crossentropy', metrics=['accuracy'])
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)


    qnn_history = qdnn_model.fit(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        epochs=config.NUM_EPOCHES,
        verbose=1,
        validation_data=val_dataset)

    # qnn_results = qdnn_model.evaluate(x_test_filtered, y_test_filtered)
    # print("[INFO] Final Validation Results: ", qnn_results)

    hist_df = pd.DataFrame(qnn_history.history)
    hist_csv_file = os.path.join(config.LOG_DIR, "result.csv")
    # if not os.path.exists(hist_csv_file):
    #     os.makedirs(hist_csv_file)
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

if __name__ == "__main__":
    args = get_args()
    config = parse_config(args.config)
    train(config)