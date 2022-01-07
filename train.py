import math
import os
import sys
import shutil
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

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

def filter_class(x, y, classes):

    keep = (y == classes[0])
    for i in range(1, len(classes)):
        keep = np.logical_or(keep, (y==classes[i]))

    keepx = keep.reshape(keep.shape[0])
    x, y = x[keepx], y[keep]

    return x, y

def train(config):
    if config.DATASET == 'CIFAR10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        x_train = tf.image.rgb_to_grayscale(x_train)
        x_test = tf.image.rgb_to_grayscale(x_test)
    elif config.DATASET == 'MNIST':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]

    N, H, W, C = x_train.shape
    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)

    print("Number of original training examples", len(x_train))
    print("Number of original test examples", len(x_test))

    if config.ENCODER == 'FRQI':
        num_qubits_row = (math.ceil(math.log2(H)))
        num_qubits_col = (math.ceil(math.log2(W)))
        num_qubits = num_qubits_col+num_qubits_row +1

        if num_qubits > config.MAX_NUM_QUBITS:
            """Since FRQI only requires 1 qubit for color, to reduce number of used qubits, we need to resize resolution 
                of image"""
            max_num_position_qubits = num_qubits-config.MAX_NUM_QUBITS
            rescaled_qubits = math.floor(max_num_position_qubits/2)
            if rescaled_qubits < 1:
                rescaled_qubits = 1

            print("[INFO] Require {} qubits excess {}".format(num_qubits, config.MAX_NUM_QUBITS))

            print("[INFO] Reshape images from {} to {}".format([H,W], [2**(num_qubits_row-rescaled_qubits), 2**(num_qubits_col-rescaled_qubits)]))
            x_train = tf.image.resize(x_train[:], (2**(num_qubits_row-rescaled_qubits), 2**(num_qubits_col-rescaled_qubits)))
            x_test = tf.image.resize(x_test[:], (2**(num_qubits_row-rescaled_qubits), 2**(num_qubits_col-rescaled_qubits)))
            N, H, W, C = x_train.shape

        x_train = preprocessFRQI(x_train)
        x_test = preprocessFRQI(x_test)
        qnn_layer = FRQI_Basis(config, image_shape=(H, W, C))
    elif config.ENCODER == 'NERQ':
        num_qubits_row = (math.ceil(math.log2(H)))
        num_qubits_col = (math.ceil(math.log2(W)))
        color_qubits = 8
        num_qubits = num_qubits_col + num_qubits_row + color_qubits
        new_scale = 255.0
        if num_qubits > config.MAX_NUM_QUBITS:
            print("[INFO] Require {} qubits excess {}".format(num_qubits, config.MAX_NUM_QUBITS))
            removed_qubits = (num_qubits - config.MAX_NUM_QUBITS) // 3

            if num_qubits_row+num_qubits_col - 2*removed_qubits < config.MIN_POS_QUBITS:
                """if number of position qubits is smaller than threshold, rescale color"""

                if (color_qubits - 3*removed_qubits) > config.MIN_COLOR_QUBITS:
                    color_qubits -= 3*removed_qubits
                else:
                    color_qubits = config.MIN_POS_QUBITS
                new_scale = 2**color_qubits-1
                print("[INFO] Rescale Color Range to {}".format(new_scale))
            else:
                x_train = tf.image.resize(x_train[:], (2 ** (num_qubits_row - removed_qubits), 2 ** (num_qubits_col - removed_qubits)))
                x_test = tf.image.resize(x_test[:], (2 ** (num_qubits_row - removed_qubits), 2 ** (num_qubits_col - removed_qubits)))

                if (color_qubits - removed_qubits) > config.MIN_COLOR_QUBITS:
                    color_qubits -= removed_qubits
                else:
                    color_qubits = config.MIN_POS_QUBITS
                new_scale = 2**color_qubits-1
                print("[INTO] Resize image from {} to {}. Rescale Color Range to {}".format([H, W],
                                                                                         [2 ** (num_qubits_row - removed_qubits), 2 ** (num_qubits_col - removed_qubits)],
                                                                                         new_scale))
                N, H, W, C = x_train.shape
        x_train = preprocessNEQR(x_train, new_scale=new_scale)
        x_test = preprocessNEQR(x_test, new_scale=new_scale)
        qnn_layer = NEQR_Basis(config, image_shape=(H, W, C), color_qubits=color_qubits)

    num_classes = len(config.CLASSES)
    # x_train = x_train.numpy()
    # x_test = x_test.numpy()
    x_train_filtered, y_train_filtered = filter_class(x_train, y_train, config.CLASSES)
    x_test_filtered, y_test_filtered = filter_class(x_test, y_test, config.CLASSES)

    print("[INFO] Training image shape: ", x_train_filtered.shape)
    print("[INFO] Test image shape: ", x_test_filtered.shape)
    y_train_filtered = to_categorical(y_train_filtered, num_classes)
    y_test_filtered = to_categorical(y_test_filtered, num_classes)

    print("[INFO] Training Label Shape", y_train_filtered.shape)
    print("[INFO] Test Label Shape", y_test_filtered.shape)

    qdnn_model = tf.keras.models.Sequential()

    qdnn_model.add(qnn_layer)
    qdnn_model.add(tf.keras.layers.Dense(num_classes))
    qdnn_model.add(tf.keras.layers.Activation('softmax'))

    class ExtendedTensorBoard(tf.keras.callbacks.TensorBoard):
        def _log_gradients(self, epoch):
            writer = self._writers['train']

            with writer.as_default(), tf.GradientTape() as g:
                # here we use test data to calculate the gradients
                features = x_test_filtered[:100]
                y_true = y_test_filtered[:100]
                # features = tf.convert_to_tensor(features)
                # y_true = tf.convert_to_tensor(y_true)
                g.watch(features)

                y_pred = self.model(features)  # forward-propagation
                loss = self.model.compiled_loss(y_true=y_true, y_pred=y_pred)  # calculate loss
                gradients = g.gradient(loss, self.model.trainable_weights)  # back-propagation

                # In eager mode, grads does not have name, so we get names from model.trainable_weights
                for weights, grads in zip(self.model.trainable_weights, gradients):
                    tf.summary.histogram(
                        weights.name.replace(':', '_') + '_grads', data=grads, step=epoch)

            writer.flush()

        def on_epoch_end(self, epoch, logs=None):
            # This function overwrites the on_epoch_end in tf.keras.callbacks.TensorBoard
            # but we do need to run the original on_epoch_end, so here we use the super function.
            super(ExtendedTensorBoard, self).on_epoch_end(epoch, logs=logs)

            if self.histogram_freq and epoch % self.histogram_freq == 0:
                self._log_gradients(epoch)


    qdnn_model.compile(optimizer=tf.keras.optimizers.Adam(config.LR), loss='categorical_crossentropy', metrics=['accuracy'])
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)
    if config.LOG_GRADIENTS:
        log_grads_dir = os.path.join(config.LOG_DIR, "logs_grads")
        if not os.path.exists(log_grads_dir):
            os.makedirs(log_grads_dir)
        callbacks = [ExtendedTensorBoard(log_dir=os.path.join(config.LOG_DIR, "logs_grads"), histogram_freq=1, write_images=True,
                                         update_freq='epoch')]
    else:
        callbacks = None

    qnn_history = qdnn_model.fit(
        x_train_filtered, y_train_filtered,
        batch_size=config.BATCH_SIZE,
        epochs=config.NUM_EPOCHES,
        verbose=1,
        validation_data=(x_test_filtered, y_test_filtered), callbacks=callbacks)

    qnn_results = qdnn_model.evaluate(x_test_filtered, y_test_filtered)
    print("[INFO] Final Validation Results: ", qnn_results)

    hist_df = pd.DataFrame(qnn_history.history)
    hist_csv_file = os.path.join(config.LOG_DIR, "result.csv")
    if not os.path.exists(hist_csv_file):
        os.makedirs(hist_csv_file)
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

if __name__ == "__main__":
    args = get_args()
    config = parse_config(args.config)
    train(config)