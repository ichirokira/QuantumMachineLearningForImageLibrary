import math
import os
import sys
import shutil
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
#import tensorflow_io as tfio
import tensorflow_datasets as tfds

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


def filter_nerq(x, y, classes, num_samples=1000):
    keep0 = (y == classes[0])
    keepx = keep0.reshape(keep0.shape[0])
    x0, y0 = x[keepx], y[keep0]

    keep1 = (y == classes[1])
    keepx = keep1.reshape(keep1.shape[0])
    x1, y1 = x[keepx], y[keep1]

    x = np.concatenate([x0[:num_samples], x1[:num_samples]], 0)
    y = np.concatenate([y0[:num_samples], y1[:num_samples]], 0)
    idx = np.random.permutation(len(x))
    x, y = x[idx], y[idx]
    return x, y



def preprocess_sound(ds, ds_info,type="1", encoder="frqi"):
    neqr_scale = 2**config.MIN_COLOR_QUBITS-1
    num_classes = len(config.CLASSES)

    def normalize_frqi_audio1d(audio, label):
        audio = tf.cast(audio, tf.float32)
        return (audio / tf.reduce_max(audio) + 1.0) / 2. * (np.pi / 2.0), tf.one_hot(label, num_classes)

    def normalize_neqr_audio1d(audio, label):
        audio = tf.cast(audio, tf.float32)
        audio = (audio / tf.reduce_max(audio) + 1.0) / 2. * (neqr_scale)
        audio = tf.cast(audio, tf.uint64)
        return audio, tf.one_hot(label, num_classes)

    def normalize_frqi_audio2d(audio, label):
        audio = tf.cast(audio, tf.float32)
        a = tf.math.abs(
            tf.signal.stft(
                audio,
                frame_length=256,
                frame_step=64,
                fft_length=64,
                window_fn=tf.signal.hann_window,
                pad_end=True,
            )
        )
        a = tf.expand_dims(a, axis=-1)
        a = tf.image.resize(a, size=(32, 32))
        return (a / tf.reduce_max(a)) * (np.pi / 2.0), tf.one_hot(label, num_classes)

    def normalize_neqr_audio2d(audio, label):
        audio = tf.cast(audio, tf.float32)
        a = tf.math.abs(
            tf.signal.stft(
                audio,
                frame_length=256,
                frame_step=64,
                fft_length=64,
                window_fn=tf.signal.hann_window,
                pad_end=True,
            )
        )
        a = tf.expand_dims(a, axis=-1)
        a = tf.image.resize(a, size=(32, 32))
        a = (a / tf.reduce_max(a)) * (neqr_scale)
        a = tf.cast(a, tf.uint64)
        return a, tf.one_hot(label, num_classes)
    if type=="1":
        if encoder == "frqi":
            ds = ds.map(normalize_frqi_audio1d, num_parallel_calls=tf.data.AUTOTUNE)
        elif encoder == "neqr":
            ds = ds.map(normalize_neqr_audio1d, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()
        ds = ds.shuffle(ds_info.splits["train"].num_examples)
        ds = ds.batch(config.BATCH_SIZE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
    else:
        if encoder == "frqi":
            ds = ds.map(normalize_frqi_audio2d, num_parallel_calls=tf.data.AUTOTUNE)
        elif encoder == "neqr":
            ds = ds.map(normalize_neqr_audio2d, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()
        ds = ds.shuffle(ds_info.splits["train"].num_examples)
        ds = ds.batch(config.BATCH_SIZE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def train(config):
    assert config.DATASET == "AUDIO_MNIST", "Please using audio dataset"
    num_classes = len(config.CLASSES)
    ds, ds_info = tfds.load("spoken_digit", split="train", with_info=True, as_supervised=True )
    ds = ds.filter(lambda au, label: label == config.CLASSES[0] or label == config.CLASSES[1])
    if config.ENCODER == "FRQI":
        ds = preprocess_sound(ds, ds_info, type="2", encoder="frqi")
    else:
        ds = preprocess_sound(ds, ds_info, type="2", encoder="neqr")

    if config.ENCODER == 'FRQI':
        qnn_layer = FRQI_Basis(config, image_shape=(32, 32, 1))
    elif config.ENCODER == 'NERQ':
        qnn_layer = NEQR_Basis(config, image_shape=(32, 32, 1), color_qubits=config.MIN_COLOR_QUBITS)

    qdnn_model = tf.keras.models.Sequential()

    qdnn_model.add(qnn_layer)

    if config.TRANSFORMATION != "Farhi":
        if config.MEASUREMENT == 'selection':
            qdnn_model.add(tf.keras.layers.Activation('softmax'))

        elif config.MEASUREMENT == 'full':
            qdnn_model.add(tf.keras.layers.Dense(num_classes))
            qdnn_model.add(tf.keras.layers.Activation('softmax'))

    class ExtendedTensorBoard(tf.keras.callbacks.TensorBoard):
        def _log_gradients(self, epoch):
            writer = self._writers['train']

            with writer.as_default(), tf.GradientTape() as g:
                # here we use test data to calculate the gradients
                features = x_test_filtered[:100]
                y_true = y_test_filtered[:100]
                features = tf.convert_to_tensor(features)
                y_true = tf.convert_to_tensor(y_true)
                g.watch(features)

                y_pred = self.model(features)  # forward-propagation
                loss = self.model.compiled_loss(y_true=y_true, y_pred=y_pred)  # calculate loss
                gradients = g.gradient(loss, self.model.trainable_weights)  # back-propagation

                # In eager mode, grads does not have name, so we get names from model.trainable_weights
                for weights, grads in zip(self.model.trainable_weights, gradients):
                    tf.summary.histogram(
                        weights.name.replace(':', '_') + '_grads', data=grads, step=epoch)

            writer.flush()
        def _log_values(self, epoch):
            writer = self._writers['train']

            with writer.as_default(), tf.GradientTape() as g:
                # here we use test data to calculate the gradients
                features = x_test_filtered[:100]
                qnn = qdnn_model.get_layer(index=0)
                values = qnn(features)

                values = tf.math.reduce_mean(values, axis=0)
                print(values.shape)
                # In eager mode, grads does not have name, so we get names from model.trainable_weights
                for i, v in enumerate(values):
                    tf.summary.histogram("qubits {}".format(i), data=v, step=epoch)

            writer.flush()
        def on_epoch_end(self, epoch, logs=None):
            # This function overwrites the on_epoch_end in tf.keras.callbacks.TensorBoard
            # but we do need to run the original on_epoch_end, so here we use the super function.
            super(ExtendedTensorBoard, self).on_epoch_end(epoch, logs=logs)

            if self.histogram_freq and epoch % self.histogram_freq == 0:
                if config.LOG_GRADIENTS:
                    self._log_gradients(epoch)
                if config.LOG_OUTPUT_VALUES:
                    self._log_values(epoch)

    if config.TRANSFORMATION == "Farhi" or config.MEASUREMENT == 'single':
        qdnn_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(config.LR),
                           metrics=["accuracy"])
    else:
        qdnn_model.compile(optimizer=tf.keras.optimizers.Adam(config.LR), loss='categorical_crossentropy', metrics=['accuracy'])
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)

    log_grads_dir = os.path.join(config.LOG_DIR, "logs")
    if not os.path.exists(log_grads_dir):
        os.makedirs(log_grads_dir)
    callbacks = [ExtendedTensorBoard(log_dir=os.path.join(config.LOG_DIR, "logs"), histogram_freq=1, write_images=True,
                                         update_freq='epoch')]

    qnn_history = qdnn_model.fit(
        ds,
        batch_size=config.BATCH_SIZE,
        epochs=config.NUM_EPOCHES,
        verbose=1, callbacks=callbacks)

    #qnn_results = qdnn_model.evaluate(x_test_filtered, y_test_filtered)
    #print("[INFO] Final Validation Results: ", qnn_results)

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