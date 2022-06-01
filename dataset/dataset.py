import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import math

def normalize(x, min=0):
    if min == 0:
        scaler = MinMaxScaler([0, 1])
    else:  # min=-1
        scaler = MinMaxScaler((-1, 1))
    norm_x = scaler.fit_transform(x)
    norm_x = norm_x*np.pi/2
    return norm_x

def filter_class(x, y, classes):

    keep = (y == classes[0])
    for i in range(1, len(classes)):
        keep = np.logical_or(keep, (y==classes[i]))

    keepx = keep.reshape(keep.shape[0])
    x, y = x[keepx], y[keep]
    for i in range(len(classes)):
        y[y==classes[i]] = i
    return x, y

class HandWrittenDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, config, train=True, y_transfrom=None, x_transform=None):
        self.data_path = config.DATA_PATH
        self.train = train
        self.batch_size = config.BATCH_SIZE
        self.config = config
        dataset = sio.loadmat(self.data_path)
        if config.VIEWS is None:
            self.view_numbers = int((len(dataset) - 5) / 2)
        else:
            assert len(config.VIEWS) <= int((len(dataset) - 5) / 2), "The number images should be not higher than maximum views: Found {} exceed {}".format(config.NUM_IMAGES, int((len(dataset) - 5) / 2))
            self.view_numbers = len(config.VIEWS)
        self.X = dict()
        if train:
            y = dataset['gt_train']

            keep = (y == config.CLASSES[0])
            for i in range(1, len(config.CLASSES)):
                keep = np.logical_or(keep, (y == config.CLASSES[i]))


            y = y[keep]
            for i in range(len(config.CLASSES)):
                y[y == config.CLASSES[i]] = i

            idx = np.random.permutation(y.shape[0])
            y = y[idx]
            for v_num in config.VIEWS:
                data_vnum = normalize(dataset['x' + str(v_num + 1) + '_train'])
                # print(type(data_vnum))
                # print((keep))
                self.X[v_num] = data_vnum[np.squeeze(keep)]
                self.X[v_num] = self.X[v_num][idx]
                #self.X.append(normalize(dataset['x' + str(v_num + 1) + '_train']))
            #self.X = np.stack(self.X, axis=0)

        else:
            y = dataset['gt_test']

            keep = (y == config.CLASSES[0])
            for i in range(1, len(config.CLASSES)):
                keep = np.logical_or(keep, (y == config.CLASSES[i]))

            y = y[keep]
            for i in range(len(config.CLASSES)):
                y[y == config.CLASSES[i]] = i
            idx = np.random.permutation(y.shape[0])
            y = y[idx]
            for v_num in config.VIEWS:
                data_vnum = normalize(dataset['x' + str(v_num + 1) + '_test'])
                self.X[v_num] = data_vnum[np.squeeze(keep)]
                self.X[v_num] = self.X[v_num][idx]
                #self.X.append(normalize(dataset['x' + str(v_num + 1) + '_test']))
            #self.X = np.stack(self.X, axis=0)

        # if np.min(y) == 1:
        #     y = y - 1
        tmp = np.zeros(y.shape[0])
        y = np.reshape(y, np.shape(tmp))
        # y = y-1
        # print(y)
        # print(config.CLASSES)
        # print(y.shape)
        # print(type(y))

        self.y = to_categorical(y, num_classes=len(config.CLASSES))

    def __len__(self):
        return math.ceil(len(self.X[self.config.VIEWS[0]]) / self.batch_size)

    def __getitem__(self, index):
        data = []
        for v_num in self.config.VIEWS:
            sequence = (np.array(self.X[v_num][index*self.batch_size:(index+1)*self.batch_size]).astype(np.float32))
            sequence_length = sequence.shape[-1]
            if sequence_length < self.config.MAX_LENGTH:
                sequence = np.pad(sequence, ((0,0),(0,self.config.MAX_LENGTH-sequence_length)), constant_values=np.pi/2)
            data.append(sequence)

        data = np.stack(data, axis=0)
        data = np.transpose(data, axes=(1,0,2))# BS, V, C

        data = data[..., np.newaxis]
        target = self.y[index*self.batch_size:(index+1)*self.batch_size]

        return data, target

