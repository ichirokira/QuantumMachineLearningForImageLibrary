import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
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

        dataset = sio.loadmat(data_path)
        if config.NUM_IMAGES is None:
            self.view_numbers = int((len(dataset) - 5) / 2)
        else:
            assert config.NUM_IMAGES <= int((len(dataset) - 5) / 2), "The number images should be not higher than maximum views: Found {} exceed {}".format(config.NUM_IMAGES, int((len(dataset) - 5) / 2))
            self.view_numbers = config.NUM_IMAGES
        self.X = []
        if train:
            y = dataset['gt_train']

            keep = (y == config.CLASSES[0])
            for i in range(1, len(config.CLASSES)):
                keep = np.logical_or(keep, (y == config.CLASSES[i]))
            np.random.shuffle(keep)
            y = y[keep]
            for i in range(len(config.CLASSES)):
                y[y == classes[i]] = i

            for v_num in range(self.view_numbers):
                data_vnum = normalize(dataset['x' + str(v_num + 1) + '_train'])
                self.X[v_num] = data_vnum[keep]
                #self.X.append(normalize(dataset['x' + str(v_num + 1) + '_train']))
            #self.X = np.stack(self.X, axis=0)

        else:
            y = dataset['gt_test']

            keep = (y == config.CLASSES[0])
            for i in range(1, len(config.CLASSES)):
                keep = np.logical_or(keep, (y == config.CLASSES[i]))
            np.random.shuffle(keep)
            y = y[keep]
            for i in range(len(config.CLASSES)):
                y[y == classes[i]] = i

            for v_num in range(self.view_numbers):
                data_vnum = normalize(dataset['x' + str(v_num + 1) + '_test'])
                self.X[v_num] = data_vnum[keep]
                #self.X.append(normalize(dataset['x' + str(v_num + 1) + '_test']))
            #self.X = np.stack(self.X, axis=0)

        # if np.min(y) == 1:
        #     y = y - 1
        tmp = np.zeros(y.shape[0])
        y = np.reshape(y, np.shape(tmp))
        self.y = y

    def __len__(self):
        return math.ceil(len(self.X[0]) / self.batch_size)

    def __getitem__(self, index):
        data = []
        for v_num in range(len(self.X)):
            data.append((self.X[v_num][index*self.batch_size:(index+1)*self.batch_size]).astype(np.float32))

        data = np.stack(data, axis=0)
        data = np.transpose(data, axis=(1,0,2))# BS, V, C
        data = data[..., np.newaxis]
        target = self.y[index*self.batch_size:(index+1)*self.batch_size]

        return data, target

