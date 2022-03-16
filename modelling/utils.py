import numpy as np
import tensorflow as tf


def preprocessFRQI(x):
    x = x/255.0
    x = x*np.pi/2
    return x
def preprocessNEQR(x, new_scale=255.0):
    x = x/255.0*new_scale
    x = tf.cast(x, tf.uint8)
    return x



