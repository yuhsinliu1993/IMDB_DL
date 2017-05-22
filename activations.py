import tensorflow as tf
from utils import get_from_module


def get(identifier):
    if hasattr(identifier, '__call__'):
        return identifier
    else:
        return get_from_module(identifier, globals(), 'activation')


def linear(x):
    return x


def tanh(x):
    return tf.tanh(x)


def softmax(x):
    return tf.nn.softmax(x)


def sigmoid(x):
    return tf.nn.sigmoid(x)


def relu(x):
        # Computes rectified linear: `max(features, 0)`.
    return tf.nn.relu(x)


def relu6(x):
    # Computes Rectified Linear 6: `min(max(features, 0), 6)`.
    return tf.nn.relu6(x)


def leaky_relu(x, alpha=0.1, name="LeakyReLU"):
    i_scope = ""
    if hasattr(x, 'scope'):
        if x.scope:
            i_scope = x.scope
    with tf.name_scope(i_scope + name) as scope:
        x = tf.nn.relu(x)
        m_x = tf.nn.relu(-x)
        x -= alpha * m_x

    x.scope = scope

    return x
