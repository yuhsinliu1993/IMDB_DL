from __future__ import division, print_function, absolute_import

import tensorflow as tf

from variables import variable


def is_training(is_training=False, session=None):
    """
    Set the graph training mode.

    This is meant to be used to control ops that have different output at training and testing time.
    """

    if not session:
        session = tf.get_default_session()

    init_training_mode()

    if is_training:
        tf.get_collection('is_training_ops')[0].eval(session=session)
    else:
        tf.get_collection('is_training_ops')[1].eval(session=session)


def get_training_mode():
    """
    Returns variable in-use to set training model

    Returns:
        A `Variable` (the training mode holder)
    """

    init_training_mode()
    coll = tf.get_collection('is_training')
    return coll[0]


def init_training_mode():
    """
    Creates `is_training` variable and its ops.
    This op is required if you are using layers such as dropout or batch norm
    """
    coll = tf.get_collection('is_training')
    if len(coll) == 0:
        tr_var = variable("is_training", dtype=tf.bool, shape=[], initializer=tf.constant_initializer(False), trainable=False)
        # 'is_training_ops' stores the ops to update training mode variable
        a = tf.assign(tr_var, True)
        b = tf.assign(tr_var, False)
        tf.add_to_collection('is_training_ops', a)
        tf.add_to_collection('is_training_ops', b)
