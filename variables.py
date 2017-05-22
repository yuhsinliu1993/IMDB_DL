from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn

from tensorflow.contrib.framework.python.ops import add_arg_scope as contrib_add_arg_scope
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope


@contrib_add_arg_scope
def variable(name, shape=None, dtype=tf.float32, initializer=None,
             regularizer=None, trainable=True, collections=None,
             caching_device=None, validate_shape=True, device=None,
             restore=True):
    """ variable.

    Instantiate a new variable.

    Arguments:
        name: `str`. A name for this variable.
        shape: list of `int`. The variable shape (optional).
        dtype: `type`. The variable data type.
        initializer: `str` or `Tensor`. The variable initialization. (See
            tflearn.initializations for references).
        regularizer: `str` or `Tensor`. The variable regularizer. (See
            tflearn.losses for references).
        trainable: `bool`. If True, this variable weights will be trained.
        collections: `str`. A collection to add the new variable to (optional).
        caching_device: `str`. Optional device string or function describing
            where the Variable should be cached for reading.  Defaults to the
            Variable's device.
        validate_shape: `bool`. Validate or not shape when restoring.
        device: `str`. Optional device ID to store the variable.
        restore: `bool`. Restore or not this variable when loading a
            pre-trained model (Only compatible with tflearn pre-built
            training functions).

    Returns:
        A Variable.

    """

    if isinstance(initializer, str):
        initializer = tflearn.initializations.get(initializer)()
    # Remove shape param if initializer is a Tensor
    if not callable(initializer) and isinstance(initializer, tf.Tensor):
        shape = None

    if isinstance(regularizer, str):
        regularizer = tflearn.losses.get(regularizer)

    collections = set(collections or [])
    collections |= set([ops.GraphKeys.GLOBAL_VARIABLES,
                        ops.GraphKeys.MODEL_VARIABLES])

    with ops.device(device or ''):
        var = variable_scope.get_variable(name, shape=shape, dtype=dtype,
                                          initializer=initializer,
                                          regularizer=regularizer,
                                          trainable=trainable,
                                          collections=collections,
                                          caching_device=caching_device,
                                          validate_shape=validate_shape)

    if not restore:
        tf.add_to_collection(tf.GraphKeys.EXCL_RESTORE_VARS, var)

    return var
