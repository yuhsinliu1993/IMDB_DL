import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.util.nest import is_sequence

import config
import activations


def _linear(args, output_size, use_bias, bias_start=0.0, reuse=False, weights_init=None, scope=None):
    if args is None or (is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not is_sequence(args):
        args = [args]

    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]

    for shape in shapes:
        total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    with tf.variable_scope(scope or "linear", reuse=reuse) as s:
        weights = tf.get_variable("weights", [total_arg_size, output_size], dtype, initializer=weights_init)
        if len(args) == 1:
            res = tf.matmul(args[0], weights)
        else:
            res = tf.matmul(tf.concat(args, axis=1), weights)
        if not use_bias:
            return res

        biases = tf.get_variable("biases", [output_size], dtype, initializer=tf.constant_initializer(bias_start, dtype=dtype))

    return res + biases


class BasicRNNCell(core_rnn_cell.RNNCell):

    def __init__(self, num_units, activation=tf.nn.tanh, bias=True, weights_init=None, trainable=True, reuse=False):
        self._num_units = num_units

        if isinstance(activation, str):
            self._activation = activations.get(activation)
        elif hasattr(activation, '__call__'):
            self._activation = activation
        else:
            raise ValueError("Invalid Activation")

        self.bias = bias
        self.weights_init = weights_init
        self.trainable = trainable
        self.reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """
        output = new_state = activation(W * input + U * state + b)
        """
        with tf.variable_scope(scope or "basic_rnn_cell"):
            output = self._activation(
                _linear([inputs, state], self._num_units, True, scope=scope)
            )

            with tf.variable_scope('linear', reuse=True):
                self.W = tf.get_variable('weights')
                self.b = tf.get_variable('biases')

        return output, output


class GRUCell(core_rnn_cell.RNNCell):

    def __init__(self, num_units, activation=tf.tanh, inner_activation=tf.sigmoid, bias=True, trainable=True, weights_init=None, reuse=False):
        self._num_units = num_units

        if isinstance(activation, str):
            self._activation = activations.get(activation)
        elif hasattr(activation, '__call__'):
            self._activation = activation
        else:
            raise ValueError("Invalid Activation.")
        if isinstance(inner_activation, str):
            self._inner_activation = activations.get(inner_activation)
        elif hasattr(inner_activation, '__call__'):
            self._inner_activation = inner_activation
        else:
            raise ValueError("Invalid Activation.")

        self.bais = bias
        self.weights_init = weights_init
        self.trainable = trainable
        self.reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "gru_cell"):
            with tf.variable_scope("gates"):
                # Reset Gate and Update Gate
                r, u = tf.split(
                    value=_linear([inputs, state], 2 * self._num_units, True, 1.0, weights_init=self.weights_init, scope=scope),
                    num_or_size_splits=2,
                    axis=1
                )
                r, u = self._inner_activation(r), self._inner_activation(u)

            # r * state: element-wise multiplication
            with tf.variable_scope("candidate"):
                c = self._activation(
                    _linear([inputs, r * state], self._num_units, True, 0., scope=scope, reuse=self.reuse, weights_init=self.weights_init)
                )

            new_h = u * state + (1 - u) * c

            self.W, self.b = list(), list()
            with tf.variable_scope('gates/linear', reuse=True):
                self.W.append(tf.get_variable('weights'))
                self.b.append(tf.get_variable('biases'))
            with tf.variable_scope('candidate/linear', reuse=True):
                self.W.append(tf.get_variable('weights'))
                self.b.append(tf.get_variable('biases'))

        return new_h, new_h


class BasicLSTMCell(core_rnn_cell.RNNCell):

    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, state_is_tuple=True,
                 inner_activation=tf.sigmoid, bias=True, weights_init=None, trainable=True, reuse=False):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self.bias = bias
        self.weights_init = weights_init
        self.trainable = trainable
        self.reuse = reuse

        if isinstance(activation, str):
            self._activation = activations.get(activation)
        elif hasattr(activation, '__call__'):
            self._activation = activation
        else:
            raise ValueError("Invalid Activation.")
        if isinstance(inner_activation, str):
            self._inner_activation = activations.get(inner_activation)
        elif hasattr(inner_activation, '__call__'):
            self._inner_activation = inner_activation
        else:
            raise ValueError("Invalid Activation.")

    @property
    def state_size(self):
        return core_rnn_cell.LSTMStateTuple(self._num_units, self._num_units) if self._state_is_tuple else 2 * self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):

        with tf.variable_scope(scope or "basic_lstm_cell"):
            # split the `state` from previous rnn cell
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(
                    value=state,
                    num_or_size_splits=2,
                    axis=1
                )

            concat = _linear([inputs, h], 4 * self._num_units, True, 0., scope=scope, reuse=self.reuse, weights_init=self.weights_init)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)

            new_c = (c * self._inner_activation(f + self._forget_bias) + self._inner_activation(i) * self._activation(j))

            new_h = self._activation(new_c) * self._inner_activation(o)

            if self._state_is_tuple:
                new_state = core_rnn_cell.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h], axis=1)

            # Retrieve RNN Variables
            with tf.variable_scope('linear', reuse=True):
                self.W = tf.get_variable('weights')
                self.b = tf.get_variable('biases')

            # return: output and state
            return new_h, new_state


class DropoutWrapper(core_rnn_cell.RNNCell):
    """
    Adding dropout to inputs and outputs of the given cell
    p.s. Dropout is never used on the state
    """

    def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0, seed=None):
        if not isinstance(cell, core_rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not a RNNCell")

        if (isinstance(input_keep_prob, float) and not (input_keep_prob >= 0.0 and input_keep_prob <= 1.0)):
            raise ValueError(
                "Parameter input_keep_prob must be between 0 and 1: %d"
                % input_keep_prob)

        if (isinstance(output_keep_prob, float) and not (output_keep_prob >= 0.0 and output_keep_prob <= 1.0)):
            raise ValueError(
                "Parameter output_keep_prob must be between 0 and 1: %d"
                % output_keep_prob)

        self._cell = cell
        self._input_keep_prob = input_keep_prob
        self._output_keep_prob = output_keep_prob
        self._seed = seed

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared dropouts."""

        is_training = config.get_training_mode()

        if (not isinstance(self._input_keep_prob, float) or
                self._input_keep_prob < 1):
            inputs = tf.cond(is_training,
                             lambda: tf.nn.dropout(inputs,
                                                   self._input_keep_prob,
                                                   seed=self._seed),
                             lambda: inputs)

        output, new_state = self._cell(inputs, state)
        if (not isinstance(self._output_keep_prob, float) or
                self._output_keep_prob < 1):
            o = tf.cond(is_training,
                        lambda: tf.nn.dropout(output,
                                              self._output_keep_prob,
                                              seed=self._seed),
                        lambda: output)
        return o, new_state
