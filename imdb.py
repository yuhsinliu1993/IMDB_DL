from __future__ import division, print_function, absolute_import
import argparse
import os
from time import time
import tensorflow as tf
import numpy as np
import rnn_cells
from utils import to_categorical


tf.logging.set_verbosity(tf.logging.INFO)


class Config:
    time_steps = 80
    hidden_unit = 100
    embedding_size = 128
    batch_size = 128
    val_ratio = 0.25
    n_classes = 2
    learning_rate = 1e-4
    keep_prob = 0.8
    n_epochs = 10


def load_data(data_path, validation_ratio=0.2):
    # Load IMDB data
    print("[*] Loading IMDB data ...")

    data = np.load(file=data_path)
    X = data['X_train']
    y = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    # print("Trining data size: %d" % X.shape[0])
    # print("Trining data num of timesteps: %d" % X.shape[1])
    # print("Trining data embedding vector size: %d\n" % X.shape[2])
    # print("Testing data size: %d" % X_test.shape[0])
    # print("Testing data num of timesteps: %d" % X_test.shape[1])
    # print("Tetsing data embedding vector size: %d" % X_test.shape[2])

    N = int(X.shape[0] * (1 - validation_ratio))

    X_train = X[:N]
    y_train = y[:N]
    X_val = X[N:]
    y_val = y[N:]

    return {'train': (X_train, y_train), 'val': (X_val, y_val), 'test': (X_test, y_test)}


def fully_connected(incoming, n_units, bias=True, bias_init='zeros', regularizer=None, weight_decay=0.001, name="FullyConnected"):

    input_shape = incoming.shape
    assert len(input_shape) > 1, "Incoming Tensor shape must be at least 2-D"
    n_inputs = int(np.prod(input_shape[1:]))

    with tf.variable_scope('fc'):
        W = tf.get_variable(name='fc_weights', shape=[n_inputs, n_units], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name='fc_biases', shape=[n_units], dtype=tf.float32, initializer=tf.zeros_initializer())

    inference = incoming
    # If input is not 2d, flatten it.
    if len(input_shape) > 2:
        inference = tf.reshape(inference, [-1, n_inputs])

    return tf.matmul(inference, W) + b


def batch_generator(X, y, batch_size):
    num_train = X.shape[0]
    while True:
        batch_mask = np.random.choice(num_train, batch_size)
        yield X[batch_mask], y[batch_mask]


def evaluate(X, y, X_inputs, Y_inputs, accuracy, num_batches, checkpoint_dir=None, eval_once=True, name=None):
    saver = tf.train.Saver()
    sess = tf.Session()

    best_accuracy = 0.0
    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir)
        except tf.errors.OutOfRangeError as e:
            tf.logging.erorr('Cat\'t restore checkpoint: %s', e)
            continue

        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval at %s', checkpoint_dir)
            continue

        # Restore the parameters
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        print("[*] Evaluating %s model..." % name)

        avg_test_acc = 0.0
        for j in xrange(num_batches):
            X_batch = X[j * Config.batch_size: (j + 1) * Config.batch_size]
            y_batch = y[j * Config.batch_size: (j + 1) * Config.batch_size]

            test_acc = sess.run([accuracy], feed_dict={X_inputs: X_batch, Y_inputs: y_batch})[0]
            avg_test_acc += test_acc / num_batches

            if test_acc > best_accuracy:
                best_accuracy = test_acc

        print("Average test accuracy: %.4f" % avg_test_acc)
        print("Best test accuracy: %.4f" % best_accuracy)

        if eval_once:
            break

        time.sleep(60)


def _get_kwargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cell", type=str,
                        help="Specify the Cell type. (rnn, lstm, gru)", required=True)

    parser.add_argument("-m", "--mode", type=str,
                        help="Specify the mode: train or eval", required=True)

    return vars(parser.parse_args())


def run(**kwargs):

    if not kwargs:
        kwargs = _get_kwargs()

    train_dir = './train/' + kwargs['cell']
    eval_dir = './eval/' + kwargs['cell']
    checkpoint_dir = './ckpt/' + kwargs['cell']

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Load IMDB data
    data_dict = load_data('data/imdb_word_emb.npz', Config.val_ratio)

    training_samples = int(data_dict['train'][0].shape[0] * (1 - Config.val_ratio))
    val_samples = data_dict['train'][0].shape[0] - training_samples
    num_batches = training_samples // Config.batch_size
    num_val_batches = val_samples // Config.batch_size

    # Define placeholders
    # input shape = (batch, time steps, embedding vector size)
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, Config.time_steps, Config.embedding_size], name='inputs')
    # output shape = (batch, num of classes)
    outputs = tf.placeholder(tf.float32, shape=[None, Config.n_classes], name='outputs')

    # Build model
    if kwargs['cell'] == 'rnn':
        rnn_cell = rnn_cells.BasicRNNCell(num_units=Config.hidden_unit)
    elif kwargs['cell'] == 'gru':
        rnn_cell = rnn_cells.GRUCell(num_units=Config.hidden_unit)
    elif kwargs['cell'] == 'lstm':
        rnn_cell = rnn_cells.BasicLSTMCell(num_units=Config.hidden_unit)
    else:
        raise ValueError("You need to specify the RNNCell type in (rnn, lstm, gru)")

    init_state = rnn_cell.zero_state(Config.batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=inputs, initial_state=init_state, time_major=False)
    rnn_outputs = tf.nn.dropout(rnn_outputs, Config.keep_prob)
    logits = fully_connected(rnn_outputs, Config.n_classes)

    # Calculate loss
    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=outputs, logits=logits)
        loss = tf.reduce_mean(cross_entropy)
        loss_summary = tf.summary.scalar("loss", loss)

    # Define training operation
    train_op = tf.train.GradientDescentOptimizer(Config.learning_rate).minimize(loss)

    # Accuracy
    with tf.name_scope("Accuracy"):
        preds = tf.nn.softmax(logits)
        correct_preds = tf.equal(tf.argmax(preds, axis=1), tf.argmax(outputs, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)

    # Convert training and testing label to one_hot vectors
    y_train = to_categorical(data_dict['train'][1], 2)
    y_val = to_categorical(data_dict['val'][1], 2)
    y_test = to_categorical(data_dict['test'][1], 2)

    training_set = batch_generator(data_dict['train'][0], y_train, Config.batch_size)

    if kwargs['mode'] == 'train':
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            train_summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
            eval_summary_writer = tf.summary.FileWriter(eval_dir, sess.graph)

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            # restore session
            if ckpt and ckpt.model_checkpoint_path:
                tf.logging.info('Loading checkpoint %s', ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                tf.logging.info('Create new session')
                sess.run(tf.global_variables_initializer())

            for epoch in xrange(Config.n_epochs):
                print("Epoch %d:" % (epoch + 1))
                avg_training_loss = 0.0
                avg_val_acc = 0.0
                avg_test_acc = 0.0

                for j in xrange(num_batches):
                    X_batch, y_batch = training_set.next()
                    _, _loss, loss_summ = sess.run([train_op, loss, loss_summary], feed_dict={inputs: X_batch, outputs: y_batch})
                    train_summary_writer.add_summary(loss_summ, epoch * num_batches + j)
                    avg_training_loss += _loss / Config.batch_size

                for j in xrange(num_val_batches):
                    start = j * Config.batch_size
                    end = (j + 1) * Config.batch_size
                    X_val_batch = data_dict['val'][0][start:end]
                    y_val_batch = y_val[start:end]
                    val_acc, val_acc_summary = sess.run([accuracy, accuracy_summary], feed_dict={inputs: X_val_batch, outputs: y_val_batch})
                    train_summary_writer.add_summary(val_acc_summary, epoch * num_batches + j)
                    avg_val_acc += val_acc / num_val_batches

                saver.save(sess, checkpoint_dir + '/xd.ckpt', epoch)

                for j in xrange(num_batches):
                    start = j * Config.batch_size
                    end = (j + 1) * Config.batch_size
                    X_test_batch = data_dict['test'][0][start:end]
                    y_test_batch = y_test[start:end]

                    test_acc, test_acc_summary = sess.run([accuracy, accuracy_summary], feed_dict={inputs: X_test_batch, outputs: y_test_batch})
                    avg_test_acc += test_acc / num_batches
                eval_summary_writer.add_summary(test_acc_summary, epoch * num_batches)

                print("Average Loss: %.4f\tValidation Accuracy: %.4f\tTest Accuracy: %.4f" % (avg_training_loss, avg_val_acc, avg_test_acc))
    elif kwargs['mode'] == 'eval':
        evaluate(data_dict['test'][0], y_test, inputs, outputs, accuracy, num_batches, checkpoint_dir, True, kwargs['cell'])
    else:
        pass


run()
