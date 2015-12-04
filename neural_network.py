import numpy as np
import tensorflow as tf


def one_hot_encoding(labels):
    return np.array([(1.0, 0.0) if l == 1 else (0.0, 1.0) for l in labels])

def softmax(x):
     ee = np.exp(x)
     return ee / np.sum(ee)

def make_probs(l):
    return np.array([softmax(row) for row in l])

def one_hidden_layer(data, weights_h, weights_out, p_in, p_h):
    data = tf.nn.dropout(data, p_in)
    h_layer = tf.nn.relu(tf.matmul(data, weights_h))
    h_layer = tf.nn.dropout(h_layer, p_h)
    return tf.matmul(h_layer, weights_out)

def two_hidden_layer(data, weights_h1, weights_h2, weights_out, p_in, p_h):
    data = tf.nn.dropout(data, p_in)
    h1 = tf.nn.relu(tf.matmul(data, weights_h1))
    h1 = tf.nn.dropout(h1, p_h)
    h2 = tf.nn.relu(tf.matmul(h1, weights_h2))
    h2 = tf.nn.dropout(h2, p_h)
    return tf.matmul(h2, weights_out)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def run_network(train_x, train_y, test_x, test_y, n_layers, layers, m_batch, learning_rate, n_epochs):

    data_holder = tf.placeholder("float", [None, 4])
    label_holder = tf.placeholder("float", [None, 2])
    p_in = tf.placeholder("float")
    p_h = tf.placeholder("float")

    if n_layers == 1:
        weights_h1 = init_weights([4, layers[0]])
        weights_out = init_weights([layers[0], 2])
        network = one_hidden_layer(data_holder, weights_h1, weights_out, p_in, p_h)
    elif n_layers == 2:
        weights_h1 = init_weights([4, layers[0]])
        weights_h2 = init_weights([layers[0], layers[1]])
        weights_out = init_weights([layers[1], 2])
        network = two_hidden_layer(data_holder, weights_h1, weights_h2,
                                   weights_out, p_in, p_h)


    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(network, label_holder))
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

    train_y = one_hot_encoding(train_y)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    for i in range(n_epochs):
        for start, end in zip(range(0, len(train_x), m_batch), range(m_batch, len(train_x), m_batch)):
            sess.run(training_step, feed_dict={data_holder: train_x[start:end],
                                          label_holder: train_y[start:end],
                                          p_in: 1.,
                                          p_h: 0.5})

    return make_probs(sess.run(network, feed_dict={data_holder: test_x,
                                                p_in: 1.0,
                                                p_h: 1.0}))[:, 0]
