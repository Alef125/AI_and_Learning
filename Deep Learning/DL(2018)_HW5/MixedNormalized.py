import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# import matplotlib.pyplot as plt


def dropout(neuron_in, prob):
    shape = tf.shape(neuron_in)
    sto_var = tf.random_uniform(shape=shape, minval=0, maxval=1)
    r = tf.cast((sto_var <= prob), tf.float32)
    neuron_out = tf.multiply(neuron_in, r)
    return neuron_out


mnist = input_data.read_data_sets("mnist", one_hot=True)
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="input_image")
y = tf.placeholder(tf.float32, shape=(None, 10), name="labels")
keep_prob_input = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
in_existence = tf.placeholder(tf.float32)
existence = tf.placeholder(tf.float32)

with tf.name_scope("conv_layer_1"):
    # 28 * 28 ---> 24 * 24 * 64
    # Activation: ReLU
    h1 = tf.Variable(
            tf.random_normal(shape=(5, 5, 1, 64), mean=0, stddev=0.01),
            # tf.random_uniform(shape=(5, 5, 1, 64), minval=-1, maxval=1),
            name='h1')
    b_h1 = tf.Variable(
        tf.zeros(shape=[64]),
        name='b_h1')
    x_dropped = dropout(x, prob=keep_prob_input)
    conv1 = tf.nn.conv2d(input=x_dropped * in_existence, filter=h1, strides=[1, 1, 1, 1], padding='VALID')
    conv1_out = tf.nn.relu(conv1 + b_h1)

with tf.name_scope("batch_normalization_1_and_dropout"):
    eps1 = 0.0000001
    mean_b_1, var_b_1 = tf.nn.moments(conv1_out, axes=[0, 1, 2], keep_dims=True)
    sigma_1 = tf.sqrt(eps1 + var_b_1)
    bn1_normalized = tf.divide((conv1_out - mean_b_1), sigma_1)
    gamma1 = tf.Variable(
        tf.random_normal(shape=(1, 1, 1, 64), mean=0, stddev=0.01),
        name='gamma1')
    beta1 = tf.Variable(
        tf.zeros(shape=(1, 1, 1, 64)),
        name='beta1')
    bn1_out = existence * dropout((bn1_normalized * gamma1 + beta1), prob=keep_prob)

with tf.name_scope("max_pooling_1"):
    # 24 * 24 * 64 --> 12 * 12 * 64
    max_pool_1 = tf.nn.max_pool(bn1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

with tf.name_scope("conv_layer_2"):
    # 12 * 12 * 64 --> 8 * 8 * 64
    # Activation: ReLU
    h2 = tf.Variable(
        tf.random_normal(shape=(5, 5, 64, 64), mean=0, stddev=0.01),
        # tf.random_uniform(shape=(5, 5, 64, 64), minval=-1, maxval=1),
        name='h2')
    b_h2 = tf.Variable(
        tf.zeros(shape=[64]),
        name='b_h2')
    conv2 = tf.nn.conv2d(input=max_pool_1, filter=h2, strides=[1, 1, 1, 1], padding='VALID')
    conv2_out = tf.nn.relu(conv2 + b_h2)

with tf.name_scope("batch_normalization_2_and_dropout"):
    eps2 = 0.0000001
    mean_b_2, var_b_2 = tf.nn.moments(conv2_out, axes=[0, 1, 2], keep_dims=True)
    sigma_2 = tf.sqrt(eps2 + var_b_2)
    bn2_normalized = tf.divide((conv2_out - mean_b_2), sigma_2)
    gamma2 = tf.Variable(
        tf.random_normal(shape=(1, 1, 1, 64), mean=0, stddev=0.01),
        name='gamma2')
    beta2 = tf.Variable(
        tf.zeros(shape=(1, 1, 1, 64)),
        name='beta2')
    bn2_out = existence * dropout((bn2_normalized * gamma2 + beta2), prob=keep_prob)


with tf.name_scope("max_pooling_2"):
    # 8 * 8 * 64 --> 4 * 4 * 64
    max_pool_2 = tf.nn.max_pool(bn2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

with tf.name_scope("fully_connected_layer"):
    # 4 * 4 * 64 = 1024 --> 256
    # Activation: ReLU
    w1 = tf.Variable(
        tf.random_normal(shape=(1024, 256), mean=0, stddev=0.01),
        # tf.random_uniform(shape=(1024, 256), minval=-1, maxval=1),
        name='w1')
    max_pool_2_linear = tf.reshape(max_pool_2, [-1, 1024])
    b1 = tf.Variable(tf.zeros(256), name='b1')
    z1 = tf.matmul(max_pool_2_linear, w1) + b1
    hidden_layer_out = existence * dropout(tf.nn.relu(z1), prob=keep_prob)

with tf.name_scope("output_layer"):
    # 256 --> 10
    # SoftMax
    w2 = tf.Variable(
        tf.random_normal(shape=(256, 10), mean=0, stddev=0.01),
        # tf.random_uniform(shape=(256, 10), minval=-1, maxval=1),
        name='w2')
    # b2 = tf.Variable(tf.zeros(10), name='b1')
    z2 = tf.matmul(hidden_layer_out, w2)

with tf.name_scope("batch_normalization_3"):
    eps3 = 0.0000001
    mean_b_3, var_b_3 = tf.nn.moments(z2, axes=[0], keep_dims=True)
    sigma_3 = tf.sqrt(eps3 + var_b_3)
    z2_normalized = tf.divide((z2 - mean_b_3), sigma_3)
    gamma3 = tf.Variable(
        tf.random_normal(shape=(1, 10), mean=0, stddev=0.01),
        name='gamma3')
    beta3 = tf.Variable(
        tf.zeros(shape=(1, 10)),
        name='beta3')
    bn3_out = z2_normalized * gamma3 + beta3
    output = bn3_out

with tf.name_scope("loss"):
    # Cross Entropy
    # reg_conv_coef1 = 0.01
    # reg_conv_coef2 = 0.01
    # reg_coeff1 = 0.01
    # reg_coeff2 = 0.01
    # reg_conv_loss1 = reg_conv_coef1 * tf.nn.l2_loss(h1)
    # reg_conv_loss2 = reg_conv_coef2 * tf.nn.l2_loss(h2)
    # reg_loss1 = reg_coeff1 * tf.nn.l2_loss(w1)
    # reg_loss2 = reg_coeff2 * tf.nn.l2_loss(w2)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    )

with tf.name_scope("optimizer"):
    # SGD
    # Learning Rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

with tf.name_scope(name="accuracy"):
    true_label = tf.argmax(y, 1)
    pred_label = tf.argmax(output, 1)
    is_correct = tf.equal(true_label, pred_label)
    accuracy_train = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32), name="accuracy_train")

# with tf.name_scope(name="save"):
    # tf.summary.scalar("Loss", loss)
    # tf.summary.scalar("Accuracy", accuracy_train)
    # tf.summary.histogram("h1", h1)
    # tf.summary.histogram("W1", w1)
    # tf.summary.histogram("W2", w2)
    # merge = tf.summary.merge_all()
    # file_writer = tf.summary.FileWriter("Monitoring BN CNN")
    # saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_size = 64
    check_batch = 100
    input_prob = 0.8
    neuron_prob = 0.5
    for i in range(2000):
        x_sample, y_sample = mnist.train.next_batch(batch_size=batch_size)
        # x_sample = tf.reshape(x_sample, shape=(batch_size, 28, 28, 1))
        x_sample = np.reshape(x_sample, newshape=(batch_size, 28, 28, 1))
        sess.run(optimizer, feed_dict={x: x_sample, y: y_sample,
                                       keep_prob_input: input_prob, keep_prob: neuron_prob,
                                       in_existence: 1.0, existence: 1.0})
        if i % 200 == 0:
            x_sample, y_sample = mnist.train.next_batch(batch_size=check_batch)
            # x_sample = tf.reshape(x_sample, shape=(1, 28, 28, 1))
            x_sample = np.reshape(x_sample, newshape=(check_batch, 28, 28, 1))
            # b = sess.run(merge, feed_dict={x: x_sample, y: y_sample})
            # file_writer.add_summary(b, i)
            current_loss, accuracy = sess.run((loss, accuracy_train), feed_dict={x: x_sample, y: y_sample,
                                                                                 keep_prob_input: 1.0, keep_prob: 1.0,
                                                                                 in_existence: input_prob,
                                                                                 existence: neuron_prob})
            print ("step %5i accuracy is %.5f and cross_entropy(loss) is %.5f" % (i, accuracy, current_loss))
    x_sample, y_sample = mnist.validation.next_batch(batch_size=check_batch)
    # x_sample = tf.reshape(x_sample, shape=(1, 28, 28, 1))
    x_sample = np.reshape(x_sample, newshape=(check_batch, 28, 28, 1))
    accuracy_test = sess.run(accuracy_train, feed_dict={x: x_sample, y: y_sample,
                                                        keep_prob_input: 1.0, keep_prob: 1.0,
                                                        in_existence: input_prob, existence: neuron_prob})
    print "Test Accuracy: ", accuracy_test
    # file_writer.add_graph(sess.graph)
