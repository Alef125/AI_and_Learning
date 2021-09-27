import tensorflow as tf
import numpy as np


def dropout(neuron_in, prob):
    shape = tf.shape(neuron_in)
    sto_var = tf.random_uniform(shape=shape, minval=0, maxval=1)
    r = tf.cast((sto_var <= prob), tf.float32)
    neuron_out = tf.multiply(neuron_in, r)
    return neuron_out


# Place Holders
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1), name="input_fonts")
y = tf.placeholder(tf.float32, shape=(None, 26), name="labels")
# keep_prob = tf.placeholder(tf.float32)

# Conv. Layer 1: [batch, 32, 32, 1] --> [batch, 32, 32, 32]
with tf.name_scope('conv_layer_1'):
    # 16[ conv 3*3 ] + 16[ conv 5*5 ]
    # Activation: ReLU
    h1_3 = tf.Variable(
        tf.random_normal(shape=(3, 3, 1, 16), mean=0, stddev=0.01),
        name='h1_3')

    h1_5 = tf.Variable(
        tf.random_normal(shape=(5, 5, 1, 16), mean=0, stddev=0.01),
        name='h1_5')

    b_h1 = tf.Variable(
        tf.zeros(shape=[32]),
        name='b_h1')

    conv1_3 = tf.nn.conv2d(input=x, filter=h1_3, strides=[1, 1, 1, 1], padding='SAME')
    conv1_5 = tf.nn.conv2d(input=x, filter=h1_5, strides=[1, 1, 1, 1], padding='SAME')
    # 2 * [Batch_size, 32, 32, 16] --> [Batch_size, 32, 32, 32]
    conv1 = tf.concat(axis=3, values=[conv1_3, conv1_5])
    conv1_out = tf.nn.relu(conv1 + b_h1)


# Batch Normalization on Conv. Layer 1
with tf.name_scope("batch_normalization_1"):
    eps1 = 0.0000001
    mean_b_1, var_b_1 = tf.nn.moments(conv1_out, axes=[0, 1, 2], keep_dims=True)
    sigma_1 = tf.sqrt(eps1 + var_b_1)
    bn1_normalized = tf.divide((conv1_out - mean_b_1), sigma_1)
    gamma1 = tf.Variable(
        tf.random_normal(shape=(1, 1, 1, 32), mean=0, stddev=0.01),
        name='gamma1')
    beta1 = tf.Variable(
        tf.zeros(shape=(1, 1, 1, 32)),
        name='beta1')
    bn1_out = bn1_normalized * gamma1 + beta1


# Conv. Layer 2: [batch, 32, 32, 32]  --> [batch, 32, 32, 48]
with tf.name_scope('conv_layer_2'):
    # 16[ conv 1*1 ] + 16[ conv 3*3 ] + 16[ conv 5*5 ]
    # Activation: ReLU
    h2_1 = tf.Variable(
        tf.random_normal(shape=(1, 1, 32, 16), mean=0, stddev=0.01),
        name='h2_1')

    h2_3 = tf.Variable(
        tf.random_normal(shape=(3, 3, 32, 16), mean=0, stddev=0.01),
        name='h2_3')

    h2_5 = tf.Variable(
        tf.random_normal(shape=(5, 5, 32, 16), mean=0, stddev=0.01),
        name='h2_5')

    b_h2 = tf.Variable(
        tf.zeros(shape=[48]),
        name='b_h2')

    conv2_1 = tf.nn.conv2d(input=bn1_out, filter=h2_1, strides=[1, 1, 1, 1], padding='SAME')
    conv2_3 = tf.nn.conv2d(input=bn1_out, filter=h2_3, strides=[1, 1, 1, 1], padding='SAME')
    conv2_5 = tf.nn.conv2d(input=bn1_out, filter=h2_5, strides=[1, 1, 1, 1], padding='SAME')
    # 3 * [Batch_size, 32, 32, 16] --> [Batch_size, 32, 32, 48]
    conv2 = tf.concat(axis=3, values=[conv2_1, conv2_3, conv2_5])
    conv2_out = tf.nn.relu(conv2 + b_h2)


# Batch Normalization on Conv. Layer 2
with tf.name_scope("batch_normalization_2"):
    eps2 = 0.0000001
    mean_b_2, var_b_2 = tf.nn.moments(conv2_out, axes=[0, 1, 2], keep_dims=True)
    sigma_2 = tf.sqrt(eps2 + var_b_2)
    bn2_normalized = tf.divide((conv2_out - mean_b_2), sigma_2)
    gamma2 = tf.Variable(
        tf.random_normal(shape=(1, 1, 1, 48), mean=0, stddev=0.01),
        name='gamma2')
    beta2 = tf.Variable(
        tf.zeros(shape=(1, 1, 1, 48)),
        name='beta2')
    bn2_out = bn2_normalized * gamma2 + beta2


# Max Pooling 1: [batch, 32, 32, 48] --> [batch, 16, 16, 48]
with tf.name_scope('max_pool_1'):
    max_pool_1 = tf.nn.max_pool(bn2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# Conv. Layer 3: [batch, 16, 16, 48]  --> [batch, 16, 16, 72]
with tf.name_scope('conv_layer_3'):
    # 24[ conv 1*1 ] + 24[ conv 3*3 ] + 24[ conv 5*5 ]
    # Activation: ReLU
    h3_1 = tf.Variable(
        tf.random_normal(shape=(1, 1, 48, 24), mean=0, stddev=0.01),
        name='h3_1')

    h3_3 = tf.Variable(
        tf.random_normal(shape=(3, 3, 48, 24), mean=0, stddev=0.01),
        name='h3_3')

    h3_5 = tf.Variable(
        tf.random_normal(shape=(5, 5, 48, 24), mean=0, stddev=0.01),
        name='h3_5')

    b_h3 = tf.Variable(
        tf.zeros(shape=[72]),
        name='b_h3')

    conv3_1 = tf.nn.conv2d(input=max_pool_1, filter=h3_1, strides=[1, 1, 1, 1], padding='SAME')
    conv3_3 = tf.nn.conv2d(input=max_pool_1, filter=h3_3, strides=[1, 1, 1, 1], padding='SAME')
    conv3_5 = tf.nn.conv2d(input=max_pool_1, filter=h3_5, strides=[1, 1, 1, 1], padding='SAME')
    # 3 * [Batch_size, 32, 32, 16] --> [Batch_size, 32, 32, 48]
    conv3 = tf.concat(axis=3, values=[conv3_1, conv3_3, conv3_5])
    conv3_out = tf.nn.relu(conv3 + b_h3)


# Batch Normalization on Conv. Layer 3
with tf.name_scope("batch_normalization_3"):
    eps3 = 0.0000001
    mean_b_3, var_b_3 = tf.nn.moments(conv3_out, axes=[0, 1, 2], keep_dims=True)
    sigma_3 = tf.sqrt(eps3 + var_b_3)
    bn3_normalized = tf.divide((conv3_out - mean_b_3), sigma_3)
    gamma3 = tf.Variable(
        tf.random_normal(shape=(1, 1, 1, 72), mean=0, stddev=0.01),
        name='gamma3')
    beta3 = tf.Variable(
        tf.zeros(shape=(1, 1, 1, 72)),
        name='beta3')
    bn3_out = bn3_normalized * gamma3 + beta3


# Conv. Layer 4: 72 * 16 * 16  --> 108 * 16 * 16
with tf.name_scope('conv_layer_4'):
    # 36[ conv 1*1 ] + 36[ conv 3*3 ] + 36[ conv 5*5 ]
    # Activation: ReLU
    h4_1 = tf.Variable(
        tf.random_normal(shape=(1, 1, 72, 36), mean=0, stddev=0.01),
        name='h4_1')

    h4_3 = tf.Variable(
        tf.random_normal(shape=(3, 3, 72, 36), mean=0, stddev=0.01),
        name='h4_3')

    h4_5 = tf.Variable(
        tf.random_normal(shape=(5, 5, 72, 36), mean=0, stddev=0.01),
        name='h4_5')

    b_h4 = tf.Variable(
        tf.zeros(shape=[108]),
        name='b_h4')

    conv4_1 = tf.nn.conv2d(input=bn3_out, filter=h4_1, strides=[1, 1, 1, 1], padding='SAME')
    conv4_3 = tf.nn.conv2d(input=bn3_out, filter=h4_3, strides=[1, 1, 1, 1], padding='SAME')
    conv4_5 = tf.nn.conv2d(input=bn3_out, filter=h4_5, strides=[1, 1, 1, 1], padding='SAME')
    # 3 * [Batch_size, 32, 32, 36] --> [Batch_size, 32, 32, 108]
    conv4 = tf.concat(axis=3, values=[conv4_1, conv4_3, conv4_5])
    conv4_out = tf.nn.relu(conv4 + b_h4)


# Batch Normalization on Conv. Layer 4
with tf.name_scope("batch_normalization_4"):
    eps4 = 0.0000001
    mean_b_4, var_b_4 = tf.nn.moments(conv4_out, axes=[0, 1, 2], keep_dims=True)
    sigma_4 = tf.sqrt(eps4 + var_b_4)
    bn4_normalized = tf.divide((conv4_out - mean_b_4), sigma_4)
    gamma4 = tf.Variable(
        tf.random_normal(shape=(1, 1, 1, 108), mean=0, stddev=0.01),
        name='gamma4')
    beta4 = tf.Variable(
        tf.zeros(shape=(1, 1, 1, 108)),
        name='beta4')
    bn4_out = bn4_normalized * gamma4 + beta4


# Max Pooling 2: 108 * 16 * 16 --> 108 * 8 * 8
with tf.name_scope('max_pool_2'):
    max_pool_2 = tf.nn.max_pool(bn4_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


# Flatten: 108 * 8 * 8 --> 6912
with tf.name_scope('flatten_layer'):
    flatten_layer = tf.reshape(max_pool_2, [-1, 6912])


# Fully Connected 1: 6912 --> 1024
with tf.name_scope('fully_connected_1'):
    # Activation: ReLU
    w1 = tf.Variable(
        tf.random_normal(shape=(6912, 1024), mean=0, stddev=0.01),
        name='w1')
    b1 = tf.Variable(tf.zeros(1024), name='b1')
    z1 = tf.matmul(flatten_layer, w1) + b1
    hidden_layer_1 = tf.nn.relu(z1)


# Batch Normalization on Fully Connected 1
with tf.name_scope("batch_normalization_5"):
    eps5 = 0.0000001
    mean_b_5, var_b_5 = tf.nn.moments(hidden_layer_1, axes=[0], keep_dims=True)
    sigma_5 = tf.sqrt(eps5 + var_b_5)
    z1_normalized = tf.divide((hidden_layer_1 - mean_b_5), sigma_5)
    gamma5 = tf.Variable(
        tf.random_normal(shape=(1, 1024), mean=0, stddev=0.01),
        name='gamma5')
    beta5 = tf.Variable(
        tf.zeros(shape=(1, 1024)),
        name='beta5')
    bn5_out = z1_normalized * gamma5 + beta5


# Fully Connected 2: 1024 --> 1024
with tf.name_scope('fully_connected_2'):
    # Activation: ReLU
    w2 = tf.Variable(
        tf.random_normal(shape=(1024, 1024), mean=0, stddev=0.01),
        name='w2')
    b2 = tf.Variable(tf.zeros(1024), name='b2')
    z2 = tf.matmul(bn5_out, w2) + b2
    hidden_layer_2 = tf.nn.relu(z2)


# Batch Normalization on Fully Connected 6
with tf.name_scope("batch_normalization_6"):
    eps6 = 0.0000001
    mean_b_6, var_b_6 = tf.nn.moments(hidden_layer_2, axes=[0], keep_dims=True)
    sigma_6 = tf.sqrt(eps6 + var_b_6)
    z2_normalized = tf.divide((hidden_layer_2 - mean_b_6), sigma_6)
    gamma6 = tf.Variable(
        tf.random_normal(shape=(1, 1024), mean=0, stddev=0.01),
        name='gamma6')
    beta6 = tf.Variable(
        tf.zeros(shape=(1, 1024)),
        name='beta6')
    bn6_out = z2_normalized * gamma6 + beta6


# Fully Connected 3 (Output Layer): 1024 --> 26
with tf.name_scope('fully_connected_3'):
    # Activation: ReLU
    w3 = tf.Variable(
        tf.random_normal(shape=(1024, 26), mean=0, stddev=0.01),
        name='w3')
    b3 = tf.Variable(tf.zeros(26), name='b3')
    z3 = tf.matmul(bn6_out, w3) + b3
    output = z3


# Loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    )


# Optimizer
with tf.name_scope('optimizer'):
    # Adam
    # Learning Rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


# Accuracy
with tf.name_scope('accuracy'):
    true_label = tf.argmax(y, 1)
    pred_label = tf.argmax(output, 1)
    is_correct = tf.equal(true_label, pred_label)
    accuracy_train = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32), name="accuracy_train")


# Generating Test Data

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver()


def selector(images, label):
    # images: num * 32 * 32
    input_images = np.array(images)
    input_images = np.reshape(input_images, newshape=(-1, 32, 32, 1))
    label_num = ord(label) - ord('A')

    with tf.Session(config=config) as sess:
        # Pre Definitions
        sess.run(tf.global_variables_initializer())

        saver.restore(sess=sess, save_path='./Classifier Model/model.ckpt')
        softmax_activations = sess.run(tf.nn.softmax(output), feed_dict={x: input_images})  # 10 * 26 array

        indicated_activations = softmax_activations[:][label_num]

        best_image_index = np.argmax(indicated_activations)

        return images[best_image_index][:][:]
