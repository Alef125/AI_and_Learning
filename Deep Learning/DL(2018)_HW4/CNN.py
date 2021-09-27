import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("mnist", one_hot=True)
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="input_image")
y = tf.placeholder(tf.float32, shape=(None, 10), name="labels")

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
    conv1 = tf.nn.conv2d(input=x, filter=h1, strides=[1, 1, 1, 1], padding='VALID')
    conv1_out = tf.nn.relu(conv1 + b_h1)

with tf.name_scope("max_pooling_1"):
    # 24 * 24 * 64 --> 12 * 12 * 64
    max_pool_1 = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

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

with tf.name_scope("max_pooling_2"):
    # 8 * 8 * 64 --> 4 * 4 * 64
    max_pool_2 = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

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
    hidden_layer_out = tf.nn.relu(z1)

with tf.name_scope("output_layer"):
    # 256 --> 10
    # SoftMax
    w2 = tf.Variable(
        tf.random_normal(shape=(256, 10), mean=0, stddev=0.01),
        # tf.random_uniform(shape=(256, 10), minval=-1, maxval=1),
        name='w2')
    b2 = tf.Variable(tf.zeros(10), name='b1')
    z2 = tf.matmul(hidden_layer_out, w2) + b2
    output = z2

with tf.name_scope("loss"):
    # Cross Entropy
    reg_conv_coef1 = 0.01
    reg_conv_coef2 = 0.01
    reg_coeff1 = 0.01
    reg_coeff2 = 0.01
    reg_conv_loss1 = reg_conv_coef1 * tf.nn.l2_loss(h1)
    reg_conv_loss2 = reg_conv_coef2 * tf.nn.l2_loss(h2)
    reg_loss1 = reg_coeff1 * tf.nn.l2_loss(w1)
    reg_loss2 = reg_coeff2 * tf.nn.l2_loss(w2)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output) +
        reg_conv_loss1 +
        reg_conv_loss2 +
        reg_loss1 +
        reg_loss2
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

with tf.name_scope(name="save"):
    tf.summary.scalar("Loss", loss)
    tf.summary.scalar("Accuracy", accuracy_train)
    tf.summary.histogram("h1", h1)
    tf.summary.histogram("W1", w1)
    tf.summary.histogram("W2", w2)
    merge = tf.summary.merge_all()
    file_writer = tf.summary.FileWriter("Monitoring CNN")
    saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_size = 1
    check_batch = 100
    for i in range(40000):
        x_sample, y_sample = mnist.train.next_batch(batch_size=batch_size)
        # x_sample = tf.reshape(x_sample, shape=(batch_size, 28, 28, 1))
        x_sample = np.reshape(x_sample, newshape=(batch_size, 28, 28, 1))
        sess.run(optimizer, feed_dict={x: x_sample, y: y_sample})
        if i % 2000 == 0:
            x_sample, y_sample = mnist.train.next_batch(batch_size=check_batch)
            # x_sample = tf.reshape(x_sample, shape=(1, 28, 28, 1))
            x_sample = np.reshape(x_sample, newshape=(check_batch, 28, 28, 1))
            b = sess.run(merge, feed_dict={x: x_sample, y: y_sample})
            file_writer.add_summary(b, i)
            current_loss, accuracy = sess.run((loss, accuracy_train), feed_dict={x: x_sample, y: y_sample})
            print ("step %5i accuracy is %.5f and cross_entropy(loss) is %.5f" % (i, accuracy, current_loss))
    x_sample, y_sample = mnist.validation.next_batch(batch_size=check_batch)
    # x_sample = tf.reshape(x_sample, shape=(1, 28, 28, 1))
    x_sample = np.reshape(x_sample, newshape=(check_batch, 28, 28, 1))
    accuracy_test = sess.run(accuracy_train, feed_dict={x: x_sample, y: y_sample})
    print "Test Accuracy: ", accuracy_test
    file_writer.add_graph(sess.graph)

    # Figuring Weights
    fig1 = plt.figure(figsize=(40, 40))
    fig1.subplots_adjust(wspace=0, hspace=0)
    kernel1 = np.reshape(sess.run(h1).T, newshape=(64, 5, 5))
    # kernel1 = sess.run(h1)
    # print(np.size(kernel1))
    for row in range(8):
        for col in range(8):
            image_num = row * 8 + col
            fig1.add_subplot(8, 8, image_num + 1)
            plt.imshow((kernel1[image_num][:][:]).T)
            plt.axis('off')
    # fig1.suptitle("Kernels")
    fig1.show()

    # Figuring Layer1 Images
    image_five = np.reshape(mnist.test.images[15], newshape=(1, 28, 28, 1))
    # fig2 = plt.figure()
    plt.imshow(np.reshape(image_five, newshape=(28, 28)))
    # plt.title("5")
    plt.show()
    fig3 = plt.figure(figsize=(40, 40))
    fig3.subplots_adjust(wspace=0, hspace=0)
    conv1_five = tf.nn.conv2d(input=image_five, filter=h1, strides=[1, 1, 1, 1], padding='VALID')
    conv1_five_channel = np.reshape(sess.run(conv1_five).T, newshape=(64, 24, 24))
    for row in range(8):
        for col in range(8):
            image_num = row * 8 + col
            fig3.add_subplot(8, 8, image_num + 1)
            plt.imshow((conv1_five_channel[image_num][:][:]).T)
            plt.axis('off')
    # fig3.suptitle("Conv1")
    fig3.show()

    # Figuring Layer2 Image
    fig4 = plt.figure(figsize=(40, 40))
    fig4.subplots_adjust(wspace=0, hspace=0)
    conv1_five_out = tf.nn.relu(conv1_five + b_h1)
    max_pool_five_1 = tf.nn.max_pool(conv1_five_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv2_five = tf.nn.conv2d(input=max_pool_five_1, filter=h2, strides=[1, 1, 1, 1], padding='VALID')
    conv2_five_channel = np.reshape(sess.run(conv2_five).T, newshape=(64, 8, 8))
    for row in range(8):
        for col in range(8):
            image_num = row * 8 + col
            fig4.add_subplot(8, 8, image_num + 1)
            plt.imshow((conv2_five_channel[image_num][:][:]).T)
            plt.axis('off')
    # fig4.suptitle("Conv2")
    fig4.show()

    # Saving
    saver.save(sess=sess, save_path="Save/")


# Part 4

# y_adapted = tf.placeholder(tf.float32, shape=(None, 2), name="labels_adapted")
y_adapted = tf.placeholder(tf.float32, shape=(None, 10), name="labels_adapted")
# y = tf.placeholder(tf.float32, shape=(None, 10), name="labels")

with tf.name_scope("adapted_fully_connected_layer"):
    # 4 * 4 * 64 = 1024 --> 256
    # Activation: ReLU
    w1_adapted = tf.Variable(initial_value=w1, name='w1_adapted')
    max_pool_2_linear_adapted = tf.reshape(max_pool_2, [-1, 1024])
    b1_adapted = tf.Variable(initial_value=b1, name='b1_adapted')
    z1_adapted = tf.matmul(max_pool_2_linear_adapted, w1_adapted) + b1_adapted
    hidden_layer_out_adapted = tf.nn.relu(z1_adapted)

with tf.name_scope("output_layer_adapted"):
    # 256 --> 2
    # SoftMax
    w2_adapted = tf.Variable(initial_value=tf.random_normal(shape=(256, 10), mean=0, stddev=0.01),
                             name='w2_adapted')
    b2_adapted = tf.Variable(initial_value=tf.zeros(10),
                             name='b1_adapted')
    z2_adapted = tf.matmul(hidden_layer_out_adapted, w2_adapted) + b2_adapted
    output_adapted = z2_adapted

with tf.name_scope("loss_adapted"):
    # Cross Entropy
    reg_coeff1_adapted = 0.01
    reg_coeff2_adapted = 0.01
    reg_loss1_adapted = reg_coeff1_adapted * tf.nn.l2_loss(w1_adapted)
    reg_loss2_adapted = reg_coeff2_adapted * tf.nn.l2_loss(w2_adapted)
    loss_adapted = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=[y_adapted[-1][1], y_adapted[-1][4]],
                                                logits=[output_adapted[-1][1], output_adapted[-1][4]]) +
        reg_loss1_adapted +
        reg_loss2_adapted
    )

with tf.name_scope("optimizer_adapted"):
    # SGD
    # Learning Rate = 0.01
    optimizer_adapted = tf.train.GradientDescentOptimizer(
        learning_rate=0.01).minimize(
        loss_adapted, var_list=[w1_adapted, b1_adapted, w2_adapted, b2_adapted])
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

with tf.name_scope(name="accuracy_adapted"):
    true_label_adapted = tf.argmax(y_adapted, 1)
    # 0 --> 1 & 1 --> 4
    # pred_label_adapted = 3 * tf.argmax(output_adapted, 1) + 1
    pred_label_adapted = tf.argmax(output_adapted, 1)
    is_correct_adapted = tf.equal(true_label_adapted, pred_label_adapted)
    accuracy_train_adapted = tf.reduce_mean(
        tf.cast(is_correct_adapted, dtype=tf.float32),
        name="accuracy_train_adapted")

with tf.name_scope(name="save_adapted"):
    tf.summary.scalar("Loss_Adapted", loss_adapted)
    tf.summary.scalar("Accuracy_Adapted", accuracy_train_adapted)
    merge_adapted = tf.summary.merge_all()
    file_writer_adapted = tf.summary.FileWriter("Monitoring Adapted CNN")

with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    # saver.restore(sess=sess2, save_path='/Save')
    batch_size = 1
    check_batch = 100
    for i in range(10000):
        x_sample, y_sample = mnist.train.next_batch(batch_size=batch_size)
        # y_sample_adapted = [[y_sample[0][1], y_sample[0][4]]]
        # while y_sample_adapted[0][0] == 0 and y_sample_adapted[0][1] == 0:
        #    x_sample, y_sample = mnist.train.next_batch(batch_size=batch_size)
        #    y_sample_adapted = [y_sample[0][1], y_sample[0][4]]
        # x_sample = tf.reshape(x_sample, shape=(batch_size, 28, 28, 1))
        y_sample_adapted = y_sample
        x_sample = np.reshape(x_sample, newshape=(batch_size, 28, 28, 1))
        sess2.run(optimizer_adapted, feed_dict={x: x_sample, y_adapted: y_sample_adapted})
        if i % 2000 == 0:
            x_sample, y_sample = mnist.train.next_batch(batch_size=check_batch)
            y_sample_adapted = y_sample  # [[y_sample[1][:], y_sample[4][:]]]
            # x_sample = tf.reshape(x_sample, shape=(1, 28, 28, 1))
            x_sample = np.reshape(x_sample, newshape=(check_batch, 28, 28, 1))
            # b = sess2.run(merge_adapted, feed_dict={x: x_sample, y: y_sample})
            b = sess2.run(merge, feed_dict={x: x_sample, y: y_sample_adapted})
            file_writer_adapted.add_summary(b, i)
            current_loss, accuracy = sess2.run((loss_adapted, accuracy_train_adapted),
                                               feed_dict={x: x_sample, y_adapted: y_sample_adapted})
            print ("step %5i accuracy is %.5f and cross_entropy(loss) is %.5f" % (i, accuracy, current_loss))
    x_sample, y_sample = mnist.validation.next_batch(batch_size=check_batch)
    y_sample_adapted = y_sample  # [[y_sample[:][1], y_sample[:][4]]]
    # x_sample = tf.reshape(x_sample, shape=(1, 28, 28, 1))
    x_sample = np.reshape(x_sample, newshape=(check_batch, 28, 28, 1))
    accuracy_test = sess2.run(accuracy_train_adapted, feed_dict={x: x_sample, y_adapted: y_sample_adapted})
    print "Test Accuracy: ", accuracy_test
    file_writer_adapted.add_graph(sess2.graph)
