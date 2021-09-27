import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("mnist", one_hot=True)
#x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="input_image")
#y = tf.placeholder(tf.float32, shape=(None, 10), name="labels")

#with tf.name_scope("loss"):
    # Cross Entropy
#    reg_coeff1 = 0.01
#    reg_coeff2 = 0.01
#    reg_loss1 = reg_coeff1 * tf.nn.l2_loss(w1)
#    reg_loss2 = reg_coeff2 * tf.nn.l2_loss(w2)
#    loss = tf.reduce_mean(
#        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output) +
#        reg_loss1 +
#        reg_loss2
#    )

#with tf.name_scope("optimizer"):
    # SGD
    # Learning Rate = 0.01
 #   optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss, var_list=[w1, b1, w2, b2])
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

#with tf.name_scope(name="accuracy"):
#    true_label = tf.argmax(y, 1)
#    pred_label = tf.argmax(output, 1)
#    is_correct = tf.equal(true_label, pred_label)
#    accuracy_train = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32), name="accuracy_train")


with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path='/Save')
    x_sample, y_sample = mnist.train.next_batch(batch_size=1)
    print(sess.run(loss, feed_dict={x: x_sample, y: y_sample}))
