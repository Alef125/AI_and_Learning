import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist", one_hot=True)
x = tf.placeholder(tf.float32, shape=(None, 784), name="Input")
y = tf.placeholder(tf.float32, shape=(None, 10), name="Label")

with tf.name_scope(name="Hidden_Layer_1"):
    w1 = tf.Variable(tf.random_uniform(shape=(784, 50), minval=-1, maxval=1), name="w1")
    b1 = tf.Variable(tf.zeros(50), name="b1")
    z1_output = tf.matmul(x, w1) + b1
    hidden_1_output = tf.tanh(z1_output)

with tf.name_scope(name="Hidden_Layer_2"):
    w2 = tf.Variable(tf.random_uniform(shape=(50, 50), minval=-1, maxval=1), name="w2")
    b2 = tf.Variable(tf.zeros(50), name="b2")
    z2_output = tf.matmul(hidden_1_output, w2) + b2
    hidden_2_output = tf.tanh(z2_output)

with tf.name_scope(name="Output_Layer"):
    w3 = tf.Variable(tf.random_uniform(shape=(50, 10), minval=-1, maxval=1), name="w3")
    b3 = tf.Variable(tf.zeros(10), name="b3")
    z3_output = tf.matmul(hidden_2_output, w3) + b3
    output = z3_output

with tf.name_scope(name="Loss"):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))

with tf.name_scope(name="Optimizer"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

with tf.name_scope(name="Accuracy"):
    true_label = tf.argmax(y, 1)
    pred_label = tf.argmax(output, 1)
    is_correct = tf.equal(true_label, pred_label)
    accuracy_train = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32), name="accuracy_train")

with tf.name_scope(name="Save"):
    tf.summary.scalar("Loss", loss)
    tf.summary.scalar("Accuracy", accuracy_train)
    tf.summary.histogram("Wights_Hidden1_layer", w1)
    tf.summary.histogram("Wights_Hidden2_layer", w2)
    tf.summary.histogram("Wights_Hidden3_layer", w3)
    merge = tf.summary.merge_all()
    file_writer = tf.summary.FileWriter("Monitoring9")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for i in range(10000):
        x_batch, y_batch = mnist.train.next_batch(64)
        sess.run(optimizer, feed_dict={x: x_batch, y: y_batch})
        if i % 500 == 0:
            b = sess.run(merge, feed_dict={x: x_batch, y: y_batch})
            file_writer.add_summary(b, i)
            current_loss, accuracy = sess.run((loss, accuracy_train), feed_dict={x: x_batch, y: y_batch})
            print ("step %5i accuracy is %.5f and cross_entropy(loss) is %.5f" % (i, accuracy, current_loss))
    x_batch, y_batch = mnist.validation.next_batch(100)
    accuracy_test = sess.run(accuracy_train, feed_dict={x: x_batch, y: y_batch})
    print "Test Error: ", accuracy_test
    file_writer.add_graph(sess.graph)
