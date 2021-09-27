import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # This loads 60'000 Training samples (28*28 + Label),
#                                                           and 10'000 test samples

# Discriminator : 784 --> 128 --> 1 Fully Connected
X = tf.placeholder(tf.float32, shape=[None, 784], name='Image')

d_w1 = tf.Variable(tf.random_normal(shape=[784, 128], mean=0, stddev=0.01), name='d_w1')
d_b1 = tf.Variable(tf.zeros(shape=[128]), name='d_b1')

d_w2 = tf.Variable(tf.random_normal(shape=[128, 1], mean=0, stddev=0.01), name='d_w2')
d_b2 = tf.Variable(tf.zeros(shape=[1]), name='d_b2')

d_variables = [d_w1, d_w2, d_b1, d_b2]


def discriminator(image):
    d_z1 = tf.matmul(image, d_w1) + d_b1
    d_h1 = tf.nn.leaky_relu(d_z1)  # Why Relu ---> Leaky Relu
    d_z2 = tf.matmul(d_h1, d_w2) + d_b2
    d_prob = tf.nn.sigmoid(d_z2)
    return d_prob


# Generator: 100 --> 128 --> 784
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

g_w1 = tf.Variable(tf.random_normal(shape=[100, 128], mean=0, stddev=0.01), name='g_w1')
g_b1 = tf.Variable(tf.zeros(shape=[128]), name='g_b1')

g_w2 = tf.Variable(tf.random_normal(shape=[128, 784], mean=0, stddev=0.01), name='g_w2')
g_b2 = tf.Variable(tf.zeros(shape=[784]), name='g_b2')

g_w3 = tf.Variable(tf.random_normal(shape=[784, 784], mean=0, stddev=0.01), name='g_w3')
g_b3 = tf.Variable(tf.zeros(shape=[784]), name='g_b3')

g_variables = [g_w1, g_w2, g_w3, g_b1, g_b2, g_b3]


def generator(z):
    g_z1 = tf.matmul(z, g_w1) + g_b1
    g_h1 = tf.nn.leaky_relu(g_z1)  # Why Relu --> Leaky Relu
    g_z2 = tf.matmul(g_h1, g_w2) + g_b2
    g_h2 = tf.nn.leaky_relu(g_z2)
    g_z3 = tf.matmul(g_h2, g_w3) + g_b3
    g_prob = tf.nn.tanh(g_z3)
    return g_prob


# Loss Function
G_sample = generator(Z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

D_loss = -tf.reduce_mean(tf.log(0.01 + D_real) + tf.log(1 - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))  # Why log(1 - D_fake) --> -log(D_fake) ?

d_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=d_variables)  # Why Adam --> learning_rate = ?
g_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=g_variables)  # Why Adam --> learning_rate = ?


def sample_z(m, n):
    z = np.random.uniform(low=-1.0, high=1.0, size=[m, n])  # Why Uniform ?
    return z


iters = 100000
batch_size = 128
z_dim = 100
x_train_net = np.float32(np.reshape(x_train, newshape=[60000, 784])) / 255
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training
for it in range(iters):
    x_batch = x_train_net[np.random.choice(60000, batch_size), :]
    # zz = sample_z(batch_size, z_dim)
    _, d_loss_curr = sess.run([d_solver, D_loss], feed_dict={X: x_batch, Z: sample_z(batch_size, z_dim)})
    _, g_loss_curr = sess.run([g_solver, G_loss], feed_dict={Z: sample_z(batch_size, z_dim)})
    #while g_loss_curr > 2:
          #z = sample_z(batch_size, z_dim)
    #      _, g_loss_curr = sess.run([g_solver, G_loss], feed_dict={Z: sample_z(batch_size, z_dim)})
    #     d_loss_curr = sess.run(D_loss, feed_dict={X: x_batch, Z: sample_z(batch_size, z_dim)})
    if it % 100 == 0:
        # print sess.run(D_real, feed_dict={X: x_batch})
        # print sess.run(D_loss, feed_dict={X: x_batch, Z: zz})
        print 100 * it / iters, d_loss_curr, g_loss_curr
        # plt.imshow(np.reshape(sess.run(G_sample, feed_dict={Z: sample_z(1, z_dim)}), newshape=[28, 28]))
        # plt.show()
        if it % 1000 == 0:
            ggg = sess.run(G_sample, feed_dict={Z: sample_z(1, z_dim)})
            plt.imshow(np.reshape(ggg, newshape=[28, 28]))
            plt.show()
            print ggg
            print x_train_net[10]

# Test
for i in range(10):
    new_z = sample_z(1, z_dim)
    new_image = sess.run(G_sample, feed_dict={Z: new_z})
    plt.imshow(np.reshape(new_image, newshape=[28, 28]))
