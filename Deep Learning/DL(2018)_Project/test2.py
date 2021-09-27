import Selector
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ################################################################################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import glob
import os
import time
import itertools
import imageio
import pickle
import random
# import imageio as imageio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from tqdm import tqdm
from sklearn.model_selection import train_test_split


def png2np(im_path):
    images = []
    for im_path in glob.glob(im_path):
        img = imageio.imread(im_path)
        # img = np.reshape(im_filtered, newshape=784)
        images.append(img)
    images = np.array(images)
    return images


Char2idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12,
            'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24,
            'Z': 25}

idx2char = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
            13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
            25: 'Z'}

# pickle_in = open("train_data.pickle", "rb")
# train_data = pickle.load(pickle_in)
# x_train, x_validation, y_train, y_validation = train_test_split(train_data['data'], train_data['labels'], test_size=0.2)

# train_X = (x_train.reshape(-1, 32, 32, 1))
# test_X = (x_validation.reshape(-1, 32, 32, 1))

# train_X = (train_X - np.mean(train_X))/(np.std(train_X))
# test_X = (test_X - np.mean(train_X))/(np.std(train_X))


y_label = tf.placeholder(tf.float32, shape=[None, 26], name='Y_Label')

# Discriminator : 784 --> 128 --> 1 Fully Connected
X = tf.placeholder(tf.float32, shape=[None, 1024], name='Image')

d_w1 = tf.Variable(tf.random_normal(shape=[1024 + 26, 128], mean=0, stddev=0.01), name='d_w1')
d_b1 = tf.Variable(tf.zeros(shape=[128]), name='d_b1')

d_w2 = tf.Variable(tf.random_normal(shape=[128, 1], mean=0, stddev=0.01), name='d_w2')
d_b2 = tf.Variable(tf.zeros(shape=[1]), name='d_b2')

d_variables = [d_w1, d_w2, d_b1, d_b2]


def discriminator(raw_image, y_label):
    image = tf.concat([raw_image, y_label], 1)
    d_z1 = tf.matmul(image, d_w1) + d_b1
    d_h1 = tf.nn.relu(d_z1)  # Why Relu ---> Leaky Relu
    d_z2 = tf.matmul(d_h1, d_w2) + d_b2
    d_prob = tf.nn.sigmoid(d_z2)
    return d_prob


# Generator: 100 --> 128 --> 784
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

g_w1 = tf.Variable(tf.random_normal(shape=[100 + 26, 128], mean=0, stddev=0.01), name='g_w1')
g_b1 = tf.Variable(tf.zeros(shape=[128]), name='g_b1')

g_w2 = tf.Variable(tf.random_normal(shape=[128, 784], mean=0, stddev=0.01), name='g_w2')
g_b2 = tf.Variable(tf.zeros(shape=[784]), name='g_b2')

g_w3 = tf.Variable(tf.random_normal(shape=[784, 1024], mean=0, stddev=0.01), name='g_w3')
g_b3 = tf.Variable(tf.zeros(shape=[1024]), name='g_b3')

g_variables = [g_w1, g_w2, g_w3, g_b1, g_b2, g_b3]


def generator(raw_z, y_label):
    z = tf.concat([raw_z, y_label], 1)
    g_z1 = tf.matmul(z, g_w1) + g_b1
    g_h1 = tf.nn.relu(g_z1)  # Why Relu --> Leaky Relu
    g_z2 = tf.matmul(g_h1, g_w2) + g_b2
    g_h2 = tf.nn.relu(g_z2)
    g_z3 = tf.matmul(g_h2, g_w3) + g_b3
    g_prob = tf.nn.tanh(g_z3)
    return g_prob


# Loss Function
G_sample = generator(Z, y_label)
D_real = discriminator(X, y_label)
D_fake = discriminator(G_sample, y_label)

batch_size = 260

# D_loss = -tf.reduce_mean(tf.log(0.01 + D_real) + tf.log(1 - D_fake))
# D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real , labels=tf.ones(shape=D_real.shape, dtype=tf.float32)) +
#                         tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake , labels=tf.zeros(shape=D_fake.shape, dtype=tf.float32)))
# G_loss = -tf.reduce_mean(tf.log(D_fake))  # Why log(1 - D_fake) --> -log(D_fake) ?


D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real,
                                                                     labels=tf.ones([batch_size, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake,
                                                                     labels=tf.zeros([batch_size, 1])))
D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake,
                                                                labels=tf.ones([batch_size, 1])))

d_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=d_variables)  # Why Adam --> learning_rate = ?
g_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=g_variables)  # Why Adam --> learning_rate = ?

img_size = 32


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x_s = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x_s, y1, label='D_loss')
    plt.plot(x_s, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def sample_z(m, n):
    z = np.random.uniform(low=-1.0, high=1.0, size=[m, n])  # Why Uniform ?
    return z


z_dim = 100

# ################################################################################


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver()

word_image = []

with tf.Session(config=config) as sess:
    # Pre Definitions
    sess.run(tf.global_variables_initializer())
    saver.restore(sess=sess, save_path='./Gan Model/model.ckpt')
    # final_labels = sess.run(pred_label, feed_dict={x: test_images})
    # final_letters = [which_letter(curr_label) for curr_label in final_labels]
    for i in range(26):
        curr_label = np.float32(np.zeros(shape=[10, 26]))
        curr_label[:][i] = np.float32(np.ones([26]))
        new_image = sess.run(G_sample, feed_dict={Z: sample_z(10, 100), y_label: curr_label})
        word_image.append(new_image)

word_images = np.array([])  # 26 * 10 * 32 * 32
words = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
         'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
         'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
best_images = [Selector.selector(word_images[i][:][:][:], words[i]) for i in range(26)]  # list of 32 * 32 images

sentence = 'THE QUICK BROWN FOX JUMPS OVER A LAZY DOG'
sentence_length = 41
generated_image = np.ones(shape=(32, 1312))
for i in range(sentence_length):
    word = sentence[i]
    if word != ' ':
        word_index = ord(word) - ord('A')
        word_image = best_images[word_index]
        generated_image[:][i * 32: i * 32 + 32] = word_image[:][:]

plt.imshow(generated_image)
plt.show()
my_image = Image.fromarray(generated_image)
my_image.save('Sentence.png')
