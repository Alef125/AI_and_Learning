# from google.colab import drive
# drive.mount('/content/gdrive')

# cd /content/gdrive/My Drive/Training

# Import libraries
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


# leaky_relu
def lrelu(x_r, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x_r + f2 * tf.abs(x_r)


# G(z)
def generator(xg, yg_label, is_train=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # concat layer
        cat1 = tf.concat([xg, yg_label], 3)
        print("generator", cat1.get_shape())
        # 1st hidden layer
        deconv1 = tf.layers.conv2d_transpose(cat1, 256, [8, 8], strides=(1, 1), padding='valid',
                                             kernel_initializer=w_init, bias_initializer=b_init)
        lrelu1 = lrelu(tf.layers.batch_normalization(deconv1, training=is_train), 0.2)
        print("lrelu1", lrelu1.get_shape())
        # 2nd hidden layer
        deconv2 = tf.layers.conv2d_transpose(lrelu1, 128, [5, 5], strides=(2, 2), padding='same',
                                             kernel_initializer=w_init, bias_initializer=b_init)
        lrelu2 = lrelu(tf.layers.batch_normalization(deconv2, training=is_train), 0.2)
        print("lrelu2", lrelu2.get_shape())
        # output layer
        deconv3 = tf.layers.conv2d_transpose(lrelu2, 1, [5, 5], strides=(2, 2), padding='same',
                                             kernel_initializer=w_init, bias_initializer=b_init)
        o = tf.nn.tanh(deconv3)
        print("o", o.get_shape())
        return o


# D(x)
def discriminator(xd, yd_fill, is_train=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # concat layer
        # cat1 = tf.concat([xd, yd_fill], 3)
        cat1 = xd
        print("discriminator", cat1.get_shape())
        # 1st hidden layer (128X-->64)
        conv1 = tf.layers.conv2d(cat1, 64, [5, 5], strides=(2, 2), padding='same',
                                 kernel_initializer=w_init, bias_initializer=b_init)
        lrelu1 = lrelu(conv1, 0.2)
        print("lrelu1", lrelu1.get_shape())
        # 2nd hidden layer (256X-->128)
        conv2 = tf.layers.conv2d(lrelu1, 128, [5, 5], strides=(2, 2),
                                 padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=is_train), 0.2)
        print("lrelu2", lrelu2.get_shape())
        # output layer --> 3rd hidden layer
        # conv3 = tf.layers.conv2d(lrelu2, 1, [8, 8], strides=(1, 1), padding='valid', kernel_initializer=w_init)
        conv3_1 = tf.layers.dense(tf.reshape(lrelu2, shape=(-1, 8192)), 998,
                                  kernel_initializer=w_init, bias_initializer=b_init)
        conv3_2 = tf.concat([conv3_1, tf.reshape(yd_fill, shape=(-1, 26))], 1)
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3_2, training=is_train), 0.2)
        # 4th hidden layer
        conv4 = tf.layers.dense(lrelu3, 1024, kernel_initializer=w_init, bias_initializer=b_init)
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=is_train), 0.2)
        # 5th hidden layer
        conv5_1 = tf.layers.dense(lrelu4, 1, kernel_initializer=w_init, bias_initializer=b_init)
        conv5 = tf.reshape(conv5_1, shape=(-1, 1, 1, 1))
        o = tf.nn.sigmoid(conv5)
        print("o", o.get_shape())
        return o, conv5


# preprocess
img_size = 32
onehot = np.eye(26)
temp_z_ = np.random.normal(0, 1, (26, 1, 1, 100))
fixed_z_ = temp_z_
fixed_y_ = np.zeros((26, 1))
for i in range(25):
    fixed_z_ = np.concatenate([fixed_z_, temp_z_], 0)
    temp = np.ones((26, 1)) + i
    fixed_y_ = np.concatenate([fixed_y_, temp], 0)

fixed_y_ = onehot[fixed_y_.astype(np.int32)].reshape((26 * 26, 1, 1, 26))


def show_result(num_epoch, show=False, save=False, path='result.png'):
    test_images = sess.run(G_z, {z: fixed_z_, y_label: fixed_y_, isTrain: False})

    size_figure_grid = 26
    fig, ax = plt.subplots(10, size_figure_grid, figsize=(32, 32))
    for ii, j in itertools.product(range(10), range(size_figure_grid)):
        ax[ii, j].get_xaxis().set_visible(False)
        ax[ii, j].get_yaxis().set_visible(False)

    for k in range(10*26):
        ii = k // 26
        jj = k % 26
        ax[ii, jj].cla()
        ax[ii, jj].imshow(np.reshape(test_images[k], (img_size, img_size)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


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


# training parameters
batch_size = 100
# lr = 0.0002
train_epoch = 30
global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(0.0002, global_step, 500, 0.95, staircase=True)

pickle_in = open("train_data.pickle", "rb")
train_data = pickle.load(pickle_in)
x_train, x_validation, y_train, y_validation = train_test_split(train_data['data'], train_data['labels'], test_size=0.2)

train_X = (x_train.reshape(-1, 32, 32, 1))
test_X = (x_validation.reshape(-1, 32, 32, 1))

train_X = (train_X - np.mean(train_X))/(np.std(train_X))
test_X = (test_X - np.mean(train_X))/(np.std(train_X))

# variables : input
x = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
y_label = tf.placeholder(tf.float32, shape=(None, 1, 1, 26))
# y_fill = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 26))
y_fill = tf.placeholder(tf.float32, shape=(None, 1, 1, 26))
# y_fill = tf.placeholder(tf.float32, shape=(None, 26))
isTrain = tf.placeholder(dtype=tf.bool)


# networks : generator
G_z = generator(z, y_label, isTrain, reuse=tf.AUTO_REUSE)

# networks : discriminator
D_real, D_real_logits = discriminator(x, y_fill, isTrain, reuse=tf.AUTO_REUSE)
D_fake, D_fake_logits = discriminator(G_z, y_fill, isTrain, reuse=True)

# loss for each network
# D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits,
#                                                                      labels=tf.ones([batch_size, 1])))
# D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,
#                                                                      labels=tf.zeros([batch_size, 1])))
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits,
                                                                     labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,
                                                                     labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
# G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,
#                                                                 labels=tf.ones([batch_size, 1])))
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,
                                                                labels=tf.ones([batch_size, 1, 1, 1])))

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optim = tf.train.AdamOptimizer(lr, beta1=0.5)
    D_optim = optim.minimize(D_loss, global_step=global_step, var_list=D_vars)
    # D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

train_set = train_X
train_label = y_train

# results save folder
root = 'MNIST_cDCGAN_results/'
model = 'MNIST_cDCGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')


train_hist = dict()
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []


shuffle_idxs = random.sample(range(0, train_set.shape[0]), train_set.shape[0])
shuffled_set = train_set[shuffle_idxs]
shuffled_label = train_label[shuffle_idxs]
x_ = shuffled_set[0*batch_size:(0+1)*batch_size]
itter = 0
y_label_ = shuffled_label[itter*batch_size:(itter+1)*batch_size].reshape([batch_size, 1, 1, 26])
# y_label_ = shuffled_label[itter*batch_size:(itter+1)*batch_size].reshape([batch_size, 26])
print(y_label.shape)
# y_fill_ = y_label_ * np.ones([batch_size, img_size, img_size, 26])
# y_fill_ = y_label_
y_fill_ = np.reshape(y_label_, newshape=[-1, 1, 1, 26])
# print(y_fill.shape)


# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    shuffle_idxs = random.sample(range(0, train_set.shape[0]), train_set.shape[0])
    shuffled_set = train_set[shuffle_idxs]
    shuffled_label = train_label[shuffle_idxs]
    for itter in range(shuffled_set.shape[0] // (100 * batch_size)):
        # update discriminator
        x_ = shuffled_set[itter*batch_size:(itter+1)*batch_size]
        y_label_ = shuffled_label[itter*batch_size:(itter+1)*batch_size].reshape([batch_size, 1, 1, 26])
        # y_label_ = shuffled_label[itter*batch_size:(itter+1)*batch_size].reshape([batch_size, 26])
        # y_fill_ = y_label_ * np.ones([batch_size, img_size, img_size, 26])
        # y_fill_ = y_label_
        y_fill_ = np.reshape(y_label_, newshape=[-1, 1, 1, 26])
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, y_fill: y_fill_, y_label: y_label_, isTrain: True})
        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        y_ = np.random.randint(0, 25, (batch_size, 1))
        y_label_ = onehot[y_.astype(np.int32)].reshape([batch_size, 1, 1, 26])
        # y_label_ = onehot[y_.astype(np.int32)].reshape([batch_size, 26])
        # y_fill_ = y_label_ * np.ones([batch_size, img_size, img_size, 26])
        # y_fill_ = y_label_
        y_fill_ = np.reshape(y_label_, newshape=[-1, 1, 1, 26])
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, y_fill: y_fill_, y_label: y_label_, isTrain: True})

        errD_fake = D_loss_fake.eval({z: z_, y_label: y_label_, y_fill: y_fill_, isTrain: False})
        errD_real = D_loss_real.eval({x: x_, y_label: y_label_, y_fill: y_fill_, isTrain: False})
        errG = G_loss.eval({z: z_, y_label: y_label_, y_fill: y_fill_, isTrain: False})

        D_losses.append(errD_fake + errD_real)
        G_losses.append(errG)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime,
                                                                np.mean(D_losses), np.mean(G_losses)))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    show_result((epoch + 1), save=True, path=fixed_p)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)