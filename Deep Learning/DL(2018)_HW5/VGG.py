import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model


model = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet',
                                          input_tensor=None, input_shape=None,
                                          pooling='max', classes=1000)

w1 = model.weights[0]
w2 = model.weights[2]
w3 = model.weights[4]
w4 = model.weights[6]
w5 = model.weights[8]
w6 = model.weights[10]
w7 = model.weights[12]
w8 = model.weights[14]
w9 = model.weights[16]
w10 = model.weights[18]
w11 = model.weights[20]
w12 = model.weights[22]
w13 = model.weights[24]
w14 = model.weights[26]
w15 = model.weights[28]
w16 = model.weights[30]

print model.summary()
plot_model(model, to_file='model_part_2.png')


layer_dict = dict([(layer.name, layer) for layer in model.layers])

layer_names = ['block1_conv1', 'block1_conv2',
               'block2_conv1', 'block2_conv2',
               'block3_conv1', 'block3_conv2', 'block3_conv3',
               'block4_conv1', 'block4_conv2', 'block4_conv3',
               'block5_conv1', 'block5_conv2', 'block5_conv3']

for layer_name in layer_names:

    layer_w = layer_dict[layer_name].get_weights()[0][:, :, 0, :]

    plt.figure(figsize=(20, 20))
    for i in range(64):
        plt.subplot(8, 8, i+1)
        plt.imshow(layer_w[:, :, i])

    plt.savefig('vgg_weights/' + layer_name + '.png')


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     ww = sess.run(w1)
#     im = np.reshape(ww[0], newshape=(24, 24))
#     plt.imshow(im)
#     plt.show()
