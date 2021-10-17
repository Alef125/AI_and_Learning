"""Code for constructing the model and get the outputs from the model."""
import tensorflow.compat.v1 as tf
import layers

ngf = 32
ndf = 64


def make_labels_tensor(label, num_classes, h, w):
    """ This Function is used just because of impossibility of eager implementation in tensorflow 1 :)) """
    labels_list = []
    for j in range(num_classes):
        if j == label:
            label_tensor = tf.ones(shape=[h, w], dtype=tf.float32)
        else:
            label_tensor = tf.zeros(shape=[h, w], dtype=tf.float32)
        labels_list.append(label_tensor)
    labels_tensor = tf.stack(labels_list, axis=2)  # [:, :, label]
    return labels_tensor


def get_label_embedded_input(input_batch, input_label, output_label, num_classes):
    """
    This Function, Embeds input label and output label with input
    Input number of channels will be input.channels + 2 * num_class
    (one for input label and one for output label)
    """
    input_batch_size, input_height, input_width = input_batch.shape[0], input_batch.shape[1], input_batch.shape[2]
    # input_label & output_label are lists in length [batch_size]

    inputs_labels_list = []
    outputs_labels_list = []
    for i in range(input_batch_size):  # todo: using batch size cause inability to use None shape
        inputs_labels_list.append(make_labels_tensor(input_label[i], num_classes, h=input_height, w=input_width))
        outputs_labels_list.append(make_labels_tensor(output_label[i], num_classes, h=input_height, w=input_width))

    input_label_tensor = tf.stack(inputs_labels_list)
    output_label_tensor = tf.stack(outputs_labels_list)
    label_embedded_input = tf.concat(values=[input_batch, input_label_tensor, output_label_tensor],
                                     axis=3)
    return label_embedded_input


# def convert_labels_to_numerical(labels, all_classes):
#     numerical_labels = []
#     for a_label in labels:
#         numerical_labels.append(all_classes.index(a_label))
#     return numerical_labels


def get_outputs(inputs, num_classes, image_channels=3, skip=False):
    real_images_a = inputs['images_a']
    # labels_a = inputs['labels_a']
    # numerical_labels_a = convert_labels_to_numerical(labels_a, classes)
    numerical_labels_a = inputs['labels_a']

    real_images_b = inputs['images_b']
    # labels_b = inputs['labels_b']
    # numerical_labels_b = convert_labels_to_numerical(labels_b, classes)
    numerical_labels_b = inputs['labels_b']

    # Old fake images
    fake_samples_a = inputs['fake_samples_a']
    fake_samples_b = inputs['fake_samples_b']

    with tf.variable_scope("Model"):  # ToDo: 1)discriminators have num_classes channels 2)totally weight sharing
        prob_real_a_in_classes = discriminator(real_images_a, num_classes=num_classes, name="discriminator")
        prob_real_b_in_classes = discriminator(real_images_b, num_classes=num_classes, name="discriminator", reuse=True)

        fake_images_b = generator(get_label_embedded_input(real_images_a,
                                                           input_label=numerical_labels_a,
                                                           output_label=numerical_labels_b,
                                                           num_classes=num_classes),
                                  image_channels=image_channels, name="generator", skip=skip)
        fake_images_a = generator(get_label_embedded_input(real_images_b,
                                                           input_label=numerical_labels_a,
                                                           output_label=numerical_labels_b,
                                                           num_classes=num_classes),
                                  image_channels=image_channels, name="generator", skip=skip, reuse=True)

        prob_fake_a_in_classes = discriminator(fake_images_a, num_classes=num_classes, name="discriminator", reuse=True)
        prob_fake_b_in_classes = discriminator(fake_images_b, num_classes=num_classes, name="discriminator", reuse=True)

        cycle_images_a = generator(get_label_embedded_input(fake_images_b,
                                                            input_label=numerical_labels_a,
                                                            output_label=numerical_labels_b,
                                                            num_classes=num_classes),
                                   image_channels=image_channels, name="generator", skip=skip, reuse=True)
        cycle_images_b = generator(get_label_embedded_input(fake_images_b,
                                                            input_label=numerical_labels_a,
                                                            output_label=numerical_labels_b,
                                                            num_classes=num_classes),
                                   image_channels=image_channels, name="generator", skip=skip, reuse=True)

        prob_fake_samples_a_in_classes = discriminator(fake_samples_a, num_classes=num_classes,
                                                       name="discriminator", reuse=True)
        prob_fake_samples_b_in_classes = discriminator(fake_samples_b, num_classes=num_classes,
                                                       name="discriminator", reuse=True)

    return {
        'prob_real_a_in_classes': prob_real_a_in_classes,
        'prob_real_b_in_classes': prob_real_b_in_classes,
        'prob_fake_a_in_classes': prob_fake_a_in_classes,
        'prob_fake_b_in_classes': prob_fake_b_in_classes,
        'prob_fake_samples_a_in_classes': prob_fake_samples_a_in_classes,
        'prob_fake_samples_b_in_classes': prob_fake_samples_b_in_classes,
        'cycle_images_a': cycle_images_a,
        'cycle_images_b': cycle_images_b,
        'fake_images_a': fake_images_a,
        'fake_images_b': fake_images_b,
    }


def resnet_block(input_res, dim, name="resnet", padding="REFLECT"):
    """build a single block of resnet.
    :param input_res: input_res
    :param dim: dim
    :param name: name
    :param padding: for tensorflow version use REFLECT; for pytorch version use
     CONSTANT
    :return: a single block of resnet.
    """
    with tf.variable_scope(name):
        out_res = tf.pad(input_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(out_res,
                                        output_num_channels=dim,
                                        kernel_size=3,
                                        strides=1,
                                        stddev=0.02, padding="VALID", name="c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(out_res,
                                        output_num_channels=dim,
                                        kernel_size=3,
                                        strides=1,
                                        stddev=0.02, padding="VALID", name="c2", do_relu=False)
        return tf.nn.relu(out_res + input_res)


def generator(input_gen, image_channels=3, name="generator", skip=False, reuse=False):
    """
    :param input_gen: input image
    :param image_channels: usually 3
    :param name: name of generator
    :param skip: to contain skip connection or not
    :param reuse: variables in scope reused or not
    :return: generated image
    """
    with tf.variable_scope(name, reuse=reuse):
        ks = 3
        padding = "REFLECT"
        pad_input = tf.pad(input_gen, [[0, 0], [ks, ks], [ks, ks], [0, 0]], padding)
        o_c1 = layers.general_conv2d(pad_input,
                                     output_num_channels=ngf,
                                     kernel_size=7,
                                     strides=1,
                                     stddev=0.02, name="c1")
        o_c2 = layers.general_conv2d(o_c1,
                                     output_num_channels=ngf * 2,
                                     kernel_size=ks,
                                     strides=2,
                                     stddev=0.02, padding="SAME", name="c2")
        o_c3 = layers.general_conv2d(o_c2,
                                     output_num_channels=ngf * 4,
                                     kernel_size=ks,
                                     strides=2,
                                     stddev=0.02, padding="SAME", name="c3")
        o_r1 = resnet_block(o_c3, dim=ngf * 4, name="r1", padding=padding)
        o_r2 = resnet_block(o_r1, dim=ngf * 4, name="r2", padding=padding)
        o_r3 = resnet_block(o_r2, dim=ngf * 4, name="r3", padding=padding)
        o_r4 = resnet_block(o_r3, dim=ngf * 4, name="r4", padding=padding)
        o_r5 = resnet_block(o_r4, dim=ngf * 4, name="r5", padding=padding)
        o_r6 = resnet_block(o_r5, dim=ngf * 4, name="r6", padding=padding)
        o_r7 = resnet_block(o_r6, dim=ngf * 4, name="r7", padding=padding)
        o_r8 = resnet_block(o_r7, dim=ngf * 4, name="r8", padding=padding)
        o_r9 = resnet_block(o_r8, dim=ngf * 4, name="r9", padding=padding)

        o_c4 = layers.general_deconv2d(o_r9,  # out_shape = [BATCH_SIZE, 128, 128, ngf * 2]
                                       output_num_channels=ngf * 2,
                                       kernel_size=ks,
                                       strides=2,
                                       stddev=0.02, padding="SAME", name="c4")
        o_c5 = layers.general_deconv2d(o_c4,  # out_shape = [BATCH_SIZE, 256, 256, ngf]
                                       output_num_channels=ngf,
                                       kernel_size=ks,
                                       strides=2,
                                       stddev=0.02, padding="SAME", name="c5")
        o_c6 = layers.general_conv2d(o_c5,
                                     output_num_channels=image_channels,
                                     kernel_size=7,
                                     strides=1,
                                     stddev=0.02, padding="SAME", name="c6", do_norm=False, do_relu=False)
        if skip:
            real_image = input_gen[:, :, :, 0:image_channels]
            out_gen = tf.nn.tanh(real_image + o_c6, "t1")
        else:
            out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen


def discriminator(input_disc, num_classes, name="discriminator", reuse=False):
    with tf.variable_scope(name, reuse=reuse):  # reuse is set True manually
        o_c1 = layers.general_conv2d(input_disc,
                                     output_num_channels=ndf,
                                     kernel_size=4,
                                     strides=2,
                                     stddev=0.02, padding="SAME", name="c1", do_norm=False, relu_factor=0.2)
        o_c2 = layers.general_conv2d(o_c1,
                                     output_num_channels=ndf * 2,
                                     kernel_size=4,
                                     strides=2,
                                     stddev=0.02, padding="SAME", name="c2", relu_factor=0.2)
        o_c3 = layers.general_conv2d(o_c2,
                                     output_num_channels=ndf * 4,
                                     kernel_size=4,
                                     strides=2,
                                     stddev=0.02, padding="SAME", name="c3", relu_factor=0.2)
        o_c4 = layers.general_conv2d(o_c3,
                                     output_num_channels=ndf * 8,
                                     kernel_size=4,
                                     strides=1,
                                     stddev=0.02, padding="SAME", name="c4", relu_factor=0.2)
        o_c5 = layers.general_conv2d(o_c4,
                                     output_num_channels=num_classes,
                                     kernel_size=4,
                                     strides=1,
                                     stddev=0.02, padding="SAME", name="c5", do_norm=False, do_relu=False)
        return o_c5
