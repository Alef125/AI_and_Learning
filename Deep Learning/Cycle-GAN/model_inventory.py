"""Code for constructing the model and get the outputs from the model."""
import tensorflow.compat.v1 as tf
import layers

IMG_CHANNELS = 3
ngf = 32
ndf = 64


def get_outputs(inputs, skip=False):
    real_images_a = inputs['images_a']
    real_images_b = inputs['images_b']
    # Old fake images
    fake_samples_a = inputs['fake_samples_a']
    fake_samples_b = inputs['fake_samples_b']

    with tf.variable_scope("Model"):
        prob_real_a_is_real = discriminator(real_images_a, "d_A_real")
        prob_real_b_is_real = discriminator(real_images_b, "d_B_real")
        fake_images_b = generator(real_images_a, name="g_A", skip=skip)
        fake_images_a = generator(real_images_b, name="g_B", skip=skip)

        prob_fake_a_is_real = discriminator(fake_images_a, "d_A_fake")
        prob_fake_b_is_real = discriminator(fake_images_b, "d_B_fake")
        cycle_images_a = generator(fake_images_b, "g_B_fake", skip=skip)
        cycle_images_b = generator(fake_images_a, "g_A_fake", skip=skip)

        prob_fake_samples_a_is_real = discriminator(fake_samples_a, "d_A_fakeSamples")
        prob_fake_samples_b_is_real = discriminator(fake_samples_b, "d_B_fakeSamples")

    return {
        'prob_real_a_is_real': prob_real_a_is_real,
        'prob_real_b_is_real': prob_real_b_is_real,
        'prob_fake_a_is_real': prob_fake_a_is_real,
        'prob_fake_b_is_real': prob_fake_b_is_real,
        'prob_fake_samples_a_is_real': prob_fake_samples_a_is_real,
        'prob_fake_samples_b_is_real': prob_fake_samples_b_is_real,
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


def generator(input_gen, name="generator", skip=False):
    with tf.variable_scope(name):
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
                                     output_num_channels=IMG_CHANNELS,
                                     kernel_size=7,
                                     strides=1,
                                     stddev=0.02, padding="SAME", name="c6", do_norm=False, do_relu=False)
        if skip:
            out_gen = tf.nn.tanh(input_gen + o_c6, "t1")
        else:
            out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen


def discriminator(input_disc, name="discriminator"):
    with tf.variable_scope(name):
        # patch_input = tf.random_crop(input_disc, [1, 70, 70, 3])
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
                                     output_num_channels=1,
                                     kernel_size=4,
                                     strides=1,
                                     stddev=0.02, padding="SAME", name="c5", do_norm=False, do_relu=False)
        return o_c5
