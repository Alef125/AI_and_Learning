import tensorflow.compat.v1 as tf


def instance_norm(x):
    # out = tf.contrib.layers.batch_norm(x, decay=0.9,
    # updates_collections=None, epsilon=1e-5, scale=True,
    # scope="batch_norm")
    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(
                                    mean=1.0, stddev=0.02
        ))
        offset = tf.get_variable(
            'offset', [x.get_shape()[-1]],
            initializer=tf.constant_initializer(0.0)
        )
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out


def general_conv2d(input_conv, output_num_channels=64, kernel_size=7, strides=1,
                   stddev=0.02, padding="VALID", name="conv2d",
                   do_norm=True, do_relu=True, relu_factor=0):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(
            input_conv,
            filters=output_num_channels,  # num_outputs
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=stddev
            ),
            bias_initializer=tf.constant_initializer(0.0)
        )

        if do_norm:
            conv_norm = instance_norm(conv)
        else:
            conv_norm = conv

        if do_relu:
            if relu_factor == 0:
                out = tf.nn.relu(conv_norm, "relu")
            else:
                out = tf.nn.leaky_relu(conv_norm, alpha=relu_factor, name="lrelu")
        else:
            out = conv_norm

        return out


def general_deconv2d(input_conv, output_num_channels=64, kernel_size=7, strides=1,
                     stddev=0.02, padding="VALID", name="deconv2d",
                     do_norm=True, do_relu=True, relu_factor=0):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d_transpose(
            input_conv,
            filters=output_num_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            bias_initializer=tf.constant_initializer(0.0)
        )

        if do_norm:
            conv_norm = instance_norm(conv)
        else:
            conv_norm = conv

        if do_relu:
            if relu_factor == 0:
                out = tf.nn.relu(conv_norm, "relu")
            else:
                out = tf.nn.leaky_relu(conv_norm, alpha=relu_factor, name="lrelu")
        else:
            out = conv_norm

        return out
