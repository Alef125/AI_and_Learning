"""Contains losses used for performing image-to-image domain adaptation."""
import tensorflow.compat.v1 as tf


# def convert_labels_to_numerical(labels, all_classes):
#     numerical_labels = []
#     for a_label in labels:
#         numerical_labels.append(all_classes.index(a_label))
#     return numerical_labels


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


def make_true_labels_tensor(true_labels, output_shape):
    # true_real_labels = np.zeros(shape=output_shape)
    batch_size = output_shape[0]  # Also = len(true_labels)
    # numerical_labels = convert_labels_to_numerical(true_labels, all_classes=classes)
    true_labels_list = []
    for i in range(batch_size):
        # true_real_labels[i, :, :, numerical_labels[i]] = 1
        true_labels_list.append(make_labels_tensor(true_labels[i], num_classes=output_shape[3],
                                                   h=output_shape[1], w=output_shape[2]))
        # true_real_labels[i, :, :, true_labels[i]] = 1
    return tf.stack(true_labels_list)


def cycle_consistency_loss(real_images, generated_images):
    """Compute the cycle consistency loss.
    The cycle consistency loss is defined as the sum of the L1 distances
    between the real images from each domain and their generated (fake)
    counterparts.
    Args:
        real_images: A batch of images from domain X, a `Tensor` of shape
            [batch_size, height, width, channels].
        generated_images: A batch of generated images made to look like they
            came from domain X, a `Tensor` of shape
            [batch_size, height, width, channels].
    Returns:
        The cycle consistency loss.
    """
    return tf.reduce_mean(tf.abs(real_images - generated_images))


def gan_loss_generator(prob_fake_in_classes, true_labels):
    """Computes the LS-GAN loss as minimized by the generator.
    Rather than compute the negative loglikelihood, a least-squares loss is
    used to optimize the discriminators.
    Args:
        prob_fake_in_classes: The discriminator's estimate that generated images
            made to look like real images are real.
        true_labels: same size with batch_size, and shows true labels
    Returns:
        The total LS-GAN loss.
    """
    true_real_labels = make_true_labels_tensor(true_labels=true_labels,
                                               output_shape=prob_fake_in_classes.shape)
    return tf.reduce_mean(tf.squared_difference(prob_fake_in_classes, true_real_labels))


def gan_loss_discriminator(prob_real_in_classes, prob_fake_in_classes, true_labels):
    """Computes the LS-GAN loss as minimized by the discriminator.
    Rather than compute the negative loglikelihood, a least-squares loss is
    used to optimize the discriminators.
    Args:
        prob_real_in_classes: The discriminator's estimate that images actually
            drawn from the real domain are in fact real in (correct class).
        prob_fake_in_classes: The discriminator's estimate that generated images
            made to look like real images are real (in correct class).
        true_labels: same size with batch_size, and shows true labels
    Returns:
        The total LS-GAN loss.

    *** This is for real_i + fake_i = 1 implementation; in which for fake images independent from it's class,
    we consider prob vector to be whole 0 ***
    """
    true_real_labels = make_true_labels_tensor(true_labels=true_labels,
                                               output_shape=prob_real_in_classes.shape)
    return (tf.reduce_mean(tf.squared_difference(prob_real_in_classes, true_real_labels)) +
            tf.reduce_mean(tf.squared_difference(prob_fake_in_classes, 0))) * 0.5
