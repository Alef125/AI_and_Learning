"""Contains losses used for performing image-to-image domain adaptation."""
import tensorflow.compat.v1 as tf


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


def gan_loss_generator(prob_fake_is_real):
    """Computes the LS-GAN loss as minimized by the generator.
    Rather than compute the negative loglikelihood, a least-squares loss is
    used to optimize the discriminators.
    Args:
        prob_fake_is_real: The discriminator's estimate that generated images
            made to look like real images are real.
    Returns:
        The total LS-GAN loss.
    """
    return tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 1))


def gan_loss_discriminator(prob_real_is_real, prob_fake_is_real):
    """Computes the LS-GAN loss as minimized by the discriminator.
    Rather than compute the negative loglikelihood, a least-squares loss is
    used to optimize the discriminators.
    Args:
        prob_real_is_real: The discriminator's estimate that images actually
            drawn from the real domain are in fact real.
        prob_fake_is_real: The discriminator's estimate that generated images
            made to look like real images are real.
    Returns:
        The total LS-GAN loss.
    """
    return (tf.reduce_mean(tf.squared_difference(prob_real_is_real, 1)) +
            tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 0))) * 0.5
