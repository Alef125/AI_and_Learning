import tensorflow.compat.v1 as tf


def _load_samples(csv_path, image_type, image_channels):
    filename_queue = tf.train.string_input_producer(string_tensor=[csv_path])
    reader = tf.TextLineReader()
    _, csv_filename = reader.read(filename_queue)
    record_defaults = [tf.constant([], dtype=tf.string),
                       tf.constant([], dtype=tf.string)]
    filename_i, filename_j = tf.decode_csv(csv_filename,
                                           record_defaults=record_defaults)

    file_content_i = tf.read_file(filename_i)
    file_content_j = tf.read_file(filename_j)

    if image_type == '.jpg':
        images_decoded_i = tf.image.decode_jpeg(file_content_i, channels=image_channels)
        images_decoded_j = tf.image.decode_jpeg(file_content_j, channels=image_channels)
    elif image_type == '.png':
        images_decoded_i = tf.image.decode_png(file_content_i, channels=image_channels, dtype=tf.uint8)
        images_decoded_j = tf.image.decode_png(file_content_j, channels=image_channels, dtype=tf.uint8)

    else:
        raise ValueError('type %s is invalid.' % image_type)

    return images_decoded_i, images_decoded_j


def load_data(dataset_name, image_shape_to_resize, image_shape_after_crop, batch_size=1,
              do_shuffle=True, do_flipping=False):
    """
    :param dataset_name: name of dataset
    :param image_shape_to_resize: Resize to this size before random cropping.
    :param image_shape_after_crop: Random crop to this size, usually [height, width, 3]
    :param batch_size: Size of batch
    :param do_shuffle: Shuffle switch.
    :param do_flipping: Flip switch.
    :return:
    """
    image_channels = image_shape_after_crop[2]  # basically is 3
    base_images_i, base_images_j = _load_samples(csv_path=cyclegan_datasets_info.get_info(dataset_name,
                                                                                          info="PATH_TO_CSV"),
                                                 image_type=clegan_datasets_info.get_info(dataset_name,
                                                                                          info="TYPE"),
                                                 image_channels=image_channels)
    # Preprocessing:
    resized_images_i = tf.image.resize_images(base_images_i, size=image_shape_to_resize)
    resized_images_j = tf.image.resize_images(base_images_j, size=image_shape_to_resize)

    if do_flipping:
        after_flip_images_i = tf.image.random_flip_left_right(resized_images_i)
        after_flip_images_j = tf.image.random_flip_left_right(resized_images_j)

    else:
        after_flip_images_i = resized_images_i
        after_flip_images_j = resized_images_j

    cropped_images_i = tf.random_crop(after_flip_images_i, size=image_shape_after_crop)
    cropped_images_j = tf.random_crop(after_flip_images_j, size=image_shape_after_crop)

    normalized_images_i = tf.subtract(tf.div(cropped_images_i, 127.5), 1)
    normalized_images_j = tf.subtract(tf.div(cropped_images_j, 127.5), 1)

    # Batch
    if do_shuffle:
        images_i, images_j = tf.train.shuffle_batch([normalized_images_i, normalized_images_j],
                                                    batch_size=batch_size,
                                                    capacity=5000,
                                                    min_after_dequeue=100)
    else:
        images_i, images_j = tf.train.batch([normalized_images_i, normalized_images_j], batch_size=batch_size)

    inputs = {
        'images_i': images_i,
        'images_j': images_j
    }

    return inputs
