import tensorflow.compat.v1 as tf
import tf_slim as slim
import json
import data_loader
import model
import losses
import os
from datetime import datetime
from keras_preprocessing.image import save_img as imsave
import numpy as np
import random
tf.disable_v2_behavior()


def get_current_learning_rate(current_step, max_step, base_learning_rate):
    if current_step < max_step / 2:
        current_lr = base_learning_rate
    else:
        current_lr = base_learning_rate * 2 * (1 - current_step / max_step)
    return current_lr


def predict_classes(class_probabilities):
    num_images = len(class_probabilities)
    labels = []
    for i in range(num_images):
        local_labels_array = np.argmax(class_probabilities[i][:, :, :], axis=3).ravel()
        label = np.bincount(local_labels_array).argmax()
        labels.append(label)
    return labels


def get_generation_desires(base_images, base_labels, num_classes):
    """ This Function, Replicates base_images and base_labels by num_classes-1 to assign other
     num_classes-1 labels as output labels"""
    num_images = len(base_labels)  # Also = base_images.shape[0]
    replicated_images = np.repeat(base_images, num_classes-1, axis=0)
    replicated_labels = np.repeat(base_labels, num_classes-1)
    desired_labels = []
    for i in range(num_images):
        the_labels = list(range(num_classes))
        the_labels.remove(base_labels[i])
        desired_labels.extend(the_labels)
    replicated_labels_array = np.array(replicated_labels).reshape((num_images * (num_classes-1), 1))
    desired_labels_array = np.array(desired_labels).reshape((num_images * (num_classes-1), 1))
    return replicated_images, replicated_labels_array, desired_labels_array


class CycleGAN:
    def __init__(self, dataset_name, image_resize_shape, image_shape, batch_size, do_flipping, skip_connection,
                 steps_in_epoch, num_images_to_save, base_learning_rate, lambda_a, lambda_b, adam_beta1,
                 fake_pool_size, log_dir, checkpoint_dir, dataset_size, dataset_csv_path, dataset_images_type,
                 data_classes, fake_replacement_prob, test_images_csv_path, test_images_saving_path):
        self._dataset_name = dataset_name
        self._shape_before_crop = image_resize_shape
        self._IMG_HEIGHT = image_shape[0]
        self._IMG_WIDTH = image_shape[1]
        self._IMG_CHANNELS = image_shape[2]
        self._BATCH_SIZE = batch_size
        self._INPUT_IMG_SHAPE = [self._BATCH_SIZE, self._IMG_HEIGHT, self._IMG_WIDTH, self._IMG_CHANNELS]
        self._LABELS_SHAPE = [self._BATCH_SIZE, 1]
        self._FAKE_SAMPLES_SHAPE = [None, self._IMG_HEIGHT, self._IMG_WIDTH, self._IMG_CHANNELS]
        self._fake_pool_size = fake_pool_size
        self._POOL_SHAPE = [self._fake_pool_size, 1, self._IMG_HEIGHT, self._IMG_WIDTH, self._IMG_CHANNELS]
        self._do_flipping = do_flipping

        self._input_data = None
        self._network_input_a = tf.placeholder(tf.float32, self._INPUT_IMG_SHAPE, name="input_A")
        self._network_input_b = tf.placeholder(tf.float32, self._INPUT_IMG_SHAPE, name="input_B")
        self._input_class_a = tf.placeholder(tf.int32, self._LABELS_SHAPE, name="class_A")  # used to be string
        self._input_class_b = tf.placeholder(tf.int32, self._LABELS_SHAPE, name="class_B")  # used to be string
        self._fake_samples_a = tf.placeholder(tf.float32, self._FAKE_SAMPLES_SHAPE, name="fake_samples_a")
        self._fake_samples_b = tf.placeholder(tf.float32, self._FAKE_SAMPLES_SHAPE, name="fake_samples_b")
        self._network_outputs = None
        self._num_generated_fake_inputs = 0
        self._fake_pools = {the_class: np.zeros(self._POOL_SHAPE) for the_class in data_classes}
        self._fake_replacement_prob = fake_replacement_prob

        self._skip_connection = skip_connection

        self._steps_in_epoch = steps_in_epoch
        self._BASE_LR = base_learning_rate
        self._network_learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        self._global_step = slim.get_or_create_global_step()
        self._num_images_to_save = num_images_to_save
        self._LAMBDA_A = lambda_a  # ratio for cycle a consistency loss
        self._LAMBDA_B = lambda_b  # ratio for cycle b consistency loss
        self._adam_beta1 = adam_beta1
        self._optimizers = None
        self._losses_summary = None

        current_time = datetime.now().strftime("%Y:%m:%d-%H.%M.%S")
        self._output_log_dir = os.path.join(log_dir, current_time)
        self._saved_images_dir = os.path.join(self._output_log_dir, 'images')
        self._checkpoint_dir = checkpoint_dir
        self._dataset_size = dataset_size
        self._dataset_csv_path = dataset_csv_path
        self._dataset_images_type = dataset_images_type
        self._data_classes = dict(zip(range(len(data_classes)), data_classes))

        self._test_images_csv_path = test_images_csv_path
        self._test_images_saving_path = test_images_saving_path
        self._test_images = None

    def load_input_data(self):
        self._input_data = data_loader.load_train_data(dataset_name=self._dataset_name,
                                                       dataset_csv_path=self._dataset_csv_path,
                                                       images_type=self._dataset_images_type,
                                                       image_shape_to_resize=self._shape_before_crop,
                                                       image_shape_after_crop=[self._IMG_HEIGHT,
                                                                               self._IMG_WIDTH,
                                                                               self._IMG_CHANNELS],
                                                       batch_size=self._BATCH_SIZE,
                                                       do_shuffle=False,
                                                       do_flipping=self._do_flipping)

    def load_test_data(self):
        self._test_images = data_loader.load_test_data(dataset_csv_path=self._test_images_csv_path,
                                                       images_type=self._dataset_images_type,
                                                       image_shape_to_resize=self._shape_before_crop,
                                                       image_shape_after_crop=[self._IMG_HEIGHT,
                                                                               self._IMG_WIDTH,
                                                                               self._IMG_CHANNELS])

    def setup_model(self):
        inputs = {
            'images_a': self._network_input_a,
            'labels_a': self._input_class_a,
            'images_b': self._network_input_b,
            'labels_b': self._input_class_b,
            'fake_samples_a': self._fake_samples_a,
            'fake_samples_b': self._fake_samples_b,
        }
        self._network_outputs = model.get_outputs(inputs,
                                                  num_classes=len(self._data_classes.keys()),
                                                  image_channels=self._IMG_CHANNELS,
                                                  skip=self._skip_connection)

    def compute_losses(self):
        """
        In this function we are defining the variables for loss calculations
        and training model.
        d_loss_A/d_loss_B -> loss for discriminator A/B
        g_loss_A/g_loss_B -> loss for generator A/B
        *_optimizers -> Various trainer for above loss functions
        *_losses_summary -> Summary variables for above loss functions
        """
        cycle_consistency_loss_a = \
            self._LAMBDA_A * losses.cycle_consistency_loss(real_images=self._network_input_a,
                                                           generated_images=self._network_outputs['cycle_images_a'])
        cycle_consistency_loss_b = \
            self._LAMBDA_B * losses.cycle_consistency_loss(real_images=self._network_input_b,
                                                           generated_images=self._network_outputs['cycle_images_b'])
        gan_loss_a = losses.gan_loss_generator(self._network_outputs['prob_fake_a_in_classes'],
                                               true_labels=self._input_class_a)
        gan_loss_b = losses.gan_loss_generator(self._network_outputs['prob_fake_b_in_classes'],
                                               true_labels=self._input_class_b)
        g_loss_a = cycle_consistency_loss_a + cycle_consistency_loss_b + gan_loss_b
        g_loss_b = cycle_consistency_loss_b + cycle_consistency_loss_a + gan_loss_a
        d_loss_a = losses.gan_loss_discriminator(
            prob_real_in_classes=self._network_outputs['prob_real_a_in_classes'],
            prob_fake_in_classes=self._network_outputs['prob_fake_samples_a_in_classes'],
            true_labels=self._input_class_a)
        d_loss_b = losses.gan_loss_discriminator(
            prob_real_in_classes=self._network_outputs['prob_real_b_in_classes'],
            prob_fake_in_classes=self._network_outputs['prob_fake_samples_b_in_classes'],
            true_labels=self._input_class_b)
        optimizer = tf.train.AdamOptimizer(self._network_learning_rate, beta1=self._adam_beta1)
        model_vars = tf.trainable_variables()
        for var in model_vars:
            print(var.name)
        d_vars = [var for var in model_vars if 'discriminator' in var.name]
        g_vars = [var for var in model_vars if 'generator' in var.name]
        self._optimizers = {
            'd_A': optimizer.minimize(d_loss_a, var_list=d_vars),
            'd_B': optimizer.minimize(d_loss_b, var_list=d_vars),
            'g_A': optimizer.minimize(g_loss_a, var_list=g_vars),
            'g_B': optimizer.minimize(g_loss_b, var_list=g_vars)
        }
        # Summary variables for tensorboard
        g_a_loss_summary = tf.summary.scalar("g_A_loss", g_loss_a)
        g_b_loss_summary = tf.summary.scalar("g_B_loss", g_loss_b)
        d_a_loss_summary = tf.summary.scalar("d_A_loss", d_loss_a)
        d_b_loss_summary = tf.summary.scalar("d_B_loss", d_loss_b)
        self._losses_summary = {
            'g_A': g_a_loss_summary,
            'g_B': g_b_loss_summary,
            'd_A': d_a_loss_summary,
            'd_B': d_b_loss_summary
        }

    def get_fake_sample_from_pool(self, new_fakes, labels):
        """
        This function saves the generated image to corresponding
        pool of images.
        It keeps on feeling the pool till it is full and then randomly
        selects an already stored image and replace it with new one.
        """
        number_of_fakes = len(new_fakes)  # Also = len(labels)
        fake_samples = []
        for i in range(number_of_fakes):
            the_class = self._data_classes[int(labels[i])]  # labels[i]
            fake_pool = self._fake_pools[the_class]
            new_fake = new_fakes[i]
            if self._num_generated_fake_inputs < self._fake_pool_size:
                fake_pool[self._num_generated_fake_inputs] = new_fake
                fake_samples.append(new_fake)
            else:
                p = random.random()
                if p > self._fake_replacement_prob:
                    random_id = random.randint(0, self._fake_pool_size - 1)
                    temp = fake_pool[random_id][0]
                    fake_pool[random_id] = new_fake
                    fake_samples.append(temp)
                else:
                    fake_samples.append(new_fake)
        return np.array(fake_samples)

    def save_images(self, sess, epoch):  # TODO
        if not os.path.exists(self._saved_images_dir):
            os.makedirs(self._saved_images_dir)
        with open(os.path.join(self._output_log_dir, 'epoch_' + str(epoch) + '.html'), 'w') as images_html:
            for i in range(0, self._num_images_to_save):
                print("Saving image {}/{}".format(i, self._num_images_to_save))
                inputs = sess.run(self._input_data)  # inputs = self.input_data = [images_i, images_j]
                fake_a_temp, fake_b_temp, cyc_a_temp, cyc_b_temp = sess.run([
                    self._network_outputs['fake_images_a'],
                    self._network_outputs['fake_images_b'],
                    self._network_outputs['cycle_images_a'],
                    self._network_outputs['cycle_images_b']
                ], feed_dict={
                    self._network_input_a: inputs['images_i'],
                    self._input_class_a: inputs['labels_i'],
                    self._network_input_b: inputs['images_j'],
                    self._input_class_b: inputs['labels_j']
                })
                names = ['inputA_', 'inputB_', 'fakeA_', 'fakeB_', 'cycA_', 'cycB_']
                tensors = [inputs['images_i'], inputs['images_j'],
                           fake_b_temp, fake_a_temp, cyc_a_temp, cyc_b_temp]
                for name, tensor in zip(names, tensors):
                    image_name = name + str(epoch) + "_" + str(i) + ".jpg"
                    sample_picture = ((tensor[0] + 1) * 127.5).astype(np.uint8)  # Denormalized
                    imsave(os.path.join(self._saved_images_dir, image_name),
                           sample_picture)
                    images_html.write("<img src=\"" + os.path.join('images', image_name) + "\">")
                images_html.write("<br>")

    def train(self, to_restore):
        self.load_input_data()
        self.setup_model()
        self.compute_losses()
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        saver = tf.train.Saver()
        max_images = self._dataset_size

        with tf.Session() as sess:
            sess.run(init)
            if to_restore:
                checkpoint_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
                saver.restore(sess, checkpoint_fname)

            writer = tf.summary.FileWriter(self._output_log_dir)
            if not os.path.exists(self._output_log_dir):
                os.makedirs(self._output_log_dir)
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coordinator)

            # ------------ Training Loop ------------ :
            for epoch in range(sess.run(self._global_step), self._steps_in_epoch):
                print("In the epoch ", epoch)
                saver.save(sess, os.path.join(self._output_log_dir, "cyclegan"), global_step=epoch)
                # Dealing with the learning rate as per the epoch number
                current_lr = get_current_learning_rate(epoch, self._steps_in_epoch, self._BASE_LR)
                self.save_images(sess, epoch)
                for i in range(0, max_images):
                    print("Processing batch {}/{}".format(i, max_images))
                    inputs = sess.run(self._input_data)
                    # ----- Optimizing the G_A network -----
                    _, new_fake_b, summary_str = sess.run([self._optimizers['g_A'],
                                                           self._network_outputs['fake_images_b'],
                                                           self._losses_summary['g_A']],
                                                          feed_dict={self._network_input_a: inputs['images_i'],
                                                                     self._input_class_a: inputs['labels_i'],
                                                                     self._network_input_b: inputs['images_j'],
                                                                     self._input_class_b: inputs['labels_j'],
                                                                     self._network_learning_rate: current_lr})
                    writer.add_summary(summary_str, epoch * max_images + i)
                    # ----- Optimizing the D_B network -----
                    fake_b_sample = self.get_fake_sample_from_pool(new_fakes=new_fake_b,
                                                                   labels=inputs['labels_j'])
                    _, summary_str = sess.run(
                        [self._optimizers['d_B'], self._losses_summary['d_B']],
                        feed_dict={
                            self._network_input_a: inputs['images_i'],
                            self._input_class_a: inputs['labels_i'],
                            self._network_input_b: inputs['images_j'],
                            self._input_class_b: inputs['labels_j'],
                            self._network_learning_rate: current_lr,
                            self._fake_samples_b: fake_b_sample
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)
                    # ----- Optimizing the G_B network -----
                    _, new_fake_a, summary_str = sess.run([self._optimizers['g_B'],
                                                           self._network_outputs['fake_images_a'],
                                                           self._losses_summary['g_B']],
                                                          feed_dict={self._network_input_a: inputs['images_i'],
                                                                     self._input_class_a: inputs['labels_i'],
                                                                     self._network_input_b: inputs['images_j'],
                                                                     self._input_class_b: inputs['labels_j'],
                                                                     self._network_learning_rate: current_lr})
                    writer.add_summary(summary_str, epoch * max_images + i)
                    # ----- Optimizing the D_A network ------
                    fake_a_sample = self.get_fake_sample_from_pool(new_fakes=new_fake_a,
                                                                   labels=inputs['labels_i'])
                    _, summary_str = sess.run(
                        [self._optimizers['d_A'], self._losses_summary['d_A']],
                        feed_dict={
                            self._network_input_a: inputs['images_i'],
                            self._input_class_a: inputs['labels_i'],
                            self._network_input_b: inputs['images_j'],
                            self._input_class_b: inputs['labels_j'],
                            self._network_learning_rate: current_lr,
                            self._fake_samples_a: fake_a_sample
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)
                    # -------------------------------------------------------
                    writer.flush()
                    self._num_generated_fake_inputs += 1
                sess.run(tf.assign(self._global_step, epoch + 1))
            coordinator.request_stop()
            coordinator.join(threads)
            writer.add_graph(sess.graph)

    def generate_other_classes_for_images(self, images, labels, sess):
        generated_images = {the_class: [] for the_class in self._data_classes.values()}
        num_base_images = len(labels)  # Also = images.shape[0]
        num_classes = len(self._data_classes.keys())
        extended_base_images, extended_base_labels, required_labels = get_generation_desires(base_images=images,
                                                                                             base_labels=labels,
                                                                                             num_classes=num_classes)
        """ Commented Parts are due to present inability to use None as batch size """
        # fake_temps = sess.run([
        #     self._network_outputs['fake_images_b']],
        #     feed_dict={self._network_input_a: extended_base_images,
        #                self._input_class_a: extended_base_labels,
        #                self._input_class_b: required_labels})

        for j in range(num_base_images):
            the_label = self._data_classes[labels[j]]
            generated_images[the_label].append(images[j])  # [j, :, :, :]
            # for k in range(num_classes - 1):
            #     the_label = self._data_classes[required_labels[j*(num_classes-1)+k, 0]]
            #     the_image = fake_temps[j*(num_classes-1)+k]
            #     generated_images[the_label].append(the_image[0])
        for k in range(num_classes - 1):
            fake_temps = sess.run([
                self._network_outputs['fake_images_b']],
                feed_dict={self._network_input_a: extended_base_images[k*num_base_images:(k+1)*num_base_images],
                           self._input_class_a: extended_base_labels[k*num_base_images:(k+1)*num_base_images],
                           self._input_class_b: required_labels[k*num_base_images:(k+1)*num_base_images]})
            for j in range(num_base_images):
                the_label = self._data_classes[required_labels[k*num_base_images + j, 0]]
                the_image = fake_temps[j]  # [j] used to be [j*(num_classes-1) + k]
                generated_images[the_label].append(the_image[0])
        return generated_images

    def save_test_images(self, generated_dict, num_test_images, j):  # todo: j is i due to same inability
        if not os.path.exists(self._test_images_saving_path):
            os.makedirs(self._test_images_saving_path)
        for i in range(num_test_images):
            for the_class in self._data_classes.values():
                class_folder = os.path.join(self._test_images_saving_path, the_class)
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)
                image_name = str(j) + "_" + the_class + self._dataset_images_type
                the_picture = ((generated_dict[the_class][i] + 1) * 127.5).astype(np.uint8)  # Denormalized
                imsave(os.path.join(class_folder, image_name),
                       the_picture)

    def test(self):
        """Test Function: take pictures from one class (unknown) and make other expressions for it"""
        print("Testing the results")
        self.load_test_data()
        self.setup_model()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            checkpoint_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
            saver.restore(sess, checkpoint_fname)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            test_images = sess.run(self._test_images)
            for i in range(0, 4):  # todo: this is due to inability to use None as batch_size
                the_test_image = test_images[i].reshape(self._INPUT_IMG_SHAPE)
                classes_probabilities = sess.run([self._network_outputs['prob_real_a_in_classes']],
                                                 feed_dict={self._network_input_a: the_test_image})
                predicted_classes = predict_classes(classes_probabilities)
                generated_dict = self.generate_other_classes_for_images(images=the_test_image,
                                                                        labels=predicted_classes,
                                                                        sess=sess)
                self.save_test_images(generated_dict, num_test_images=the_test_image.shape[0], j=i)
            # ######################
            coord.request_stop()
            coord.join(threads)
        print("Done")


configs_filename = "./config/configs.json"
with open(configs_filename) as configs_file:
    configs = json.load(configs_file)
cycle_GAN_model = CycleGAN(dataset_name=configs['dataset_name'],
                           image_resize_shape=[configs['image_resize_height'], configs['image_resize_width']],
                           image_shape=[configs['image_height'], configs['image_width'], configs['image_channels']],
                           batch_size=configs['batch_size'],
                           do_flipping=configs['do_flipping'],
                           skip_connection=configs['skip_connection'],
                           steps_in_epoch=configs['steps_in_epoch'],
                           num_images_to_save=configs['num_images_to_save'],
                           base_learning_rate=configs['base_learning_rate'],
                           lambda_a=configs['lambda_a'],
                           lambda_b=configs['lambda_b'],
                           adam_beta1=configs['adam_beta1'],
                           fake_pool_size=configs['fake_pool_size'],
                           log_dir=configs['log_dir'],
                           checkpoint_dir=configs['checkpoint_dir'],
                           dataset_size=configs['dataset_size'],
                           dataset_csv_path=configs['path_to_train_csv'],
                           dataset_images_type=configs['image_type'],
                           data_classes=configs['classes'],
                           fake_replacement_prob=configs['fake_replacement_prob'],
                           test_images_csv_path=configs['path_to_test_csv'],
                           test_images_saving_path=configs['test_images_saving_path'])
# cycle_GAN_model.train(to_restore=False)
cycle_GAN_model.test()
