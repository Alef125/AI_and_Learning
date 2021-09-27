import tensorflow as tf
import numpy as np
import os
import time

tf.enable_eager_execution()


path_to_file = 'Book.txt'
text = open(path_to_file).read()

# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))


# Take a look at the first 250 characters in text
print(text[10:245])


# The unique characters in the file
vocab = sorted(set(text))

# Creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# The maximum length sentence we want for a single input in characters
seq_length = 40
examples_per_epoch = len(text) // seq_length

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])


sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)


for input_example, target_example in dataset.take(1):
    print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

    for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
        print("Step {:4d}".format(i))
        print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
        print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))


# Batch size
BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch // BATCH_SIZE

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 128


if tf.test.is_gpu_available():
    rnn = tf.keras.layers.CuDNNGRU
else:
    import functools
    rnn = functools.partial(
        tf.keras.layers.GRU, recurrent_activation='sigmoid')


def build_model(vocab_size_f, embedding_dim_f, rnn_units_f, batch_size_f):
    model_f = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size_f, embedding_dim_f,
                                  batch_input_shape=[batch_size_f, None]),
        rnn(rnn_units_f,
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            stateful=True),
        tf.keras.layers.Dense(vocab_size_f)
    ])
    return model_f


model = build_model(
    vocab_size_f=len(vocab),
    embedding_dim_f=embedding_dim,
    rnn_units_f=rnn_units,
    batch_size_f=BATCH_SIZE)

model.summary()


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits)  # , from_logits=True)


learning_rate = 0.01
model.compile(
    optimizer=tf.train.RMSPropOptimizer(learning_rate),
    loss=loss)


# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


EPOCHS = 20

history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
                    callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size_f=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

model.summary()


def generate_text(model_g, start_string):

    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 400

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model_g.reset_states()
    for i in range(num_generate):
        predictions_g = model_g(input_eval)
        # remove the batch dimension
        predictions_g = tf.squeeze(predictions_g, 0)

        # using a multinomial distribution to predict the word returned by the model
        predictions_g = predictions_g / temperature
        predicted_id = tf.multinomial(predictions_g, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)


# Experiment by changing the start string
print(generate_text(model, start_string="ROMEO: "))


model = build_model(
    vocab_size_f=len(vocab),
    embedding_dim_f=embedding_dim,
    rnn_units_f=rnn_units,
    batch_size_f=BATCH_SIZE)


optimizer = tf.train.AdamOptimizer()


# Training step
EPOCHS = 1

for epoch in range(EPOCHS):
    start = time.time()

    # initializing the hidden state at the start of every epoch
    # initally hidden is None
    hidden = model.reset_states()

    for (batch_n, (inp, target)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            # feeding the hidden state back into the model
            # This is the interesting step
            predictions = model(inp)
            loss = tf.losses.sparse_softmax_cross_entropy(target, predictions)

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))

        if batch_n % 100 == 0:
            template = 'Epoch {} Batch {} Loss {:.4f}'
            print(template.format(epoch + 1, batch_n, loss))

    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))

    print ('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    model.save_weights(checkpoint_prefix.format(epoch=epoch))
