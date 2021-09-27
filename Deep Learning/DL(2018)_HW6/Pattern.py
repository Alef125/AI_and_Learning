import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


def input_seq_generator(k):
    seq = ''
    for ii in range(k):
        seq = seq + 'a'
    seq = seq + 'N'
    for ii in range(k):
        seq = seq + 'b'
    return seq


def seq_to_vec(seq):
    out_vec = []
    my_dict = {'a': [0.0, 0.0, 0.0, 1.0], 'N': [0.0, 0.0, 1.0, 0.0],
               'b': [0.0, 1.0, 0.0, 0.0], 'e': [1.0, 0.0, 0.0, 0.0]}
    for ii in range(len(seq)):
        out_vec.append(my_dict[seq[ii]])
    return out_vec


def pred_to_seq(pred):
    seq = ''
    seq_dict = {0: 'e', 1: 'b', 2: 'N', 3: 'a'}
    for p in pred:
        seq = seq + seq_dict[p]
    return seq


# Preparing input sequences
k_max = 11
training_sequences = [input_seq_generator(i) for i in range(1, k_max)]
training_vec = [list(seq_to_vec(training_sequence)) for training_sequence in training_sequences]
vec_size = 4


# Input and Output
x = tf.placeholder(dtype=tf.float32, shape=(None, None, vec_size))
y = tf.placeholder(dtype=tf.float32, shape=(None, vec_size))


# Introducing LSTM
lstm_size = 10
lstm = rnn.BasicLSTMCell(lstm_size)
lstm_out, lstm_state = tf.nn.dynamic_rnn(cell=lstm, inputs=x, dtype=tf.float32)

# Producing Output
with tf.name_scope('out_vec'):
    out_weights = tf.Variable(tf.random_normal(shape=(lstm_size, vec_size)))
    out_bias = tf.Variable(tf.zeros(shape=vec_size))
    rnn_output = tf.matmul(lstm_out[-1], out_weights) + out_bias

# Loss
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=rnn_output, labels=y))
    optimizer = tf.train.AdamOptimizer()
    trainer = optimizer.minimize(loss)

# Accuracy
with tf.name_scope('Accuracy'):
    similarity = tf.equal(tf.argmax(rnn_output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(similarity, tf.float32))

# Cell States
# LSTM State Tuple: (c:(), h:()) where c is memory state and h is output
# c: [[batch1], [batch2], ... , [batch]n] , [batch] = [cell1, cell2, ... ]
cell0 = tf.summary.scalar('cell0_summary', lstm_state[0][-1][0])
cell1 = tf.summary.scalar('cell1_summary', lstm_state[0][-1][1])
cell2 = tf.summary.scalar('cell2_summary', lstm_state[0][-1][2])
cell3 = tf.summary.scalar('cell3_summary', lstm_state[0][-1][3])
cell4 = tf.summary.scalar('cell4_summary', lstm_state[0][-1][4])
cell5 = tf.summary.scalar('cell5_summary', lstm_state[0][-1][5])
cell6 = tf.summary.scalar('cell6_summary', lstm_state[0][-1][6])
cell7 = tf.summary.scalar('cell7_summary', lstm_state[0][-1][7])
cell8 = tf.summary.scalar('cell8_summary', lstm_state[0][-1][8])
cell9 = tf.summary.scalar('cell9_summary', lstm_state[0][-1][9])
merge = tf.summary.merge_all()

# Training
training_iter = 500
batch_size = 32
with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('Cell_States')

    print "*** TRAINING ***"
    sess.run(tf.global_variables_initializer())
    for i in range(training_iter):
        for j in range(k_max - 1):
            batch_in = np.repeat([training_vec[j]], batch_size, axis=0)
            # batch_out = np.array([np.vstack((b_seq[1:][:], seq_to_vec('e'))) for b_seq in batch_in])
            batch_out = np.vstack((training_vec[j][1:][:], seq_to_vec('e')))
            sess.run(trainer, feed_dict={x: np.array(batch_in), y: batch_out})
            if i % 100 == 0:
                seq_out_pred = pred_to_seq(sess.run(tf.argmax(rnn_output, 1), feed_dict={x: batch_in}))
                seq_out_true = pred_to_seq(sess.run(tf.argmax(y, 1), feed_dict={x: batch_in, y: batch_out}))
                print 'pred: ', seq_out_pred, 'true: ', seq_out_true
                print 'accuracy: ', sess.run(accuracy, feed_dict={x: batch_in, y: batch_out})

# Testing
    k_test = 15
    print "*** TESTING ***"
    for i in range(1, k_test):
        test_sequence = input_seq_generator(i)
        seq_in = test_sequence[0:i + 1]
        out_char = ''
        out_seq = ''
        while out_char != 'e':
            vec_in = np.repeat([seq_to_vec(seq_in)], 1, axis=0)  # to fix input size
            out_vector = sess.run(tf.argmax(rnn_output, 1), feed_dict={x: vec_in})
            out_seq = pred_to_seq(out_vector)
            out_char = out_seq[-1]
            seq_in = seq_in + out_char
        print 'in: ', test_sequence, 'pred: ', out_seq, 'true: ', test_sequence[1:]+'e'
        ind1 = np.argmax(seq_to_vec(out_seq), 1)
        ind2 = np.argmax(seq_to_vec(test_sequence[1:]+'e'), 1)
        size_diff = ind2.size - ind1.size
        for j in range(size_diff):
            ind1 = np.hstack((ind1, [4]))
        equality = np.equal(ind1, ind2)
        print 'same size:', size_diff == 0, 'test accuracy: ', np.mean(equality)

# Getting Cell States Summary for k=15
    test_sequence = input_seq_generator(k_test)
    seq_in = test_sequence[0:k_test + 1]
    out_char = ''
    out_seq = ''
    cnt = 0
    print '*** Saving and Testing for k=15 ***'
    while out_char != 'e':
        vec_in = np.repeat([seq_to_vec(seq_in)], 1, axis=0)  # to fix input size
        out_vector = sess.run(tf.argmax(rnn_output, 1), feed_dict={x: vec_in})
        summary = sess.run(merge, feed_dict={x: vec_in})
        file_writer.add_summary(summary, cnt)
        out_seq = pred_to_seq(out_vector)
        out_char = out_seq[-1]
        seq_in = seq_in + out_char
        cnt = cnt + 1
    print 'in: ', test_sequence, 'pred: ', out_seq, 'true: ', test_sequence[1:] + 'e'
    ind1 = np.argmax(seq_to_vec(out_seq), 1)
    ind2 = np.argmax(seq_to_vec(test_sequence[1:] + 'e'), 1)
    size_diff = ind2.size - ind1.size
    for j in range(size_diff):
        ind1 = np.hstack((ind1, [4]))
    equality = np.equal(ind1, ind2)
    print 'same size:', size_diff == 0, 'test accuracy: ', np.mean(equality)
