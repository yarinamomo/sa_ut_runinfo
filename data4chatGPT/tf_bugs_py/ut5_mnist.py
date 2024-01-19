import os
import sys
import json
import time
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn, rnn_cell
from tensorflow.examples.tutorials.mnist import input_data
assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
n_input = 28  
n_steps = 28  
n_hidden = 128  
n_classes = 10  
def rnn_model(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_steps, 0)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights) + biases

mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)  
params = json.loads(open('parameters.json').read())
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
weights = tf.Variable(tf.random_normal([n_hidden, n_classes]), name='weights')
biases = tf.Variable(tf.random_normal([n_classes]), name='biases')
pred = rnn_model(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables())
    while step * params['batch_size'] < params['training_iters']:
        batch_x, batch_y = mnist.train.next_batch(params['batch_size'])
        print(batch_x)
        print(batch_y)
        batch_x = batch_x.reshape((params['batch_size'], n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})  
        if step % params['display_step'] == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=step)
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print('Iter: {}, Loss: {:.6f}, Accuracy: {:.6f}'.format(step * params['batch_size'], loss, acc))
        step += 1
    print("The training is done")
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
