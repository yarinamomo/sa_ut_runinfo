import os
import time
import numpy as np
import tensorflow as tf
import random
assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
np.random.seed(20180130)
random.seed(20180130)
batch_size = 50
n_input = 56 * 56 * 3
n_classes = 10
def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
x = tf.placeholder(tf.float32, [None, 56, 56, 3])
y = tf.placeholder(tf.float32, [None, n_classes])
x_image = tf.reshape(x, [-1, 56, 56, 3])
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
W_fc1 = weight_variable([14 * 14 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 14 * 14 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
W_conv1 = tf.Variable(tf.zeros([5, 5, 3, 32]))
W_conv2 = tf.Variable(tf.zeros([5, 5, 32, 64]))
W_fc1 = tf.Variable(tf.zeros([12544, 1024]))
W_fc2 = tf.Variable(tf.zeros([1024, 10]))
b_conv1 = tf.Variable(tf.zeros([32,]))
b_conv2 = tf.Variable(tf.zeros([64,]))
b_fc1 = tf.Variable(tf.zeros([1024,]))
b_fc2 = tf.Variable(tf.zeros([10,]))
batch_size = 50
h_conv1 = tf.placeholder(tf.float32, [None, 56, 56, 32])
h_conv2 = tf.placeholder(tf.float32, [None, 28, 28, 64])
h_fc1 = tf.placeholder(tf.float32, [None, 1024])
h_fc1_drop = tf.placeholder(tf.float32, [None, 1024])
h_pool1 = tf.placeholder(tf.float32, [None, 28, 28, 32])
h_pool2 = tf.placeholder(tf.float32, [None, 14, 14, 64])
h_pool2_flat = tf.placeholder(tf.float32, [None, 12544])
n_classes = 10
n_input = 9408
x = tf.placeholder(tf.float32, [None, 56, 56, 3])
x_image = tf.placeholder(tf.float32, [None, 56, 56, 3])
y = tf.placeholder(tf.float32, [None, 10])
y_conv = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
W_conv1 = tf.Variable(tf.zeros([5, 5, 3, 32]))
W_conv2 = tf.Variable(tf.zeros([5, 5, 32, 64]))
W_fc1 = tf.Variable(tf.zeros([12544, 1024]))
W_fc2 = tf.Variable(tf.zeros([1024, 10]))
b_conv1 = tf.Variable(tf.zeros([32,]))
b_conv2 = tf.Variable(tf.zeros([64,]))
b_fc1 = tf.Variable(tf.zeros([1024,]))
b_fc2 = tf.Variable(tf.zeros([10,]))
batch_size = 50
correct_prediction = tf.placeholder(tf.bool, [None,])
h_conv1 = tf.placeholder(tf.float32, [None, 56, 56, 32])
h_conv2 = tf.placeholder(tf.float32, [None, 28, 28, 64])
h_fc1 = tf.placeholder(tf.float32, [None, 1024])
h_fc1_drop = tf.placeholder(tf.float32, [None, 1024])
h_pool1 = tf.placeholder(tf.float32, [None, 28, 28, 32])
h_pool2 = tf.placeholder(tf.float32, [None, 14, 14, 64])
h_pool2_flat = tf.placeholder(tf.float32, [None, 12544])
n_classes = 10
n_input = 9408
x = tf.placeholder(tf.float32, [None, 56, 56, 3])
x_image = tf.placeholder(tf.float32, [None, 56, 56, 3])
y = tf.placeholder(tf.float32, [None, 10])
y_conv = tf.placeholder(tf.float32, [None, 10])
def run():
    def generate_unit_test(length):
        return [np.random.normal(0, 0.1, [56, 56, 3]) for _ in range(length)], [random.randint(0, 9) for _ in
                                                                                range(length)]
    test_images, test_labels = generate_unit_test(10)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(20000):
            image_batch, label_batch = generate_unit_test(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: image_batch, y: label_batch, keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
                train_step.run(feed_dict={x: image_batch, y: label_batch, keep_prob: 0.5})
        print("test accuracy %g" % accuracy.eval(feed_dict={
            x: test_images, y: test_labels, keep_prob: 1.0}))
        coord.request_stop()
        coord.join(threads)
if __name__ == '__main__':
    run()