import sys
import tensorflow as tf
import numpy
assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
numpy.random.seed(20180130)
rng = numpy.random
learning_rate = 0.01
training_epochs = 2000
display_step = 50
f = open("data.csv")
data = numpy.loadtxt(f, delimiter=",")
train_X = data[:60000, :-1]
train_Y = data[:60000, -1]
test_X = data[60000:80000, :-1]
test_Y = data[60000:80000, -1]
X_val = data[80000:, :-1]
y_val = data[80000:, -1]
n_input = train_X.shape[1]
n_samples = train_X.shape[0]
print(n_input)
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None])
W = tf.Variable(tf.zeros([6, 1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")
activation = tf.add(tf.matmul(X, W), b)
cost = tf.reduce_sum(tf.pow(activation - Y, 2)) / (2 * n_samples)  
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x[numpy.newaxis, ...], Y: y[numpy.newaxis, ...]})
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=",
                  "{:.9f}".format(
                      sess.run(cost, feed_dict={X: train_X, Y: train_Y})),
                  "W=", sess.run(W), "b=", sess.run(b))
    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    print("Testing... (L2 loss Comparison)")
    testing_cost = sess.run(tf.reduce_sum(tf.pow(activation - Y, 2)) / (2 * test_X.shape[0]),
                            feed_dict={X: test_X,
                                       Y: test_Y})  
    print("Testing cost=", testing_cost)
    print("Absolute l2 loss difference:", abs(training_cost - testing_cost))