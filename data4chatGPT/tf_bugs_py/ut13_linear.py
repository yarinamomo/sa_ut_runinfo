import tensorflow as tf
assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
x_data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y_data = [[0], [1], [1], [0]]
X = tf.placeholder(tf.float32, shape=[4, 2], name='x')
Y = tf.placeholder(tf.float32, shape=[4, 1], name='y')
W = tf.Variable(tf.random_uniform([2, 2], -1, 1), name='W')
c = tf.Variable(tf.zeros([2]), name='c')
w = tf.Variable(tf.random_uniform([2, 1], -1, 1), name='w')
b = tf.Variable(tf.zeros([1]), name='b')
H1 = tf.matmul(X, W) + c
Yhat1 = tf.matmul(H1, w) + b
cross_entropy1 = tf.reduce_mean(tf.square(Y - Yhat1))
step1 = tf.train.AdamOptimizer(0.01).minimize(cross_entropy1)
graph1 = tf.initialize_all_variables()
sess1 = tf.Session()
sess1.run(graph1)
for i in range(100):
    _, loss, yhat = sess1.run([step1, cross_entropy1, Yhat1], feed_dict={X: x_data, Y: y_data})
    print("loss %g" % loss)
    print(yhat)
corrects = tf.equal(tf.argmax(Y, 1), tf.argmax(Yhat1, 1))
accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
r = sess1.run(accuracy, feed_dict={X: x_data, Y: y_data})
print('accuracy: ' + str(r * 100) + '%')