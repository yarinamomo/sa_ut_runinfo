from tensorflow.contrib.keras.api.keras.preprocessing import image
import tensorflow as tf

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)

img_path = 'sample.jpg'

import numpy as np

x = image.load_img(img_path, target_size=(250, 250))

x = image.img_to_array(x)
x_expended = np.expand_dims(x, axis=0)
x_expended_trans = np.transpose(x_expended, [0, 3, 1, 2])

# runtime info
x = np.random.normal(size=(250, 250, 3))
x_expended = np.random.normal(size=(1, 250, 250, 3))
x_expended_trans = np.random.normal(size=(1, 3, 250, 250))

X = tf.placeholder(tf.float32, [None, 250, 250, 3])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(X, feed_dict={X: x_expended_trans}))
