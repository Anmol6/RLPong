import tensorflow as tf
import numpy as np

def init_weight(shape):
    W_init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(W_init)


def init_bias(shapes):
    b_init = tf.constant(0.1, shape=[shapes])
    return tf.Variable(b_init)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1],padding='SAME')


def maxpool(x, pool_win):
    return tf.nn.max_pool(x, ksize=[1, pool_win, pool_win, 1], strides=[1, pool_win, pool_win, 1], padding='SAME')

a, b, c = 80, 80, 4
x = tf.placeholder("float", [None, 80, 80, 4])
s = tf.shape(x)

W1 = init_weight([8, 8, 4, 32])
b1 = init_bias(32)
stride1 = 4
conv1 = tf.nn.relu(conv2d(x, W1, stride1) + b1)
pool1 = maxpool(conv1, 2)

s1 = tf.shape(pool1)

W2 = init_weight([4, 4, 32, 64])
b2 = init_bias(64)
stride2 = 2
conv2 = tf.nn.relu(conv2d(pool1, W2, stride2) + b2)
pool2 = maxpool(conv2, 2)
s2 = tf.shape(pool2)

W3 = init_weight([3, 3, 64, 64])
b3 = init_bias(64)
stride3 = 1
conv3 = tf.nn.relu(conv2d(pool2, W3, stride3)+b3)
#pool3 = maxpool(conv3, 1)
s3 = tf.shape(conv3)

conv3=tf.reshape(conv3, [-1,s[0]])
W4 = init_weight([576, 576])
b4 = init_bias(576)
dense = tf.nn.relu(tf.matmul( W4, conv3))# + b4)

final = init_weight([3, 576])
out = tf.matmul(final, dense)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

X = np.random.rand(1,80,80,4)
#X =tf.random_normal(shape=( 3,80, 80, 4), mean=0.0, stddev=1.0, dtype=tf.float32)
p1=sess.run(out,feed_dict = {x: X})
sss = np.argmax(sess.run(out,feed_dict={x: X}))

print(p1.shape)
print(sss)

#print sess.run(y)
#print sess.run(shape)
