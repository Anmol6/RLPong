import pong as RLpong
import tensorflow as tf
import numpy as np
from math import *
import cv2
import sys
from collections import deque

NUM_ACTIONS = 3


def init_weight(shape):
    W_init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(W_init)


def init_bias(shape):
    b_init = tf.constant(0.1, shape)
    return tf.Variable(b_init)


def conv2d(X, w, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1])


def maxpool(x, pool_win):
    return tf.nn.max_pool(x, ksize=[1, pool_win, pool_win, 1], strides=[1, pool_win, pool_win, 1], padding='SAME')


def convnet():
    K = 4

    x = tf.placeholder(tf.float32, shape=[None, 80, 80, K])
    # y = tf.placeholder("float", shape=[NUM_ACTIONS])

    W1 = init_weight([8, 8, 4, 32])
    b1 = init_bias([32])
    stride1 = 4
    conv1 = tf.nn.relu(conv2d(x, W1, stride1) + b1)
    pool1 = maxpool(conv1, 2)

    W2 = init_weight([4, 4, 32, 64])
    b2 = init_bias([64])
    stride2 = 2
    conv2 = tf.nn.relu(conv2d(pool1, W2, stride2) + b2)
    pool2 = maxpool(conv2, 2)

    W3 = init_weight([3, 3, 64, 64])
    b3 = init_bias([64])
    stride3 = 1
    conv3 = tf.nn.relu(conv2d(pool2, W3, stride3)+b3)
    pool3 = maxpool(conv3, 1)

    tf.reshape(pool3, [-1])

    W4 = init_weight([256, 256])
    b4 = init_bias([256])
    dense = tf.nn.relu(tf.matmul(W4, pool3, transpose_b=True) + b4)

    final = init_weight([NUM_ACTIONS, 256])
    out = tf.nn.soft.matmul(final, dense)

    return out, dense, x


def train(Q_theta, final_h, Qinput, sess):
    C = 0
    K = 4  # number of image frames in a sequence
    N = 100000  # Size of replay memory(number of past frame sequences stored)
	# decay rate of epsilon, to encourage exploitation over exploration with time
    e_decay = 1000000
    gamma = 0.99
    time_observation = 500000
    e_init = 1.0  # greedy probability
    e_final = 0.05

    mini_batch_size = 30
    D = deque(maxlen=N)  # replay memory
    game = RLpong.Game()
    X_t, R, end = game.next_frame(1)  # random first action
    # rescale image to 80,80
    X_t = cv2.cvtColor(cv2.resize(X_j, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, X_t = cv2.threshold(X_t, 1, 255, cv2.THRESH_BINARY)

    a = tf.placeholder(tf.float32, [NUM_ACTIONS, NONE])
    Q_a = tf.reduce_sum(tf.mul(a, Q_theta), reduction_indices=1)
    Y_target = tf.placeholder(tf.float32, [None])

    # initialize first sequence
    S_t = np.stack(X_t, X_t, X_t, X_t, axis=2)

    # Symbolic MSELoss function and optimizer
    MSEloss = tf.reduce_mean(tf.square(Y_target - Q_a))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(MSEloss)

    init_op = tf.initialize_all_variables()

    time = 0
    e = e_init
    while True:
        if(time > time_observation):
    	    e -= (e_final - e_init)/e_decay

    	#C += 1
    	time += 1


        if (rand() < e):
	    	# pick random action
	    action = int(2*randn())
	else:
	    	# execute action with maximum Q-value
	    action = np.argmax(Q_theta.eval(feed_dict={Qinput: S_t}))
	for i in range(K):

	    	# getreward, next image sequence
	    	# store previous sequence, current sequence, reward and action in D
	    	# Challenge: Which reward from the 'K' frames to consider?
	    X_t_1, R_t_1, end = game.next_frame(action)
	    X_t_1 = cv2.cvtColor(cv2.resize(X_t_1, (80, 80)), cv2.COLOR_BGR2GRAY)
	    ret, X_t_1 = cv2.threshold(X_t_1, 1, 255, cv2.THRESH_BINARY)
	    S_t_1.append(S_t_1, X_t_1, axis=2)

	action_vect = np.zeros([NUM_ACTIONS])
	action_vect[action] = 1
	D.append([S_t, S_t_1, R_t_1, action_vect, end])

	S_t = S_t_1


	    # take random minibatch from D(to prevent divergence of Q-value function as is likely if many consecutive states are taken)
	    # target is output from second convnet(with old parameters Theta-):
	    #	phi(j+1) -> input, y  = r + gamma*( output(phi(j+1) ), where output = max.Q value
	    # output from  first convnet(with current parameters):
	    #	phi(j) -> input, Q = predicted Q-value at j,
	    # do gradient descent to update theta
	    # After every C staeps, theta- = theta

	if (t > time_observation):
	    input_batch = random.sample(D, mini_batch_size)
	    s_j = [j[0] for j in input_batch]
	    s_j_1 = [j[1] for j in input_batch]
	    r = [j[2] for j in input_batch]
	    av = [j[3] for j in input_batch]
	    y_target = []
	    for i in range(len(input_batch)):
		if input_batch[i][4]:  # if game ended
	            y_target.append(r[i])
		else:
                    y_target.append(r[i]+gamma*np.max(Q_theta.eval(feed_dict= {Qinput:s_j_1[i]})))



            sess.run(train_step, feed_dict = {Qinput: s_j, a: av, Ytarget: y_target})
	  




	    

	    


def main():
    sess = tf.InteractiveSession()
    q, h, inp = convnet()
    train(q, h, inp)

