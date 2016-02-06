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


def convet(img):
    x = tf.placeholder("float", shape=[None, 80, 80, 4])  # input
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
    out = tf.nn.sotf.matmul(final, dense)

    return out, dense, x


def train(Q_theta, final_h, input):
	e = 0.1  # greedy probability
	D = deque()
    game= RLpong.Game()
    X_t, R, end = game.next_frame(1)
    X_t = cv2.cvtColor(cv2.resize(X_j, (80, 80)), cv2.COLOR_BGR2GRAY) #rescale image to 80,80


    while True:
	    '''
		Generate new samples
		'''
	    if (rand() < e):
	    	# pick random action
	    	action  = int(2*randn())
	    	            

	    else:
	    	# execute action with maximum Q-value
	        action = Q_theta.index(max(Q_theta(X_t)))

	    X_t_1, R_1, end = game.next_frame(action)
	    X_t_1 = cv2.cvtColor(cv2.resize(X_t_1, (80, 80)), cv2.COLOR_BGR2GRAY) #rescale image to 80,80
	    D.append(X_t, X_t_1, action, R_1)
	    X_t = X_t_1

	    Y_target = tf.placeholder(tf.float32, [None, NUM_ACTIONS])

	    MSEloss = tf.pow(Y_target - q_vals,2)
	    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(MSEloss)


	   
	        

	    # getreward, next image
	    # store processed image, action, reward, processed next image in D(reward
	    # =-1 if game terminates?)

	    '''
		Train Q-function from existing minibatch of samples
		'''

	    # take random minibatch from D(to prevent divergence of Q-value function as is likely if many consecutive states are taken)
	    # target is output from second convnet(with old parameters Theta-):
	    #	phi(j+1) -> input, y  = r + gamma*( output(phi(j+1) ), where output = max.Q value
	    # output from  first convnet(with current parameters):
	    #	phi(j) -> input, Q = predicted Q-value at j,
	    # do gradient descent to update theta
	    # After every C staeps, theta- = theta


def main():
    sess = tf.InteractiveSession()
    q, h, inp = convet()
    train(q, h, inp)
