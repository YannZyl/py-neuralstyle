# -*- coding: utf-8 -*-
import tensorflow as tf
from utils.loader import Loader

class Network():
    def __init__(self, net_name):
        self.loader = Loader(net_name)
        self.activate = {}
        
    def conv2d(self, input_tensor, block_num, conv_num):
        with tf.variable_scope('conv{}'.format(conv_num)):
            w, b = self.loader.get_weights(block_num, conv_num)
            #weights = tf.get_variable('W', shape=w.shape, dtype=tf.float32, initializer=tf.constant_initializer(w), trainable=False)
            #bias = tf.get_variable('b', shape=b.shape, dtype=tf.float32, initializer=tf.constant_initializer(b), trainable=False)
            conv_out = tf.nn.conv2d(input_tensor, w, strides=[1,1,1,1], padding='SAME')
            conv_out = tf.nn.bias_add(conv_out, b)
            conv_out = tf.nn.relu(conv_out)
            self.activate['conv{}_{}'.format(block_num, conv_num)] = conv_out
            return conv_out
        
    def pool(self, input_tensor, block_num):
        return tf.nn.max_pool(input_tensor, ksize=[1,2,2,1], strides=[1,2,2,1], 
                              padding='SAME', name='pool{}'.format(block_num))
    