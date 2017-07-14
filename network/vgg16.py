# -*- coding: utf-8 -*-
import tensorflow as tf
from .network import Network

class VGG16(Network):
    def __init__(self, net_name='vgg16'):
        Network.__init__(self, net_name)
        self.alpha = [1]    
        self.beta = [1,1,1,1,1]
    
    def constant_input(self, image, input_name):
        return tf.constant(image, name=input_name)
    
    def variable_input(self, image, input_name):
        # inputs
        with tf.variable_scope(input_name):
            input_ = tf.get_variable('input', shape=image.shape, dtype=tf.float32, 
                                     initializer=tf.constant_initializer(image), trainable=True)
            return input_
            
    def forward(self, input_, reuse=None):
        # vgg16 net
        with tf.variable_scope('vgg16', reuse=reuse):
            # block1: 2 convs + 1 pool
            with tf.variable_scope('block1'):
                conv1_1 = self.conv2d(input_, 1, 1)
                conv1_2 = self.conv2d(conv1_1, 1, 2)
                pool1 = self.pool(conv1_2, 1)
            # block2: 2 convs + 1 pool
            with tf.variable_scope('block2'):
                conv2_1 = self.conv2d(pool1, 2, 1)
                conv2_2 = self.conv2d(conv2_1, 2, 2)
                pool2 = self.pool(conv2_2, 2)
            # block3: 3 convs + 1 pool
            with tf.variable_scope('block3'):
                conv3_1 = self.conv2d(pool2, 3, 1)
                conv3_2 = self.conv2d(conv3_1, 3, 2)
                conv3_3 = self.conv2d(conv3_2, 3, 3)
                pool3 = self.pool(conv3_3, 3)
            # block4: 3 convs + 1 pool
            with tf.variable_scope('block4'):
                conv4_1 = self.conv2d(pool3, 4, 1)
                conv4_2 = self.conv2d(conv4_1, 4, 2)
                conv4_3 = self.conv2d(conv4_2, 4, 3)
                pool4 = self.pool(conv4_3, 4)
            # block5: 3 convs + 1 pool
            with tf.variable_scope('block5'):
                conv5_1 = self.conv2d(pool4, 5, 1)
                #conv5_2 = self.conv2d(conv5_1, 5, 2)
                #conv5_3 = self.conv2d(conv5_2, 5, 3)
                #pool5 = self.pool(conv5_3, 5)
                
        return [conv4_2], [conv1_1, conv2_1, conv3_1, conv4_1, conv5_1]