# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from network.vgg19 import VGG19
from utils.improcess import read_image, save_image
try:
    reduce
except NameError:
    from functools import reduce
    
    
denoise_loss_weight = 100
content_loss_weight = 5
style_loss_weight = 500
num_iters = 1500
content_image = read_image('data/image/content-4.jpg',w=500)
style_image = read_image('data/image/style-5.jpg',w=500)
net = VGG19()
#net = VGG16()
 
g = tf.Graph()
with g.as_default(), tf.Session(graph=g) as sess:
    """step1: get content and style output"""
    cimage = net.constant_input(content_image, input_name='content')
    content_image_y, _ = net.forward(cimage)
    simage = net.constant_input(style_image, input_name='style')
    _, style_image_y = net.forward(simage, reuse=True)
    # get content and style label
    content_image_y_val = sess.run(content_image_y)
    style_image_y_val = sess.run(style_image_y)
    style_image_st_val = []
    for y in style_image_y_val:
        output_maps = y.shape[-1]
        st = np.reshape(y, [-1, output_maps])
        st = np.dot(np.transpose(st, (1,0)), st)/np.prod(st.shape)
        style_image_st_val.append(st)
        
    """step2: build train ops"""
    # initialize image/input
    gen_image = net.variable_input(content_image, input_name='generate')
    content_gen_y, style_gen_y = net.forward(gen_image, reuse=True)
    # 1.content loss
    loss_content = 0.0
    for l in range(len(content_gen_y)):
        loss_p = 2 * tf.nn.l2_loss(content_gen_y[l]-content_image_y_val[l]) / content_image_y_val[l].size  # l2 loss
        loss_p = net.alpha[l]/np.sum(net.alpha) * loss_p  # multiplay weights
        loss_content += loss_p
    # 2.style loss   
    loss_style = 0.0
    for l in range(len(style_gen_y)):
        # style loss
        _, h, w, c = map(lambda i: i.value, style_gen_y[l].get_shape())
        size = h * w * c
        st_ = tf.reshape(style_gen_y[l], [-1, c])
        st = tf.matmul(tf.transpose(st_, (1,0)), st_) / size
        loss_p = 2 * tf.nn.l2_loss(st-style_image_st_val[l]) / style_image_st_val[l].size  # l2 loss
        loss_p = net.beta[l]/np.sum(net.beta) * loss_p  # multiplay weights
        loss_style += loss_p
    # 3.denoise loss
    tv_y_size = reduce(lambda x, y: x * y, gen_image[:,1:,:,:].get_shape().as_list())
    tv_x_size = reduce(lambda x, y: x * y, gen_image[:,:,1:,:].get_shape().as_list())
    loss_total = 2 * (tf.nn.l2_loss(gen_image[:,1:,:,:]-gen_image[:,:-1,:,:]) / tv_y_size \
                               + tf.nn.l2_loss(gen_image[:,:,1:,:]-gen_image[:,:,:-1,:]) / tv_x_size)
    # loss = content + style + denoise
    loss = content_loss_weight*loss_content + style_loss_weight*loss_style + denoise_loss_weight*loss_total
    # optimizer
    train_step = tf.train.AdamOptimizer(learning_rate=2.0).minimize(loss)
    sess.run(tf.global_variables_initializer())
    for i in range(num_iters):
        if i % 20 == 0:
            gen_image_val = sess.run(gen_image)
            save_image(gen_image_val, i, 'data/output')
            print("L_content: {}, L_style: {}".format(sess.run(loss_content), sess.run(loss_style)))
        print("Iter:", i)
        sess.run(train_step)