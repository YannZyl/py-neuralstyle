# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)

def add_mean(image):
    for i in range(3):
        image[0,:,:,i] += mean[i]
    return image

def remove_mean(image):
    for c in range(3):
        image[0,:,:,c] -= mean[c]
    return image
    
def read_image(im_path, w=None):
    im = cv2.imread(im_path)
    if im is None:
        print('image path: {} not exist, please check!'.format(im_path))
        return None
    if w:
        ratio = w / float(im.shape[1])
        im = cv2.resize(im, (int(im.shape[1]*ratio), int(im.shape[0]*ratio)))
    im = im.astype(np.float32)
    im = np.expand_dims(im, 0)
    im = remove_mean(im)
    return im

def save_image(im, iteration, out_dir):
    img = im.copy()
    # Add the image mean
    img = add_mean(img)
    img = np.clip(img[0, ...],0,255).astype(np.uint8)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)    
    cv2.imwrite('{}/neural_art_iteration{}.png'.format(out_dir, iteration), img)