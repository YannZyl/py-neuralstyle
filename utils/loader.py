# -*- coding: utf-8 -*-
import h5py
import numpy as np

class Loader():          
    NET = {
          'vgg16' : 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
          'vgg19' : 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
    }
    def __init__(self, net_name):
        if net_name not in self.NET.keys():
            print('net name is not exist.')
            return 
        self.net_name = net_name
        self.file_path = './data/pre_train/' + self.NET[net_name]
        self.weight_tuple = self.loader_weights()
    
    # load weights and bias from  pre-train model
    def loader_weights(self):
        f = h5py.File(self.file_path, mode='r')
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        filtered_layer_name = []
        for name in layer_names:
            g = f[name]
            weight_names = [w.decode('utf8') for w in g.attrs['weight_names']]
            if weight_names:
                filtered_layer_name.append(name)
        layer_names = filtered_layer_name
        
        weight_value_tuple = {}
        for k, name in enumerate(layer_names):
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            weight_values = [g[weight_name] for weight_name in weight_names]
            weight_names = [name.split(':')[0] for name in weight_names]
            weight_value_tuple[name] = weight_values
        return weight_value_tuple
    
    def get_weights(self, block_num, conv_num):
        layer_name = 'block{}_conv{}'.format(block_num, conv_num)
        if layer_name not in self.weight_tuple.keys():
            print('layer name is not exist.')
            return None, None
        weights, bias = self.weight_tuple.get(layer_name)
        weights = np.array(weights, dtype=np.float32)
        bias = np.array(bias, dtype=np.float32)
        return weights, bias