#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""

from keras import backend as K
from keras.layers import Layer
from keras import constraints, regularizers


def flatten(data, mode):
    ndim = K.ndim(data)
    new_axes = list(range(mode, ndim)) + list(range(mode))
    data = K.permute_dimensions(data, new_axes)
    old_shape = K.shape(data)
    data = K.reshape(data, (old_shape[0], -1))
    
    return data, old_shape


def mode1_product(data, projection):
    data_shape = K.int_shape(data)
    data = K.permute_dimensions(data, (1, 2, 3, 0))
    data = K.reshape(data, (data_shape[1], -1))
    
    data = K.dot(K.transpose(projection), data)
    
    data = K.reshape(data, (K.int_shape(projection)[-1], data_shape[2], data_shape[3], -1))
    
    data = K.permute_dimensions(data, (3, 0, 1, 2))
    
    return data

def mode2_product(data, projection):
    data_shape = K.int_shape(data)
    data = K.permute_dimensions(data, (2, 3, 0, 1))
    data = K.reshape(data, (data_shape[2], -1))
    
    data = K.dot(K.transpose(projection), data)
    
    data = K.reshape(data, (K.int_shape(projection)[-1], data_shape[3], -1, data_shape[1]))
    
    data = K.permute_dimensions(data, (2, 3, 0, 1))
    
    return data

def mode3_product(data, projection):
    data_shape = K.int_shape(data)
    data = K.permute_dimensions(data, (3, 0, 1, 2))
    data = K.reshape(data, (data_shape[3], -1))
    
    data = K.dot(K.transpose(projection), data)
    
    data = K.reshape(data, (K.int_shape(projection)[-1], -1, data_shape[1], data_shape[2]))
    
    data = K.permute_dimensions(data, (1, 2, 3, 0))
    
    return data
    
class TensorSensing(Layer):

    def __init__(self, 
                 measurement_shape,
                 linear_sensing=False,
                 separate_decoder=True,
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        
        
        self.h = measurement_shape[0]
        self.w = measurement_shape[1]
        self.d = measurement_shape[2]
        self.linear_sensing = linear_sensing
        self.separate_decoder = separate_decoder
        self.constraint = constraint
        self.regularizer = regularizer
        
        super(TensorSensing, self).__init__(**kwargs)

    def build(self, input_shape):
        _, H, W, _ = input_shape

        
        if self.constraint is not None:
            constraint = constraints.max_norm(self.constraint, axis=0)
        else:
            constraint = None
            
        if self.regularizer is not None:
            regularizer = regularizers.l2(self.regularizer)
        else:
            regularizer = None
        
        self.P1_encode = self.add_weight(name='mode1_encode', 
                                          shape=(H, self.h),
                                          initializer='he_normal',
                                          constraint=constraint,
                                          regularizer=regularizer)
        
        self.P2_encode = self.add_weight(name='mode2_encode', 
                                      shape=(W, self.w),
                                      initializer='he_normal',
                                      constraint=constraint,
                                      regularizer=regularizer)
        
        self.P3_encode = self.add_weight(name='mode3_encode', 
                                      shape=(3, self.d),
                                      constraint=constraint,
                                      regularizer=regularizer,
                                      initializer='he_normal')
    
        if self.separate_decoder:
            self.P1_decode = self.add_weight(name='mode1_decode', 
                                              shape=(self.h, H),
                                              initializer='he_normal',
                                              constraint=constraint,
                                              regularizer=regularizer)
            
            self.P2_decode = self.add_weight(name='mode2_decode', 
                                          shape=(self.w, W),
                                          initializer='he_normal',
                                          constraint=constraint,
                                          regularizer=regularizer)
            
            self.P3_decode = self.add_weight(name='mode3_decode', 
                                          shape=(self.d, 3),
                                          constraint=constraint,
                                          regularizer=regularizer,
                                          initializer='he_normal')
        
        super(TensorSensing, self).build(input_shape)  # Be sure to call this at the end


    def call(self, x):
        
        encode = mode1_product(x, self.P1_encode)
        encode = mode2_product(encode, self.P2_encode)
        encode = mode3_product(encode, self.P3_encode)
        
        if not self.linear_sensing:
            encode = K.relu(encode)
            
        if self.separate_decoder:
            decode = mode1_product(encode, self.P1_decode)
            decode = mode2_product(decode, self.P2_decode)
            decode = mode3_product(decode, self.P3_decode)
        else:
            decode = mode1_product(encode, K.transpose(self.P1_encode))
            decode = mode2_product(decode, K.transpose(self.P2_encode))
            decode = mode3_product(decode, K.transpose(self.P3_encode))
        
        return decode

    def compute_output_shape(self, input_shape):
        return input_shape
    

class VectorSensing(Layer):

    def __init__(self, 
                 nb_measurement,
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        
        
        self.d = nb_measurement
        self.constraint = constraint
        self.regularizer = regularizer
        
        super(VectorSensing, self).__init__(**kwargs)

    def build(self, input_shape):
        self.in_shape = input_shape
        _, H, W, _ = input_shape
        D = H*W
        d = self.d
        
        
        if self.constraint is not None:
            constraint = constraints.max_norm(self.constraint, axis=0)
        else:
            constraint = None
            
        if self.regularizer is not None:
            regularizer = regularizers.l2(self.regularizer)
        else:
            regularizer = None
        

        self.E1 = self.add_weight(name='projection1', 
                                      shape=(D, d),
                                      initializer='he_normal',
                                      constraint=constraint,
                                      regularizer=regularizer)
        
        self.E2 = self.add_weight(name='projection2', 
                                      shape=(D, d),
                                      initializer='he_normal',
                                      constraint=constraint,
                                      regularizer=regularizer)
        
        self.E3 = self.add_weight(name='projection3', 
                                      shape=(D, d),
                                      initializer='he_normal',
                                      constraint=constraint,
                                      regularizer=regularizer)
        
        super(VectorSensing, self).build(input_shape)


    def call(self, x):
        
        x = K.reshape(x, (-1, self.in_shape[1]*self.in_shape[2], 3))
        
        R = x[:, :, 0]
        G = x[:, :, 1]
        B = x[:, :, 2]
        
        R_encode = K.dot(R, self.E1)
        G_encode = K.dot(G, self.E2)
        B_encode = K.dot(B, self.E3)
        
        
        R_decode = K.dot(R_encode, K.transpose(self.E1))
        G_decode = K.dot(G_encode, K.transpose(self.E2))
        B_decode = K.dot(B_encode, K.transpose(self.E3))
        
        R_decode = K.reshape(R_decode, (-1, self.in_shape[1], self.in_shape[2], 1))
        G_decode = K.reshape(G_decode, (-1, self.in_shape[1], self.in_shape[2], 1))
        B_decode = K.reshape(B_decode, (-1, self.in_shape[1], self.in_shape[2], 1))

        y = K.concatenate((R_decode, G_decode, B_decode), axis=-1)
            
        return y

    def compute_output_shape(self, input_shape):
        return input_shape