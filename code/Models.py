#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi

Significant proportion of this file borrows from https://keras.io/examples/cifar10_resnet/
"""

import Layers
from keras.layers import Input, Conv2D, Dropout, Activation, Dense, Add, BatchNormalization as BN, AveragePooling2D, Flatten, GlobalAveragePooling2D, Concatenate
from keras import Model, regularizers, constraints
import tensorflow as tf

def resnet_layer(inputs,
                 prefix,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 regularizer=None,
                 constraint=None):
    
    
    if regularizer is not None:
        regularizer = regularizers.l2(regularizer)
    
    if constraint is not None:
        constraint = constraints.max_norm(constraint, axis=[0,1,2])
        

    

    x = inputs
    if conv_first:
        x = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizer,
                  kernel_constraint=constraint,
                  name=prefix + '_conv1')(x)
        
        if batch_normalization:
            x = BN(name=prefix + '_BN')(x)
        if activation is not None:
            x = Activation(activation, name=prefix + '_activation')(x)
    else:
        if batch_normalization:
            x = BN(name=prefix + '_BN')(x)
        if activation is not None:
            x = Activation(activation, name=prefix + '_activation')(x)
        x = Conv2D(num_filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizer,
                   kernel_constraint=constraint,
                   name=prefix + '_conv1')(x)
    return x


def resnet_module(inputs, input_shape, depth=110, num_filters_in=16, regularizer=None, constraint=None):
    
    if input_shape[0] < 64:
        f_window = 3
        f_strides = (1,1)
        num_filters_in = 16
    else:
        f_window = 5
        f_strides = (2,2)
        num_filters_in = 24
        
    num_res_blocks = int((depth - 2) / 9)
    x = resnet_layer(inputs=inputs,
                     prefix='resnet_initial_block',
                     num_filters=num_filters_in,
                     kernel_size = f_window,
                     strides = f_strides,
                     conv_first=True,
                     regularizer=regularizer,
                     constraint=constraint)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             prefix='resnet_%s_%s_1' %(str(stage), str(res_block)),
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False,
                             regularizer=regularizer,
                             constraint=constraint)
            
            y = resnet_layer(inputs=y,
                             prefix='resnet_%s_%s_2' %(str(stage), str(res_block)),
                             num_filters=num_filters_in,
                             conv_first=False,
                             regularizer=regularizer,
                             constraint=constraint)
            
            y = resnet_layer(inputs=y,
                             prefix='resnet_%s_%s_3' %(str(stage), str(res_block)),
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False,
                             regularizer=regularizer,
                             constraint=constraint)
            
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 prefix='resnet_%s_%s_interpolate' %(str(stage), str(res_block)),
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 regularizer=regularizer,
                                 constraint=constraint)
                
            x = Add()([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BN(name='resnet_bn_final')(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)


    return y


def get_baseline_resnet(input_shape, n_class, regularizer=None, constraint=None):
    
    inputs = Input(input_shape)
    
    features = resnet_module(inputs, input_shape, regularizer=regularizer, constraint=constraint)
    
    if regularizer is not None:
        output_regularizer = regularizers.l2(regularizer)
    else:
        output_regularizer = None
        
    if regularizer is not None:
        output_constraint = constraints.max_norm(constraint, axis=0)
    else:
        output_constraint = None
        
    
    outputs = Dense(n_class, 
                    activation='softmax', 
                    kernel_initializer='he_normal', 
                    kernel_regularizer=output_regularizer,
                    kernel_constraint=output_constraint,
                    name='resnet_prediction')(features)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def allcnn_module(inputs, input_shape, regularizer, constraint):
    
    if regularizer is not None:
        conv_regularizer = regularizers.l2(regularizer)
    else:
        conv_regularizer = None

    if constraint is not None:
        conv_constraint = constraints.max_norm(constraint, axis=[0,1,2])
    else:
        conv_constraint= None
        
    if input_shape[0] == 32:
        nb_filter = 36
        filter_shape = (3,3)
        strides = (1,1)
    else:
        nb_filter = 16
        filter_shape = (5,5)
        strides = (2,2)
        
    hiddens = Conv2D(nb_filter, filter_shape, strides=strides, padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name='allcnn_conv1_1')(inputs)
    hiddens = BN(name='allcnn_bn1_1')(hiddens)
    hiddens = Activation('relu', name='activation1_1')(hiddens)
    
    hiddens = Conv2D(96, (3,3), strides=(1,1), padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name='allcnn_conv1_2')(hiddens)
    hiddens = BN(name='allcnn_bn1_2')(hiddens)
    hiddens = Activation('relu', name='activation1_2')(hiddens)
    
    hiddens = Conv2D(96, (3,3), strides=(2,2), padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name='allcnn_conv1_3')(hiddens)
    hiddens = BN(name='allcnn_bn1_3')(hiddens)
    hiddens = Activation('relu', name='activation1_3')(hiddens)
    
    #
    hiddens = Dropout(0.2, name='dropout1')(hiddens)
    
    hiddens = Conv2D(192, (3,3), strides=(1,1), padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name='allcnn_conv2_1')(hiddens)
    hiddens = BN(name='allcnn_bn2_1')(hiddens)
    hiddens = Activation('relu', name='activation2_1')(hiddens)
    
    hiddens = Conv2D(192, (3,3), strides=(1,1), padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name='allcnn_conv2_2')(hiddens)
    hiddens = BN(name='allcnn_bn2_2')(hiddens)
    hiddens = Activation('relu', name='activation2_2')(hiddens)
    
    hiddens = Conv2D(192, (3,3), strides=(2,2), padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name='allcnn_conv2_3')(hiddens)
    hiddens = BN(name='allcnn_bn2_3')(hiddens)
    hiddens = Activation('relu', name='activation2_3')(hiddens)

    #
    hiddens = Dropout(0.2, name='dropout2')(hiddens)
    
    hiddens = Conv2D(192, (3,3), strides=(1,1), padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name='allcnn_conv3_1')(hiddens)
    hiddens = BN(name='allcnn_bn3_1')(hiddens)
    hiddens = Activation('relu', name='activation3_1')(hiddens)
    
    hiddens = Conv2D(192, (1,1), strides=(1,1), padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name='allcnn_conv3_2')(hiddens)
    hiddens = BN(name='allcnn_bn3_2')(hiddens)
    hiddens = Activation('relu', name='activation3_2')(hiddens)
    
    return hiddens

def get_baseline_allcnn(input_shape, n_class, regularizer=None, constraint=None):
    
    inputs = Input(input_shape, name='inputs')
    
    if regularizer is not None:
        conv_regularizer = regularizers.l2(regularizer)
    else:
        conv_regularizer = None

    if constraint is not None:
        conv_constraint = constraints.max_norm(constraint, axis=[0,1,2])
    else:
        conv_constraint= None
    
    
    hiddens = allcnn_module(inputs, input_shape, regularizer, constraint)
    
    hiddens = Conv2D(n_class, (1,1), strides=(1,1), padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name='allcnn_conv3_3')(hiddens)
    hiddens = BN(name='allcnn_bn3_3')(hiddens)
    hiddens = Activation('relu', name='activation3_3')(hiddens)
    
    hiddens = GlobalAveragePooling2D(name='average_pooling')(hiddens)
    
    outputs = Activation('softmax', name='allcnn_prediction')(hiddens)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    
    return model

def sensing_module(inputs,
                   measurement_shape,
                   sensing_mode,
                   linear_sensing,
                   separate_decoder,
                   weight_decay,
                   weight_constraint):
    
    if sensing_mode == 'vector':
        decode = Layers.VectorSensing(nb_measurement=measurement_shape[0],
                                      regularizer=weight_decay, 
                                      constraint=weight_constraint,
                                      name='sensing')(inputs)
    elif sensing_mode == 'tensor':
        decode = Layers.TensorSensing(measurement_shape,
                                      linear_sensing=linear_sensing,  
                                      separate_decoder=separate_decoder,
                                      regularizer=weight_decay, 
                                      constraint=weight_constraint,
                                      name='sensing')(inputs)
        
    return decode

def get_sensing_model(input_shape,
                      measurement_shape,
                      sensing_mode,
                      linear_sensing,
                      separate_decoder):
    
    inputs = Input(input_shape)
    outputs = sensing_module(inputs,
                             measurement_shape,
                             sensing_mode,
                             linear_sensing,
                             separate_decoder,
                             None,
                             None)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def get_allcnn(input_shape, 
               n_class, 
               measurement_shape,
               sensing_mode,
               linear_sensing,
               separate_decoder,
               projection_regularizer=None,
               projection_constraint=None, 
               regularizer=None,
               constraint=None):
    
    if regularizer is not None:
        conv_regularizer = regularizers.l2(regularizer)
    else:
        conv_regularizer = None

    if constraint is not None:
        conv_constraint = constraints.max_norm(constraint, axis=[0,1,2])
    else:
        conv_constraint= None
        
    inputs = Input(input_shape, name='inputs')
    
    decode = sensing_module(inputs,
                            measurement_shape,
                            sensing_mode,
                            linear_sensing,
                            separate_decoder,
                            projection_regularizer,
                            projection_constraint)
    
    hiddens = allcnn_module(decode, input_shape, regularizer, constraint)
    
    hiddens = Conv2D(n_class, (1,1), strides=(1,1), padding='same',
                     kernel_regularizer=conv_regularizer,
                     kernel_constraint=conv_constraint,
                     name='allcnn_conv3_3')(hiddens)
    hiddens = BN(name='allcnn_bn3_3')(hiddens)
    hiddens = Activation('relu', name='activation3_3')(hiddens)
    
    hiddens = GlobalAveragePooling2D(name='average_pooling')(hiddens)
    
    outputs = Activation('softmax', name='allcnn_prediction')(hiddens)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    
    return model

def get_resnet(input_shape, 
               n_class, 
               measurement_shape,
               sensing_mode,
               linear_sensing,
               separate_decoder,
               projection_regularizer=None,
               projection_constraint=None, 
               regularizer=None,
               constraint=None):
    
    inputs = Input(input_shape)
    
    decode = sensing_module(inputs,
                            measurement_shape,
                            sensing_mode,
                            linear_sensing,
                            separate_decoder,
                            projection_regularizer,
                            projection_constraint)
    
    features = resnet_module(decode, input_shape, regularizer=regularizer, constraint=constraint)
    
    
    if regularizer is not None:
        output_regularizer = regularizers.l2(regularizer)
    else:
        output_regularizer = None
        
    if regularizer is not None:
        output_constraint = constraints.max_norm(constraint, axis=0)
    else:
        output_constraint = None
        
    
    outputs = Dense(n_class, 
                    activation='softmax', 
                    kernel_initializer='he_normal', 
                    kernel_regularizer=output_regularizer,
                    kernel_constraint=output_constraint,
                    name='resnet_prediction')(features)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model
    
