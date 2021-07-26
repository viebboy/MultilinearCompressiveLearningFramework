#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""
import math

import numpy as np
from tensorflow.keras.utils import to_categorical
import os
import pickle
from keras.preprocessing.image import ImageDataGenerator
import exp_configurations as Conf
import itertools

def create_configuration(hyperparameter_list, value_list):
    
    configurations = []
    
    for names, values in zip(hyperparameter_list, value_list):
        v_list = []
        for name in names:
            v_list.append(values[name])
        
        configurations += list(itertools.product(*v_list))
   
    configurations = list(set(configurations))
    configurations.sort(key=getSortableKey)
    return configurations

def getSortableKey(x):
    sortKey = []
    for element in x:
        # To avoid tuples with NoneType elements causing error while sorting
        if isinstance(element, tuple):
            theTuple = element;
            nonesRemoved = [-1 * math.inf if item is None else item for item in theTuple]
            sortKey.append(tuple(nonesRemoved));
        else:
            sortKey.append(element);

    return sortKey

def flatten(data, mode):
    
    ndim = data.ndim
    new_axes = list(range(mode, ndim)) + list(range(mode))
    
    data = np.transpose(data, axes=new_axes)
    
    old_shape = data.shape
    data = np.reshape(data, (old_shape[0], -1))
    
    return data, old_shape

def nmodeproduct(data, projection, mode):
    
    data, old_shape = flatten(data, mode)
    
    data = np.dot(projection.T, data)
    
    new_shape = list(old_shape)
    
    new_shape[0] = projection.shape[-1]
    
    
    data = np.reshape(data, new_shape)
    
    new_axes = list(range(data.ndim))
    new_axes = list(new_axes[-mode:]) + list(new_axes[:-mode])
    
    data = np.transpose(data, new_axes)
    
    return data
    
def MSE(a,b):
    return np.mean((a.flatten()-b.flatten())**2)
    
def HOSVD(data, h, w, d, centering=False, iteration=100, threshold=1e-4, regularization=0.0):

    
    N, H, W, _ = data.shape
        
    W1 = np.random.rand(H, h)
    W2 = np.random.rand(W,w)
    W3 = np.random.rand(3,d)
    
    if centering:
        mean_tensor = np.mean(data, axis=0, keepdims=True)
    else:
        mean_tensor = np.zeros(data.shape[1:])
        mean_tensor = np.expand_dims(mean_tensor, axis=0)
    
    data -= mean_tensor
    
    for i in range(iteration):
        print('iteration: %s' % str(i))
        """ compute W1 by fixing W2, W3"""
        # project mode 2, 3 -> data: N x H x w x 2
        data_tmp = nmodeproduct(data, W2, 2)
        data_tmp = nmodeproduct(data_tmp, W3, 3)
        
        # flatten to H x (N*w*2)
        data_tmp, _ = flatten(data, 1)
        cov = np.dot(data_tmp, data_tmp.T) + regularization*np.eye(H)
        U, _, _ = np.linalg.svd(cov)
        W1_new = U[:, :h]
        
        """ compute W2 by fixing W1, W3"""
        # project mode 1, 3 -> data: N x h x W x 2
        data_tmp = nmodeproduct(data, W1_new, 1)
        data_tmp = nmodeproduct(data_tmp, W3, 3)
        
        # flatten to W x (N*h*2)
        data_tmp, _ = flatten(data, 2)
        cov = np.dot(data_tmp, data_tmp.T) + regularization*np.eye(W)
        U, _, _ = np.linalg.svd(cov)
        W2_new = U[:, :w]
        
        """ compute W3 by fixing W1, W2"""
        # project mode 1, 2 -> data: N x h x w x 3
        data_tmp = nmodeproduct(data, W1_new, 1)
        data_tmp = nmodeproduct(data_tmp, W2_new, 2)
        
        # flatten to 3 x (N*h*w)
        data_tmp, _ = flatten(data, 3)
        cov = np.dot(data_tmp, data_tmp.T) + regularization*np.eye(3)
        U, _, _ = np.linalg.svd(cov)
        W3_new = U[:, :d]
        
        """ calculate error """
        data_tmp = nmodeproduct(data, W1, 1)
        data_tmp = nmodeproduct(data_tmp, W2, 2)
        data_tmp = nmodeproduct(data_tmp, W3, 3)
        
        data_tmp = nmodeproduct(data_tmp, W1.T, 1)
        data_tmp = nmodeproduct(data_tmp, W2.T, 2)
        data_tmp = nmodeproduct(data_tmp, W3.T, 3)
        
        print('Residual error: %.4f' % MSE(data_tmp, data))
        
        projection_error = MSE(W1, W1_new) + MSE(W2, W2_new) + MSE(W3, W3_new)
        print('Projection error: %.4f' % projection_error)
        
        W1 = W1_new
        W2 = W2_new
        W3 = W3_new
        
        if projection_error  < threshold:
            break
        
    return W1, W2, W3
        
def load_data(name):
    x_train = np.load(os.path.join(Conf.DATA_DIR, name + '_x_train.npy'))
    y_train = np.load(os.path.join(Conf.DATA_DIR, name + '_y_train.npy'))
    
    x_val = np.load(os.path.join(Conf.DATA_DIR, name + '_x_val.npy'))
    y_val = np.load(os.path.join(Conf.DATA_DIR, name + '_y_val.npy'))
    
    x_test = np.load(os.path.join(Conf.DATA_DIR, name + '_x_test.npy'))
    y_test = np.load(os.path.join(Conf.DATA_DIR, name + '_y_test.npy'))
    
    n_class = np.unique(y_train).size
    
    y_train = to_categorical(y_train, n_class)
    y_val = to_categorical(y_val, n_class)
    y_test = to_categorical(y_test, n_class)
    
    return x_train, y_train, x_val, y_val, x_test, y_test


def load_HOSVD_matrix(dataset, height, width, depth):
    filename = 'HOSVD_%s_%s_%s_%s.pickle' % (dataset, str(height), str(width), str(depth))
    path = os.path.join(Conf.DATA_DIR, filename)
    
    if not os.path.exists(path):
        x_train = np.load(os.path.join(Conf.DATA_DIR, dataset + '_x_train.npy'))
        W1, W2, W3 = HOSVD(x_train, height, width, depth)
        fid = open(path, 'wb')
        projection = {'W1': W1, 'W2': W2, 'W3': W3}
        pickle.dump(projection, fid)
        fid.close()
        
    else:
        fid = open(path, 'rb')
        projection = pickle.load(fid)
        fid.close()
        W1, W2, W3 = projection['W1'], projection['W2'], projection['W3']
    
    return W1, W2, W3


def PCA_(X, d, centering=False):
    # X: None x H x W
    N, H, W = X.shape
    
    X_mean = np.mean(X, axis=0, keepdims=True) if centering else np.zeros((1, H, W))
    X -= X_mean
    
    X = np.reshape(X, (N, H*W))
    
    cov = np.dot(X.T, X)
    U, _, _ = np.linalg.svd(cov)
    projection = U[:, :d]
    
    return projection, X_mean
def PCA(data, d, centering=False):
    
    _, H, W, _ = data.shape
    R = data[:, :, :, 0]
    G = data[:, :, :, 0]
    B = data[:, :, :, 0]
    
    R_p, R_mean = PCA_(R, d, centering)
    G_p, G_mean = PCA_(G, d, centering)
    B_p, B_mean = PCA_(B, d, centering)
    
    R_mean = np.expand_dims(R_mean, axis=-1)
    G_mean = np.expand_dims(G_mean, axis=-1)
    B_mean = np.expand_dims(B_mean, axis=-1)
    
    mean = np.concatenate((R_mean, G_mean, B_mean), axis=-1)
    
    return [R_p, G_p, B_p], mean

def load_PCA_matrix(dataset, d):
    filename = 'PCA_%s_%s.pickle' % (dataset, str(d))
    path = os.path.join(Conf.DATA_DIR, filename)
    
    if not os.path.exists(path):
        x_train = np.load(os.path.join(Conf.DATA_DIR, dataset + '_x_train.npy'))
        projection, _ = PCA(x_train, d)
        fid = open(path, 'wb')
        projection = {'projection': projection}
        
        pickle.dump(projection, fid)
        fid.close()
        
    else:
        fid = open(path, 'rb')
        projection = pickle.load(fid)
        fid.close()
        projection = projection['projection']
    
    return projection

def get_data_generator(x, y, batch_size=32, shuffle=False, augmentation=False):
    N = x.shape[0]
    steps = int(np.ceil(N/float(batch_size)))
    
    if augmentation:
        gen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True)
    else:
        gen = ImageDataGenerator()
    
    gen.fit(x)
    
    return gen.flow(x, y, batch_size, shuffle), steps






        
    
    
