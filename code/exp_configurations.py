#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""
import os

LR = (1e-3, 1e-3, 1e-4, 1e-5)
EPOCH = (40, 40, 40, 40)

LOG_INTERVAL = 5

LOG_DIR = os.path.join(os.path.dirname(os.getcwd()), 'log')
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
    
BASELINE_DIR = os.path.join(os.path.dirname(os.getcwd()), 'baseline_models')
if not os.path.exists(BASELINE_DIR):
    os.mkdir(BASELINE_DIR)
    
OUTPUT_DIR = os.path.join(os.path.dirname(os.getcwd()), 'output')
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
    
    
DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), 'data')
if not os.path.exists(DATA_DIR):
    raise Exception('data folder does not exist')
    
""" baseline model configuration """

baseline_var_names = ['dataset',
                      'classifier',
                      'regularizer',
                      'LR',
                      'Epoch']

conf_baseline = {'dataset': ['cfw32x32'],
                            'classifier': ['allcnn'],
                            'regularizer': [(1e-4, None)],
                            'LR': [LR],
                            'Epoch': [EPOCH]}


""" compressive sensing configurations """

hyperparameters = ['dataset',
                  'classifier',
                  'target_shape',
                  'sensing_mode',
                  'linear_sensing',
                  'regularizer',
                  'exp',
                  'LR',
                  'Epoch',
                  'separate_decoder',
                  'precompute_sensing',
                  'precompute_classifier']



""" vector configuration """

conf_vector = {'dataset': ['cfw32x32', 'cifar10', 'cifar100'],
                          'classifier': ['allcnn','resnet'],
                          'target_shape': [(256, None, None), (102, None, None), (18, None, None)],
                          'sensing_mode': ['vector'],
                          'linear_sensing': [True],
                          'regularizer': [(1e-4, None)],
                          'exp': [0, 1, 2],
                          'LR': [LR],
                          'Epoch': [EPOCH],
                          'separate_decoder': [False],
                          'precompute_sensing': [True],
                          'precompute_classifier': [True]}

""" tensor configuration """

conf_tensor_standard = {'dataset': ['cfw32x32', 'cifar10', 'cifar100'],
                          'classifier': ['allcnn', 'resnet'],
                          'target_shape': [(20, 19, 2), (28, 27, 1), (14, 11, 2), (18, 17, 1), (9, 6, 1), (6, 9, 1)],
                          'sensing_mode': ['tensor'],
                          'linear_sensing': [True],
                          'end_to_end': [True],
                          'mean_update': [False],
                          'augmentation': [True],
                          'regularizer': [(1e-4, None)],
                          'exp': [0, 1, 2],
                          'LR': [LR],
                          'Epoch': [EPOCH],
                          'separate_decoder': [True],
                          'precompute_sensing': [True],
                          'precompute_classifier': [True],
                          'centering': [False]}

conf_tensor_resolution = {'dataset': ['cfw32x32', 'cfw48x48', 'cfw64x64', 'cfw80x80'],
                          'classifier': ['allcnn'],
                          'target_shape': [(20, 19, 2), (28, 27, 1), (14, 11, 2), (18, 17, 1), (9, 6, 1), (6, 9, 1)],
                          'sensing_mode': ['tensor'],
                          'linear_sensing': [True],
                          'end_to_end': [True],
                          'mean_update': [False],
                          'augmentation': [True],
                          'regularizer': [(1e-4, None)],
                          'exp': [0, 1, 2],
                          'LR': [LR],
                          'Epoch': [EPOCH],
                          'separate_decoder': [True],
                          'precompute_sensing': [True],
                          'precompute_classifier': [True],
                          'centering': [False]}


conf_tensor_hp = {'dataset': ['cfw32x32', 'cfw48x48', 'cfw64x64', 'cfw80x80', 'cifar10', 'cifar100'],
                          'classifier': ['allcnn'],
                          'target_shape': [(20, 19, 2), (28, 27, 1), (14, 11, 2), (18, 17, 1), (9, 6, 1), (6, 9, 1)],
                          'sensing_mode': ['tensor'],
                          'linear_sensing': [True, False],
                          'end_to_end': [True],
                          'mean_update': [False],
                          'augmentation': [True],
                          'regularizer': [(1e-4, None)],
                          'exp': [0, 1, 2],
                          'LR': [LR],
                          'Epoch': [EPOCH],
                          'separate_decoder': [True, False],
                          'precompute_sensing': [True],
                          'precompute_classifier': [True],
                          'centering': [False]} 

conf_tensor_initialization = {'dataset': ['cfw32x32', 'cifar10', 'cifar100'],
                          'classifier': ['allcnn'],
                          'target_shape': [(28, 27, 1), (18, 17, 1), (9, 6, 1)],
                          'sensing_mode': ['tensor'],
                          'linear_sensing': [True],
                          'end_to_end': [True],
                          'mean_update': [False],
                          'augmentation': [True],
                          'regularizer': [(1e-4, None)],
                          'exp': [0, 1, 2],
                          'LR': [LR],
                          'Epoch': [EPOCH],
                          'separate_decoder': [True],
                          'precompute_sensing': [True, False],
                          'precompute_classifier': [True, False],
                          'centering': [False]}






