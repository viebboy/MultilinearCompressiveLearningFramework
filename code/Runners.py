#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""
import Models
import Utility, numpy as np, os
import pickle
import inspect
import exp_configurations as Conf
from tqdm import tqdm

OUTPUT_DIR = Conf.OUTPUT_DIR
BASELINE_DIR = Conf.BASELINE_DIR
LOG_DIR = Conf.LOG_DIR

def get_batch_size(sensing_mode):
    batch_size = 32
    
    if sensing_mode == 'vector':
        batch_size = 16
    
    return batch_size

def train_baseline(args):
    
    dataset, classifier, regularizer, LR, Epoch = args
    
    weight_decay, constraint = regularizer
    
    x_train, y_train, x_val, y_val, x_test, y_test = Utility.load_data(dataset)
    
    batch_size = get_batch_size('tensor')
    
    train_gen, train_steps = Utility.get_data_generator(x_train, y_train, shuffle=True, augmentation=True, batch_size=batch_size)
    val_gen, val_steps = Utility.get_data_generator(x_val, y_val, shuffle=False, augmentation=False, batch_size=batch_size)
    test_gen, test_steps = Utility.get_data_generator(x_test, y_test, shuffle=False, augmentation=False, batch_size=batch_size)
    
    input_shape = x_train.shape[1:]
    n_class = y_train.shape[-1]
    
    current_weights = None
    optimal_weights = None
    optimal_acc = 0.0
    
    history = {'train_loss':[], 'train_acc':[], 'val_acc': [], 'val_loss':[]}
    
    if classifier == 'resnet':
        get_model = Models.get_baseline_resnet
    elif classifier == 'allcnn':
        get_model = Models.get_baseline_allcnn
    else:
        raise Exception('Unknown classifier: %s' % classifier)
        
    
    for lr, epoch in zip(LR, Epoch):
        model = get_model(input_shape=input_shape,
                          n_class=n_class,
                          regularizer=weight_decay,
                          constraint=constraint)
        
        model.compile('adam', 'categorical_crossentropy', ['acc',])
        
        if current_weights is not None:
            model.set_weights(current_weights)
        
        model.optimizer.lr = lr
        
        for iteration in range(epoch):
            h = model.fit_generator(train_gen, train_steps, epochs=1, verbose=0)

            val_performance = model.evaluate_generator(val_gen, val_steps)
            acc_index = model.metrics_names.index('acc')
            loss_index = model.metrics_names.index('loss')
            
            if val_performance[acc_index] > optimal_acc:
                optimal_acc = val_performance[acc_index]
                optimal_weights = model.get_weights()
                
            history['train_loss'] += h.history['loss']
            history['train_acc'] += h.history['acc']
            history['val_loss'].append(val_performance[loss_index])
            history['val_acc'].append(val_performance[acc_index])
            
        
        current_weights = model.get_weights()
    
    
    model.set_weights(optimal_weights)
    
    train_p = model.evaluate_generator(train_gen, train_steps)
    val_p = model.evaluate_generator(val_gen, val_steps)
    test_p = model.evaluate_generator(test_gen, test_steps)
    
    train_performance = {}
    val_performance = {}
    test_performance = {}
    
    for index, metric in enumerate(model.metrics_names):
        train_performance[metric] = train_p[index]
        val_performance[metric] = val_p[index]
        test_performance[metric] = test_p[index]
        
    weights = {}
    for layer in model.layers:
        if layer.name.startswith(classifier):
            weights[layer.name] = layer.get_weights()
    
    
    data = {'weights': weights,
            'train_acc': train_performance['acc'],
            'val_acc': val_performance['acc'],
            'test_acc': test_performance['acc'],
            'train_categorical_crossentropy': train_performance['loss'],
            'val_categorical_crossentropy': val_performance['loss'],
            'test_categorical_crossentropy': test_performance['loss'],
            'history': history}
    
    return data

def train_sensing(args):
    dataset = args[0]
    classifier = args[1]
    measurement_shape = args[2]
    sensing_mode = args[3]
    linear_sensing = args[4] 
    regularizer = args[5]
    projection_decay, projection_constraint = regularizer
    weight_decay, weight_constraint = regularizer
    exp = args[6]
    LR = args[7]
    Epoch = args[8]
    separate_decoder = args[9]
    precompute_sensing = args[10]
    precompute_classifier = args[11]
    
    """ if exist result in the output dir, load and return """
    result_file = '_'.join([str(v) for v in args]) + '.pickle'
    result_file = os.path.join(OUTPUT_DIR, result_file)
    if os.path.exists(result_file):
        fid = open(result_file, 'rb')
        outputs = pickle.load(fid)
        fid.close()
        print('Exist output file')
        return outputs
    

    print('load pretrained baseline model')
    baseline_filename = '_'.join([str(dataset), str(classifier), str(regularizer), str(LR), str(Epoch)]) + '.pickle'
    baseline_filename = os.path.join(BASELINE_DIR, baseline_filename)
    
    if not os.path.exists(baseline_filename):
        raise RuntimeError('Baseline model doesnt exist!')
    
    with open(baseline_filename, 'rb') as fid:
        baseline_model = pickle.load(fid)
    
    print('baseline %s performance' % classifier)
    print('train accuracy: %.4f' % baseline_model['train_acc'])
    print('val accuracy: %.4f' % baseline_model['val_acc'])
    print('test accuracy: %.4f' % baseline_model['test_acc'])
        
    baseline_weights = baseline_model['weights']
    
    print('load dataset')
    """ handle dataset """
    x_train, y_train, x_val, y_val, x_test, y_test = Utility.load_data(dataset)
    input_shape = x_train.shape[1:]
    n_class = y_train.shape[-1]
    
    batch_size = get_batch_size(sensing_mode)
    
    train_gen, train_steps = Utility.get_data_generator(x_train,
                                                        y_train,
                                                        shuffle=True,
                                                        augmentation=True,
                                                        batch_size=batch_size)
    train_gen_fix, train_steps_fix = Utility.get_data_generator(x_train,
                                                                y_train,
                                                                shuffle=False,
                                                                augmentation=False,
                                                                batch_size=batch_size)
    val_gen, val_steps = Utility.get_data_generator(x_val,
                                                    y_val,
                                                    shuffle=False,
                                                    augmentation=False,
                                                    batch_size=batch_size)
    test_gen, test_steps = Utility.get_data_generator(x_test,
                                                      y_test,
                                                      shuffle=False,
                                                      augmentation=False,
                                                      batch_size=batch_size)
    

    height, width, depth = measurement_shape
    """ handle sensing matrix """
    if sensing_mode == 'vector':
        W = Utility.load_PCA_matrix(dataset, height)
        
    else:
        W1, W2, W3 = Utility.load_HOSVD_matrix(dataset, height, width, depth)
        
        
    """ training """
    
    if classifier == 'resnet':
        get_model = Models.get_resnet
    elif classifier == 'allcnn':
        get_model = Models.get_allcnn
    else:
        raise Exception('Unknown classifier: %s' % classifier)
        
    model = get_model(input_shape=input_shape, 
                      n_class=n_class, 
                      measurement_shape=measurement_shape,
                      sensing_mode=sensing_mode, 
                      linear_sensing=linear_sensing, 
                      separate_decoder=separate_decoder,
                      projection_regularizer=projection_decay, 
                      projection_constraint=projection_constraint, 
                      regularizer=weight_decay, 
                      constraint=weight_constraint)

    model.compile('adam', 'categorical_crossentropy', ['acc'])
    metrics = model.metrics_names
    
    log_file = '_'.join([str(v) for v in args]) + '.pickle'
    log_file = os.path.join(LOG_DIR, log_file)
    
    if not os.path.exists(log_file):
    
        """ set pretrained classifier weights if precompute_classifier is True """
        if precompute_classifier:
            print('set the model weights')
            for layer_name in baseline_weights.keys():
                model.get_layer(layer_name).set_weights(baseline_weights[layer_name])

        """ set the CS and FS component if precompute sensing is True """
        if precompute_sensing:
            if sensing_mode == 'vector':
                model.get_layer('sensing').set_weights([W[0], W[1], W[2]])
            else:
                if separate_decoder:
                    model.get_layer('sensing').set_weights([W1, W2, W3, W1.T, W2.T, W3.T])
                else:
                    model.get_layer('sensing').set_weights([W1, W2, W3])
    
        current_weights = model.get_weights()
        optimal_weights = model.get_weights() 
        train_acc_list = []
        val_acc_list = []
        test_acc_list = []
        
        print('pre training evaluation')
        train_p = model.evaluate_generator(train_gen_fix, train_steps_fix)
        val_p = model.evaluate_generator(val_gen, val_steps)
        test_p = model.evaluate_generator(test_gen, test_steps)
        
        train_acc_list.append(train_p[metrics.index('acc')])
        val_acc_list.append(val_p[metrics.index('acc')])
        test_acc_list.append(test_p[metrics.index('acc')])
        
        validation_measure = val_acc_list[0]
        
        history = {'train_acc':[], 'train_loss':[], 'val_acc':[], 'val_loss':[]}
        last_index = 0
    else:
        fid = open(log_file, 'rb')
        log_data = pickle.load(fid)
        fid.close()
        
        current_weights = log_data['current_weights']
        optimal_weights = log_data['optimal_weights']
        validation_measure = log_data['validation_measure']
        train_acc_list = log_data['train_acc_list']
        val_acc_list = log_data['val_acc_list']
        test_acc_list = log_data['test_acc_list']
        history = log_data['history']
        last_index = log_data['last_index']
        
    learning_rates = []
    for lr, sc in zip(LR, Epoch):
        learning_rates += [lr,]*sc
        
    print('training')
    for epoch_index in tqdm(range(last_index, len(learning_rates))):
        if epoch_index == 0 or (epoch_index > 0 and learning_rates[epoch_index] != learning_rates[epoch_index-1]):
            model.compile('adam', 'categorical_crossentropy', ['acc', ])
            model.optimizer.lr = learning_rates[epoch_index]
            model.set_weights(current_weights)

        h = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=1, verbose=0)

        val_p = model.evaluate_generator(val_gen, val_steps)
        
        if val_p[metrics.index('acc')] > validation_measure:
            validation_measure = val_p[metrics.index('acc')]
            optimal_weights = model.get_weights()
            
        history['train_acc'] += h.history['acc']
        history['train_loss'] += h.history['loss']
        history['val_acc'].append(val_p[metrics.index('acc')])
        history['val_loss'].append(val_p[metrics.index('loss')])
        current_weights = model.get_weights()
        
        if epoch_index % Conf.LOG_INTERVAL == 0:
            log_data = {'current_weights': current_weights,
                        'optimal_weights': optimal_weights,
                        'validation_measure': validation_measure,
                        'history': history,
                        'train_acc_list': train_acc_list,
                        'val_acc_list': val_acc_list,
                        'test_acc_list': test_acc_list,
                        'last_index': epoch_index +1}
            
            fid = open(log_file, 'wb')
            pickle.dump(log_data, fid)
            fid.close()
        
    model.set_weights(optimal_weights)
    
    train_p = model.evaluate_generator(train_gen_fix, train_steps_fix)
    val_p = model.evaluate_generator(val_gen, val_steps)
    test_p = model.evaluate_generator(test_gen, test_steps)
    
    train_acc_list.append(train_p[metrics.index('acc')])
    val_acc_list.append(val_p[metrics.index('acc')])
    test_acc_list.append(test_p[metrics.index('acc')])

    train_acc = train_acc_list[-1]
    val_acc = val_acc_list[-1]
    test_acc = test_acc_list[-1]
    train_categorical_crossentropy = train_p[metrics.index('loss')]
    val_categorical_crossentropy = val_p[metrics.index('loss')]
    test_categorical_crossentropy = test_p[metrics.index('loss')]
    
    
    weights = {}
    for layer in model.layers:
        weights[layer.name] = layer.get_weights()
        
    results = {'history': history,
               'train_acc_list': train_acc_list,
               'val_acc_list': val_acc_list,
               'test_acc_list': test_acc_list,
               'train_acc': train_acc,
               'val_acc': val_acc,
               'test_acc': test_acc,
               'train_categorical_crossentropy': train_categorical_crossentropy,
               'val_categorical_crossentropy': val_categorical_crossentropy,
               'test_categorical_crossentropy': test_categorical_crossentropy,
               'weights': weights}
    
    return results

    
    
