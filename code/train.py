#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Dat Tran
@Email: viebboy@gmail.com, dat.tranthanh@tut.fi, thanh.tran@tuni.fi
"""

import exp_configurations as Conf
import Utility
import sys, getopt
import pickle
import Runners
import os


def save_result(output, path):
    fid = open(path, 'wb')
    pickle.dump(output, fid)
    fid.close()

def main(argv):

    try:
      opts, args = getopt.getopt(argv,"h", ['index=' ])
    
    except getopt.GetoptError:
        sys.exit(2)
    
    for opt, arg in opts:             
        if opt == '--index':
            index = int(arg)
    
    baseline_configurations = Utility.create_configuration([Conf.baseline_var_names], [Conf.conf_baseline])
    experiment_configurations = Utility.create_configuration([Conf.hyperparameters]*4,
                                                   [Conf.conf_vector,
                                                    Conf.conf_tensor_standard,
                                                    # Conf.conf_tensor_linearity,
                                                    Conf.conf_tensor_resolution, 
                                                    Conf.conf_tensor_initialization])
    
    nb_configuration = len(baseline_configurations) + len(experiment_configurations)
    print('Total number of configurations: %d' % nb_configuration)

    if index >= nb_configuration:
        return
    if index < len(baseline_configurations):
        output = Runners.train_baseline(baseline_configurations[index])
        filename = '_'.join([str(v) for v in baseline_configurations[index]]) + '.pickle'
        filename = os.path.join(Conf.BASELINE_DIR, filename)
        
    else:
        output = Runners.train_sensing(experiment_configurations[index-len(baseline_configurations)])
        filename = '_'.join([str(v) for v in experiment_configurations[index-len(baseline_configurations)]]) + '.pickle'
        filename = os.path.join(Conf.OUTPUT_DIR, filename)
        
    save_result(output, filename)
    
if __name__ == "__main__":
    main(sys.argv[1:])
