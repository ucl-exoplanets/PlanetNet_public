import numpy as np
import tensorflow as tf
import os
import sys
import argparse
import logging


# import config

from yadlt.models.rbm_models import dbn
from yadlt.utils import datasets, utilities

sys.path.append('./classes')
from PlanetNet_core import PlanetNet_core
from parameters import *

class PlanetNet_dbn(PlanetNet_core):
    def __init__(self,opts):
#         PlanetNet_core.__init__(opts)
        '''initialising DBN specifics'''
        
        #loading parameters
        self.params = self.load_params(opts)       
        
        #initialising DBN model 
        self.model = dbn.DeepBeliefNetwork(
        models_dir=self.params.out_dir, data_dir='data', summary_dir='summary',
        model_name=self.params.gen_model, do_pretrain=self.params.gen_do_pretrain,
        rbm_layers=self.params.rbm_layers, dataset='custom', main_dir=self.params.out_dir,
        finetune_act_func=self.params.fine_act_func, rbm_learning_rate=self.params.rbm_learn_rate,
        verbose=self.params.verbose, rbm_num_epochs=self.params.rbm_epochs, rbm_gibbs_k = self.params.rbm_gibbs_k,
        rbm_gauss_visible=self.params.rbm_gauss_vis, rbm_stddev=self.params.rbm_stddev,
        momentum=self.params.gen_momentum, rbm_batch_size=self.params.rbm_batch_size, finetune_learning_rate=self.params.fine_learn_rate,
        finetune_num_epochs=self.params.fine_epochs, finetune_batch_size=self.params.fine_batch_size,
        finetune_opt=self.params.fine_minimiser, finetune_loss_func=self.params.fine_loss_func,
        finetune_dropout=self.params.fine_dropout)


    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                    dest='param_filename',
                    default='default.par'
                   )
    
    options = parser.parse_args()
    rob_dbn = PlanetNet_dbn(options)
