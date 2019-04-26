import numpy as np
import tensorflow as tf
import os

# import config
import logging
try:
    from ConfigParser import SafeConfigParser # python 2
except:
    from configparser import SafeConfigParser # python 3

from yadlt.models.rbm_models import dbn
from yadlt.utils import datasets, utilities

from classes.parameters import *

class PlanetNet_core(object):
    def __init__(self,opts):
        '''
        PlanetNet_core defines all the core functionality common to deep autoencoders and deep belief networks. 
        Likely this will be extended later on. It is heavily reliant on the yadlt setup and mostly constitutes 
        a differnt kind of wrapper to the yadlt provided. 
        '''
        
        #setting up logger 
        # define a Handler which writes INFO messages or higher to the sys.stderr
        self.console = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.console.setFormatter(formatter)
        logging.getLogger().addHandler(self.console)
        logging.info('Log started.')
        
#         #loading paramters 
#         self.params = parameters(parfile=opts.param_filename)
        
    
    def load_params(self,opts):
        return parameters(parfile=opts.param_filename)
        
        
    def load_data(self):
        ble =1 
        
    def do_pretrain(self,trX,trY):
        '''wrapper to pre-retain the model'''
        logging.info('Pretraining model...')
        self.model.pretrain(trX,trY)
    
    def do_finetune(self,trX, trY, vlX, vlY, restore_previous_model=False):
        '''wrapper to fine-tune the model'''
        logging.info('Fine-tuning model...')
        self.model.fit(trX, trY, vlX, vlY, restore_previous_model=restore_previous_model) 
    
    def get_test_accuracy(self,teX,teY):
        logging.info('Test set accuracy: {}'.format(self.model.compute_accuracy(teX, teY)))
        return self.model.compute_accuracy(teX, teY)
    
    def save_predictions(self, PATH,teX):
        '''saving predictions for teX as .npy to PATH'''
        logging.info('Saving predictions to: {}'.format(PATH))
        np.save(PATH,self.model.predict(teX))
        
    def get_predictions(self,teX):
        ''' returns predictions for teX'''
        logging.info('Returning predictions')
        return self.model.predict(teX)
    
    def save_layer_output(self,type, PATH):
        '''saves the output of each layer to .npy files in PATH'''
        logging.info('Saving model layers to: {}'.format(PATH))
        trout = self.model.get_layers_output(type)
        for i, o in enumerate(trout):
            np.save(PATH + '-layer-' + str(i + 1) , o)
        
    def get_layer_output(self,type):
        '''returns Nlayer deep array of output'''
        trout = self.model.get_layers_output(type)
        print(len(trout)) #@todo actually write 
        
        
        
        
    

    
    
    