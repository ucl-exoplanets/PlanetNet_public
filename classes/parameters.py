try:
    from ConfigParser import SafeConfigParser # python 2
except:
    from configparser import SafeConfigParser # python 3
import numpy as np
import logging
import os
import inspect
import subprocess


class parameters(object):
    def __init__(self, parfile='default.par'):
        
        
        #config file parser
        if parfile:
            self.parser = SafeConfigParser()
            try:
                self.parser.readfp(open(parfile, 'rb')) # python 2
            except:
                self.parser.read_file(open(parfile, 'rt', encoding='latin1')) # python 3

        self.parfile = parfile
        self.default_parser = SafeConfigParser()

        
        self.default_parser.sections()
        
        # section General
        self.gen_model              = self.getpar('General','model')
        self.gen_pretrain           = self.getpar('General','do_pretrain', 'bool')
        self.gen_seed               = self.getpar('General', 'rnd_seed', 'int')
        self.gen_momentum           = self.getpar('General', 'momentum', 'float')
        self.gen_verbose            = self.getpar('General','verbose','int')
        
        self.in_data                = self.getpar('Input', 'data') 
        self.in_restore_model       = self.getpar('Input', 'restore_previous_model', 'bool')
        self.in_prev_model          = self.getpar('Input','previous_model')
        
        self.out_save_predict       = self.getpar('Output','save_predictions','bool')
        self.out_save_layers        = self.getpar('Output','save_layers','bool')
        self.out_dir                = self.getpar('Output','out_dir')
        
        self.rbm_layers             = self.getpar('RBM', 'layers','list-int')
        self.rbm_gauss_vis          = self.getpar('RBM', 'gauss_visible','bool')
        self.rbm_stddev             = self.getpar('RBM', 'stddev','float')
        self.rbm_learn_rate         = self.getpar('RBM', 'learning_rate','float')
        self.rbm_epochs             = self.getpar('RBM', 'epochs','int')
        self.rbm_batch_size         = self.getpar('RBM', 'batch_size','int')
        self.rbm_gibbs_k            = self.getpar('RBM', 'gibbs_k','int')
        
        self.fine_act_func          = self.getpar('Finetuning', 'act_func')
        self.fine_learn_rate        = self.getpar('Finetuning', 'learning_rate','float')
        self.fine_momentum          = self.getpar('Finetuning', 'momentum','float')
        self.fine_epochs            = self.getpar('Finetuning', 'epochs','int')
        self.fine_batch_size        = self.getpar('Finetuning', 'batch_size','int')
        self.fine_minimiser         = self.getpar('Finetuning', 'minimiser')
        self.fine_loss_func         = self.getpar('Finetuning', 'loss_func')
        self.fine_dropout           = self.getpar('Finetuning', 'dropout','float')
    
    
    def getpar(self, sec, par, type=None):

        # get parameter from user defined parser. If parameter is not found there, load the default parameter
        # the default parameter file parser is self.default_parser, defined in init

        try:

            if type == None:
                try:
                    return self.parser.get(sec, par)
                except:
                    return self.default_parser.get(sec, par)
            elif type == 'float':
                try:
                    return self.parser.getfloat(sec, par)
                except:
                    return self.default_parser.getfloat(sec, par)

            elif type == 'bool':
                try:
                    return self.parser.getboolean(sec, par)
                except:
                    return self.default_parser.getboolean(sec, par)
            elif type == 'int':
                try:
                    return self.parser.getint(sec, par)
                except:
                    return self.default_parser.getint(sec, par)
            elif type == 'list-str':
                try:
                    l = self.parser.get(sec,par).split(',')
                    return [str(m).strip() for m in l]
                except:
                    l = self.default_parser.get(sec,par).split(',')
                    return [str(m).strip() for m in l]
            elif type == 'list-float':
                try:
                    l = self.parser.get(sec,par).split(',')
                    return [float(m) for m in l]
                except:
                    l = self.default_parser.get(sec,par).split(',')
                    return [float(m) for m in l]
            elif type == 'list-int':
                try:
                    l = self.parser.get(sec,par).split(',')
                    return [int(m) for m in l]
                except:
                    l = self.default_parser.get(sec,par).split(',')
                    return [int(m) for m in l]
            else:
                logging.error('Cannot set parameter %s in section %s. Parameter type %s not recognized. Set to None' (par, sec, type))
                return None
        except:
            logging.error('Cannot set parameter %s in section %s. Set to None' % (par, sec))
            return None

    def params_to_dict(self):

        # covert param variables to dictionary
        pr = {}
        for name in dir(self):
            value = getattr(self, name)
            if not name.startswith('__') and not inspect.ismethod(value) and \
                            name != 'parser' and name != 'default_parser' and name != 'console':
                pr[name] = value
        return pr