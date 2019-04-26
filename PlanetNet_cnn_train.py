'''
Training routine to train PlanetNet on data.

Set data_prefix and

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pylab as pl
import tensorflow as tf

import time

from classes.PlanetNet_cnn import PlanetNet_cnn

tf.logging.set_verbosity(tf.logging.INFO)

####################################################

data_prefix = 'data/storm_2020/20-20_saturn'
model_path = './tmp/saturn_2020_testmodel'


#loading spatial training data
map_data = np.load('{}_map_train.npy'.format(data_prefix))
map_data = np.transpose(np.asarray(map_data,dtype=np.float32))

map_test = np.load('{}_map_test.npy'.format(data_prefix))
map_test = np.transpose(np.asarray(map_test,dtype=np.float32))

#loading spectral training data
spec_test = np.load('{}_spec_test.npy'.format(data_prefix))
spec_test = np.transpose(np.asarray(spec_test,dtype=np.float32))

spec_data = np.load('{}_spec_train.npy'.format(data_prefix))
spec_data = np.transpose(np.asarray(spec_data,dtype=np.float32))

#loading labels
train_labels = np.load('{}_labels_train.npy'.format(data_prefix))
train_labels = np.asarray(train_labels,dtype=np.int32)

#loading test data
test_labels = np.load('{}_labels_test.npy'.format(data_prefix))
test_labels = np.asarray(test_labels,dtype=np.int32)


#convert to one hot labels
Nmax = np.max(train_labels)+1


#initialise PlanetNet
robcnn = PlanetNet_cnn(Nx=10,Ny=10,Ns=256, #dimensions of data to train on
                    Nlogit=Nmax,
                    model_path=model_path+'/storm1010',
                    transfer_path = './tmp/transfer',
                    model_type='both', #train on both spatial and spectral data
                    dropout=0.4)


time1 = time.clock()

#training on either spatial or spectral only
# robcnn.train_single(map_data,
#                          onehot,
#                          resume=False,
#                          resume_transfer=False,
#                          train_iter=20000,
#                          run_test=True,
#                          test_data=map_test,
#                          test_labels=test_labels)


#training on both spatial and spectral data
robcnn.train_both(map_data,
                  spec_data,
                  train_labels,
                  resume=False,
                  resume_transfer=False,
                  train_iter=40000,
                  run_test=True,
                  map_test=map_test,
                  spec_test=spec_test,
                  test_labels=test_labels)

time2 = time.clock()

print('RUNNING TIME: ',time2-time1)

robcnn.save_training_log(os.path.join(model_path,'training_log.dat'))

pl.figure()
pl.plot(robcnn.entropy_log,label='Cross entropy')
pl.title('Cross Entropy')
pl.legend()
pl.savefig(os.path.join(model_path,'training_log_entropy.pdf'))

pl.figure()
pl.plot(np.asarray(robcnn.test_acc_log),label='Tests frac')
pl.plot(np.asarray(robcnn.data_acc_log),label='Data frac')
pl.title('Test fraction')
pl.legend()
pl.savefig(os.path.join(model_path,'training_log_accuracy.pdf'))
