#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pylab as pl
import tensorflow as tf
from tensorflow import *

import time 
import itertools as iter

# from tensorflow.contrib import learn
# from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
# from tensorflow.contrib.learn.python import SKCompat

from classes.PlanetNet_cnn import PlanetNet_cnn

tf.logging.set_verbosity(tf.logging.INFO)

####################################################

# data_prefix = 'enceladus'
# data_path = './tmp/enceladus'

# data_prefix = 'saturn'
# data_path = './tmp/saturn'
# 
# map_data = np.load('{}_map_train.npy'.format(data_prefix))
# map_data = np.transpose(np.asarray(map_data,dtype=np.float32))
# 
# map_test = np.load('{}_map_test.npy'.format(data_prefix))
# map_test = np.transpose(np.asarray(map_test,dtype=np.float32))
#  
# spec_test = np.load('{}_spec_test.npy'.format(data_prefix))
# spec_test = np.transpose(np.asarray(spec_test,dtype=np.float32))
# 
# spec_data = np.load('{}_spec_train.npy'.format(data_prefix))
# spec_data = np.transpose(np.asarray(spec_data,dtype=np.float32))
#  
# train_labels = np.load('{}_labels_train.npy'.format(data_prefix))
# train_labels = np.asarray(train_labels,dtype=np.int32)
#  
# test_labels = np.load('{}_labels_test.npy'.format(data_prefix))
# test_labels = np.asarray(test_labels,dtype=np.int32)
#  
# 
# Nmax = np.max(train_labels)+1


Nx = [8,12,16,20]
# Ny = [8,12,16,20]
Ns = 256

Ncov1 = [10,20,30,40]
Ncov2 = [10,20,30,40]

Ksize1 = [4,6,8]
Ksize2 = [4,6,8]

Ncov1_1d = [5,10,20]
Ncov2_1d = [20,30,40]

Ksize1_1d = [2,4,6,8]
Ksize2_1d = [2,4,6,8]

Ndense = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300]

varprod = iter.product(Nx,Ncov1,Ncov2,Ksize1,Ksize2)

total_count = 0
for par in varprod:
    print(total_count, par)
    total_count += 1
    

exit()
varprod = iter.product(Nx,Ncov1,Ncov2,Ksize1,Ksize2)


countlist = []
trainlist = []
testlist  = []
entropylist = []

c = 0
for params in varprod:
    print('ITERATION: {0}/{1}'.format(c,total_count))
    
    #paths 
    data_path='data'
    model_path = os.path.join('save/models',str(c))
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
        
    #loading data
    data_prefix = '{0}-{0}_saturn'.format(params[0])
#         map_data = np.load(os.path.join(data_path, '{}_map_train.npy'.format(data_prefix)))
#         map_data = np.transpose(np.asarray(map_data,dtype=np.float32))
#         
#         map_test = np.load(os.path.join(data_path,'{}_map_test.npy'.format(data_prefix)))
#         map_test = np.transpose(np.asarray(map_test,dtype=np.float32))
     
    spec_data = np.load(os.path.join(data_path, '{}_spec_train.npy'.format(data_prefix)))
    spec_data = np.transpose(np.asarray(spec_data,dtype=np.float32))
     
    spec_test = np.load(os.path.join(data_path,'{}_spec_test.npy'.format(data_prefix)))
    spec_test = np.transpose(np.asarray(spec_test,dtype=np.float32))
     
    train_labels = np.load(os.path.join(data_path,'{}_labels_train.npy'.format(data_prefix)))
    train_labels = np.asarray(train_labels,dtype=np.int32)
     
    test_labels = np.load(os.path.join(data_path,'{}_labels_test.npy'.format(data_prefix)))
    test_labels = np.asarray(test_labels,dtype=np.int32)
    
    Nmax = np.max(train_labels)+1
    
    with tf.Graph().as_default():
        #initialising PlanetNet
        robcnn = PlanetNet_cnn(Nx=params[0],
                            Ny=params[0],
                            Ns=256,
                            Nlogit=Nmax,
                            model_path=model_path+'/saturn',
                            model_type='spectral')
        
        #setting convolutional layers
        robcnn.Nconv1 = np.int32(params[1])
        robcnn.Nconv2 = np.int32(params[2])
        robcnn.Ksize1 = np.int32(params[3])
        robcnn.Ksize2 = np.int32(params[4])
        robcnn.Ndense = 1000
        robcnn.reset_var_dict()
        
        #running model
        onehot = robcnn.convert_to_onehot(train_labels)
        robcnn.train_single(spec_data,
                         onehot,
                         resume=False,
                         resume_transfer=False,
                         train_iter=14000,
                         run_test=True,
                         test_data=spec_test,
                         test_labels=test_labels)
        
        print('test acc: ',robcnn.test_accuracy)
        print('train acc: ',robcnn.train_accuracy)
        print('cross entr: ',robcnn.cross_entropy_loss)
        
        out = np.zeros((3,1))
        out[0] = robcnn.test_accuracy
        out[1] = robcnn.train_accuracy
        out[2] = robcnn.cross_entropy_loss
        np.savetxt(os.path.join(model_path,'test_stats.txt'),out)
    
        countlist.append(c)
        trainlist.append(robcnn.train_accuracy)
        testlist.append(robcnn.test_accuracy)
        entropylist.append(robcnn.cross_entropy_loss)
    
        c += 1
        del robcnn


# print(countlist)
# print(trainlist)
# print(testlist)
# print(entropylist)

np.savetxt('countlist.dat',np.asarray(countlist))
np.savetxt('train_accuracy.dat',np.asarray(trainlist))
np.savetxt('test_accuracy.dat',np.asarray(testlist))
np.savetxt('cross_entropy.dat',np.asarray(entropylist))



#     break

exit()

#initialise PlanetNet
robcnn = PlanetNet_cnn(Nx=12,Ny=12,Ns=256,
                    Nlogit=Nmax,
                    model_path=data_path+'/test',
                    transfer_path = './tmp/transfer',
                    model_type='spatial')

onehot = robcnn.convert_to_onehot(train_labels)

time1 = time.clock()


robcnn.Nconv1 = np.int32(10)
robcnn.Nconv2 = np.int32(20)
robcnn.reset_var_dict()


robcnn.train_single(map_data,
                     onehot,
                     resume=False,
                     resume_transfer=False,
                     train_iter=400,
                     run_test=True,
                     test_data=map_test,
                     test_labels=test_labels)




print('test acc: ',robcnn.test_accuracy)
print('train acc: ',robcnn.train_accuracy)


# robcnn.train_both(map_data,
#                   spec_data,
#                   onehot,
#                   resume=False,
#                   resume_transfer=False,
#                   train_iter=20000,
#                   run_test=True,
#                   map_test=map_test,
#                   spec_test=spec_test,
#                   test_labels=test_labels)




time2 = time.clock()

print('RUNNING TIME: ',time2-time1)

robcnn.save_training_log(os.path.join(data_path,'training_log.dat'))

pl.figure()
pl.plot(robcnn.entropy_log,label='Cross entropy')
pl.plot(1.0-np.asarray(robcnn.test_acc_log),label='Missed tests frac')
pl.legend()
pl.savefig(os.path.join(data_path,'training_log.pdf'))
