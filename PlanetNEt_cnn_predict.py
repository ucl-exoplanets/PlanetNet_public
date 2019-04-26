from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from matplotlib import cm
import matplotlib.gridspec as gridspec


import pylab as pl
from classes.PlanetNet_cnn import PlanetNet_cnn

tf.logging.set_verbosity(tf.logging.INFO)

####################################################
#NOTES
'''
working: 
model_path='./tmp/saturn_1010_ktest4/storm1010',
# model_path='./tmp/saturn_set2test/storm1010',

'''
####################################################



# train_data = np.load('vims_map_train.npy')
# train_data = np.transpose(np.asarray(train_data,dtype=np.float32))
# train_labels = np.load('vims_labels_train.npy')
# train_labels = np.asarray(train_labels,dtype=np.int32)
# eval_data = np.load('vims_map_test.npy')
# eval_data = np.transpose(np.asarray(eval_data,dtype=np.float32))
# eval_labels = np.load('vims_labels_test.npy')
# eval_labels = np.asarray(eval_labels,dtype=np.int32)

data_prefix = 'data/storm_1010/10-10_saturn' #GOOD
# data_prefix = 'data/storm_set2test/10-10_saturn_set2train_test'

# data_prefix = 'data/storm_set3/20-20_saturn'
# data_prefix = 'data/storm_pca0/20-20_saturn_pca'
# data_prefix = 'data/storm_giant/10-10_saturn_giant_s/torm'
# save_suffix = 'storm_1010_giant_ktest4'
# save_suffix = 'storm_ktest5_1' #GOOD
save_suffix = 'storm_test'
  

#
map_data = np.load('{}_map_full_resampled_maponly.npy'.format(data_prefix))
map_data = np.transpose(np.asarray(map_data,dtype=np.float32))
spec_data = np.load('{}_spec_full_resampled_maponly.npy'.format(data_prefix))
spec_data = np.transpose(np.asarray(spec_data,dtype=np.float32))
labels = np.load('{}_labels_full_resampled_maponly.npy'.format(data_prefix))


# map_data = np.load('{}_map_full_resampled_speconly.npy'.format(data_prefix))
# map_data = np.transpose(np.asarray(map_data,dtype=np.float32))
# spec_data = np.load('{}_spec_full_resampled_speconly.npy'.format(data_prefix))
# spec_data = np.transpose(np.asarray(spec_data,dtype=np.float32))
# labels = np.load('{}_labels_full_resampled_speconly.npy'.format(data_prefix))


# map_data = np.load('{}_map_full_resampled.npy'.format(data_prefix))
# map_data = np.transpose(np.asarray(map_data,dtype=np.float32))
# spec_data = np.load('{}_spec_full_resampled.npy'.format(data_prefix))
# spec_data = np.transpose(np.asarray(spec_data,dtype=np.float32))
# labels = np.load('{}_labels_full_resampled.npy'.format(data_prefix))

#
# map_data = np.load('{}_map_full.npy'.format(data_prefix))
# map_data = np.transpose(np.asarray(map_data,dtype=np.float32))
# spec_data = np.load('{}_spec_full.npy'.format(data_prefix))
# spec_data = np.transpose(np.asarray(spec_data,dtype=np.float32))
# labels = np.load('{}_labels_full.npy'.format(data_prefix))



# train_labels = np.asarray(train_labels,dtype=np.int32)
 
# #initialise PlanetNet
# robcnn = PlanetNet_cnn(Nx=12,Ny=12,
#                     model_path='./tmp/test.ckpt',
#                     model_type='spectral')

#getting predictions
# predictions = robcnn.predict_single(vims_data)
# predictions = robcnn.predict_single(spec_data)

#convert to one hot labels
Ntrain = len(labels)
Nmax = np.max(labels)+1

Nlabel = 5

print('Map data: ',np.shape(map_data))
print('Spec data: ',np.shape(spec_data))


#initialise PlanetNet
robcnn = PlanetNet_cnn(Nx=10,Ny=10,Ns=256,
                    Nlogit=Nlabel,
                    model_path='./tmp/saturn_1010_test2018_20percent/storm1010',
                    # model_path='./tmp/saturn_1010_ktest4/storm1010',
                    # model_path='./tmp/saturn_set2test/storm1010',
                    model_type='both')


#initialise PlanetNet
# robcnn = PlanetNet_cnn(Nx=20,Ny=20,Ns=256,
#                     Nlogit=5,
#                     model_path='./tmp/storm_2/storm2',
#                     model_type='spatial')
 
#getting predictions
predictions = robcnn.predict_both(map_data,spec_data)
# predictions = robcnn.predict_single(spec_data)
# predictions = robcnn.predict_single(map_data)

cnn_layers = robcnn.get_cnn_filters(plot=False)


#getting prediction classes
pred_labels_flat = predictions['classes']

pred_prob = np.asarray(predictions['probabilities']).reshape(64,64,5)

#     pred_prob_dict[i] = np.asarray(pred_prob[:,i]).reshape(64,64)

orig_labs = labels.reshape(64,64)
pred_labs = np.asarray(pred_labels_flat).reshape(64,64)


np.save('PlanetNet_labels_pred_{0}.npy'.format(save_suffix),pred_labs)
np.save('PlanetNet_labels_pred_prob_{0}.npy'.format(save_suffix),pred_prob)

cmap = cm.get_cmap('Set1')

# pl.figure()
# pl.imshow(orig_labs,cmap=cmap,vmin=0,vmax=9)
# pl.title('Original labels')
#
#
# pl.figure()
# pl.imshow(pred_labs,cmap=cmap,vmin=0,vmax=9)
# pl.title('Predicted labels')
#
#
# pl.figure()
# ax = pl.imshow(pred_prob[:,:,0],cmap=cm.get_cmap('viridis'))
# pl.colorbar(ax)
# pl.title('Probability Label: {}'.format(0))
#
# pl.figure()
# ax = pl.imshow(pred_prob[:,:,1],cmap=cm.get_cmap('viridis'))
# pl.colorbar(ax)
# pl.title('Probability Label: {}'.format(1))
#
#
# pl.figure()
# ax = pl.imshow(pred_prob[:,:,2],cmap=cm.get_cmap('viridis'))
# pl.colorbar(ax)
# pl.title('Probability Label: {}'.format(2))
#
# pl.figure()
# ax = pl.imshow(pred_prob[:,:,3],cmap=cm.get_cmap('viridis'))
# pl.colorbar(ax)
# pl.title('Probability Label: {}'.format(3))
#
# pl.figure()
# ax = pl.imshow(pred_prob[:,:,4],cmap=cm.get_cmap('viridis'))
# pl.colorbar(ax)
# pl.title('Probability Label: {}'.format(4))


#ADD prediction comparison plots for paper
pred_errors_flat = np.zeros_like(labels)
pred_errors_flat[labels==pred_labels_flat] = 1.0

pred_errors_sum = np.sum(pred_errors_flat)
pred_errors_frac = pred_errors_sum / len(pred_errors_flat)
print('Error fraction: ',pred_errors_frac)

pred_errors = np.reshape(pred_errors_flat,(64,64))
pred_errors = np.ma.masked_where(pred_errors > 0.9, pred_errors)


fig = pl.figure(figsize=(15,6))
gs = gridspec.GridSpec(1,3)
gs.update(wspace=0.025, hspace=0.25)

ax1 = pl.subplot(gs[0])
ax2 = pl.subplot(gs[1])
ax3 = pl.subplot(gs[2])

ax1.set_yticks([])
ax1.set_xticks([])
ax2.set_yticks([])
ax2.set_xticks([])
ax3.set_yticks([])
ax3.set_xticks([])

# pl.figure()
ax1.set_title('Original labels',fontsize=20)
ax1.imshow(orig_labs,cmap=cmap,vmin=0,vmax=9)


ax2.set_title('Predicted labels',fontsize=20)
ax2.imshow(pred_labs,cmap=cmap,vmin=0,vmax=9)


ax3.set_title('Prediction errors',fontsize=20)
# ax3.imshow(orig_labs,cmap=cmap,vmin=0,vmax=9)
ax3.imshow(pred_errors,alpha=1.0,cmap=cm.get_cmap('binary_r'),vmin=0,vmax=1)

    
pl.show()


