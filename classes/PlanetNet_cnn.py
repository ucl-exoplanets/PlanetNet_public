'''
Main part of the the PlanetNet code.

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import pylab as pl
import tensorflow as tf



tf.logging.set_verbosity(tf.logging.INFO)

class PlanetNet_cnn(object):
    def __init__(self,Nx=10,Ny=10,Ns=256,Nlogit=7,learning_rate=0.001,dropout=0.4,model_type='spatial',model_path='./tmp',transfer_path=None,logs_path= './tmp'):
        
        #model path
        self.model_path = model_path
        if transfer_path is None:
            self.trans_model_path = self.model_path
        else:
            self.trans_model_path = transfer_path
            
        #log path
        self.logs_path = logs_path
        
        #flag for SPATIAL, SPECTRAL or BOTH
        self.model_type = model_type
        
        #training batch size
        self.batch_size = 200
        
        #saving checkpoint every N iterations 
        self.checkpoint_iter = 1000
        
        ########### HYPERPARAMETERS FOR SPATIAL MODEL 
        #spatial dimensions of training/recognition image
        self.Nx = np.int32(Nx)
        self.Ny = np.int32(Ny)
        
        ####Convolutional layers
        #convolutional layers kernel size 
        self.Ksize1 = 4
        self.Ksize2 = 4 
        #convolutional layer 1 filter number 
        self.Nconv1 = 30 #15
        #convolutional layer 2 filter number 
        self.Nconv2 = 50 #40
        
        ####Pooling layers @TODO Careful here. Need to properly calculate tensor sizes if Npools and Nstrides change!!!
        #number of pooling layers
        self.Npools = 2 #not in use
        #pooling kernel size 
        self.Psize1 = 2
        self.Psize2 = 2
        #stride size 
        self.Nstrides = 2
        
        
        ########### HYPERPARAMETERS FOR SPECTRAL MODEL 
        #spectral dimensions of training/recognition image
        self.Ns = np.int32(Ns)
    
        ####Convolutional layers
        #convolutional layers kernel size 
        self.Ksize1_1d = 4
        #convolutional layer 1 filter number 
        self.Nconv1_1d = 30
        #convolutional layer 2 filter number 
        self.Nconv2_1d = 50
        
        ####Pooling layers @TODO Careful here. Need to properly calculate tensor sizes if Npools and Nstrides change!!!
        #number of pooling layers
        self.Npools_1d = 2 #not in use
        #pooling kernel size 
        self.Psize1_1d = 2
        #stride size 
        self.Nstrides_1d = 2
        
        
        
        ########### HYPERPARAMETERS FOR GLOBAL MODEL
        ####Dense, fully connected layer
        #number of neurons 
        self.Ndense = 1024
        
        #number of output classes
        self.Nlogit = np.int32(Nlogit)
        
        #learning rate
        self.learning_rate = np.float(learning_rate)
        
        #dropout rate
        self.dropout_rate = np.float(dropout)
        
        
        ########### 
        #converting to 32 bits and floats 
        self.Ksize1 = np.int32(self.Ksize1)
        self.Ksize2 = np.int32(self.Ksize2)
        self.Nconv1 = np.int32(self.Nconv1)
        self.Nconv2 = np.int32(self.Nconv2)
        self.Psize1 = np.int32(self.Psize1)
        self.Psize2 = np.int32(self.Psize2)
        
        self.Ksize1_1d = np.int32(self.Ksize1_1d)
        self.Nconv1_1d = np.int32(self.Nconv1_1d)
        self.Nconv2_1d = np.int32(self.Nconv2_1d)
        self.Psize1_1d = np.int32(self.Psize1_1d)
        self.Nstrides_1d = np.int32(self.Nstrides_1d) 
        
        self.Ndense= np.int32(self.Ndense)
        self.Nredu  = np.int32(self.Nstrides*self.Npools)
        self.Nstrides_t = (np.int32(self.Nstrides),np.int32(self.Nstrides))
        
        
        ###########
        #setting convolutional layer variable dictionaries
        self.spatial_var_dict = self.generate_variables_dict(type='spatial')
        self.spectral_var_dict = self.generate_variables_dict(type='spectral')
        
        #setting logging lists 
        self.entropy_log  =[]
        self.test_acc_log = []
        self.data_acc_log = []
        
    
    def reset_var_dict(self):
        #re-initialises var_dicts when parameters chanced
        self.spatial_var_dict = self.generate_variables_dict(type='spatial')
        self.spectral_var_dict = self.generate_variables_dict(type='spectral')
        
    
    def generate_variables_dict(self,type='spatial'):
        '''
        Routine generating weights and bias dictionaries for the convolutional 
        layers of the spatial/spectral models
        '''
        if type == 'spatial':
            Ksize1 = self.Ksize1
            Ksize2 = self.Ksize2
            Nconv1 = self.Nconv1
            Nconv2 = self.Nconv2
        elif type == 'spectral':
            Ksize1 = self.Ksize1_1d
            Ksize2 = self.Ksize1_1d
            Nconv1 = self.Nconv1_1d
            Nconv2 = self.Nconv2_1d
        
        
        var_dict = {"{}_conv1_w".format(type): tf.Variable(tf.random_normal([Ksize1,Ksize2,1,Nconv1]),name='{}_conv1_w'.format(type)),
                    "{}_conv1_b".format(type): tf.Variable(tf.zeros([Nconv1]),name='{}_conv1_b'.format(type)),
                    "{}_conv2_w".format(type): tf.Variable(tf.random_normal([Ksize1,Ksize2,Nconv1,Nconv2]),name='{}_conv2_w'.format(type)),
                    "{}_conv2_b".format(type): tf.Variable(tf.zeros([Nconv2]),name='{}_conv2_b'.format(type))
                    }
        
        return var_dict
        
        
    
    def spatial_model(self,features):
        '''
        Subroutine calculating the convolutional layers for spatial (2D) data
        '''
        
        # Input Layer
        # Reshape X to 4-D tensor: [batch_size, width, height, channels]
        Nx = self.Nx
        Ny = self.Ny
        input_layer = tf.reshape(features, [-1, Nx, Ny, 1])
    
        # Convolutional Layer #1
#         conv1_spatial = tf.layers.conv2d(
#             inputs=input_layer,
#             filters=self.Nconv1,
#             kernel_size=[self.Ksize1, self.Ksize2],
#             padding="same",
#             activation=tf.nn.relu,
#             name='spatial_conv1')

        conv1_spatial = tf.nn.conv2d(input_layer, self.spatial_var_dict["spatial_conv1_w"],
                             strides=[1, 1, 1, 1], padding='SAME')
        relu1_spatial = tf.nn.relu(conv1_spatial + self.spatial_var_dict["spatial_conv1_b"])
        
    
        # Pooling Layer #1
        pool1_spatial = tf.layers.max_pooling2d(inputs=relu1_spatial, 
                                        pool_size=[self.Psize1,self.Psize2], 
                                        strides=self.Nstrides_t,
                                        name='spatial_pool1')
    
        # Convolutional Layer #2
#         conv2_spatial = tf.layers.conv2d(
#             inputs=pool1_spatial,
#             filters=self.Nconv2,
#             kernel_size=[self.Ksize1, self.Ksize2],
#             padding="same",
#             activation=tf.nn.relu,
#             name='spatial_conv2')
        
        conv2_spatial = tf.nn.conv2d(pool1_spatial, self.spatial_var_dict["spatial_conv2_w"],
                             strides=[1, 1, 1, 1], padding='SAME')
        relu2_spatial = tf.nn.relu(conv2_spatial + self.spatial_var_dict["spatial_conv2_b"])
    
        # Pooling Layer #2
        pool2_spatial = tf.layers.max_pooling2d(inputs=relu2_spatial, 
                                        pool_size=[self.Psize1,self.Psize2], 
                                        strides=self.Nstrides_t,
                                        name='spatial_pool2')
    
        # Flatten tensor into a batch of vectors
        shape = pool2_spatial.get_shape().as_list()
        pool2_spatial_flat = tf.reshape(pool2_spatial, [-1, shape[1]*shape[2]*shape[3]])
#         pool2_flat = tf.reshape(pool2, [-1, np.int(Nx/4) * np.int(Ny/4) * self.Nconv2])
        
        return pool2_spatial_flat
        
    
    def spectral_model(self,features):
        '''
        Subroutine calculating the convolutional layers for spectral (1D) data
        '''
        # Input Layer
        # Reshape X to 4-D tensor: [batch_size, width, height, channels]

        input_layer = tf.reshape(features, [-1, self.Ns,1,1])
    
        # Convolutional Layer #1
        # Computes 32 features using a 5x5 filter with ReLU activation.
        # Padding is added to preserve width and height.
#         conv1_spectral = tf.layers.conv2d(
#             inputs=input_layer,
#             filters=self.Nconv1,
#             kernel_size=[self.Ksize1_1d,1],
#             padding="same",
#             activation=tf.nn.relu,
#             name='spectral_conv1')
        
        conv1_spectral = tf.nn.conv2d(input_layer, self.spectral_var_dict["spectral_conv1_w"],
                             strides=[1, 1, 1, 1], padding='SAME')
        relu1_spectral = tf.nn.relu(conv1_spectral + self.spectral_var_dict["spectral_conv1_b"])
            
        # Pooling Layer #1
        # First max pooling layer with a 2x2 filter and stride of 2
        pool1_spectral = tf.layers.max_pooling2d(inputs=relu1_spectral, 
                                        pool_size=[self.Psize1_1d,1], 
                                        strides=[self.Nstrides_1d,1],
                                        name='spectral_pool1')
    
        # Convolutional Layer #2
        # Computes 64 features using a 5x5 filter.
        # Padding is added to preserve width and height.
#         conv2_spectral = tf.layers.conv2d(
#             inputs=pool1_spectral,
#             filters=self.Nconv2,
#             kernel_size=[self.Ksize1_1d,1],
#             padding="same",
#             activation=tf.nn.relu,
#             name='spectral_conv2')
        
        conv2_spectral = tf.nn.conv2d(pool1_spectral, self.spectral_var_dict["spectral_conv2_w"],
                             strides=[1, 1, 1, 1], padding='SAME')
        relu2_spectral = tf.nn.relu(conv2_spectral + self.spectral_var_dict["spectral_conv2_b"])
    
        # Pooling Layer #2
        # Second max pooling layer with a 2x2 filter and stride of 2
        pool2_spectral = tf.layers.max_pooling2d(inputs=relu2_spectral, 
                                        pool_size=[self.Psize1_1d,1], 
                                        strides=[self.Nstrides_1d,1],
                                        name='spectral_pool2')
    
        # Flatten tensor into a batch of vectors     
        shape = pool2_spectral.get_shape().as_list()
#         print(shape)
        
        pool2_spectral_flat = tf.reshape(pool2_spectral, [-1, shape[1]*shape[2]*shape[3]])
        
#         shape = pool2.get_shape().as_list()
#         print(shape)
        
        return pool2_spectral_flat
        

    def model(self,input_map=None,input_spec=None, mode=None,train=False):
        """Model function for CNN."""
        # Input Layer
        # Reshape X to 4-D tensor: [batch_size, width, height, channels]
        
        if self.model_type == 'spatial':
            pool2_flat = self.spatial_model(input_map)
        elif self.model_type == 'spectral':
            pool2_flat = self.spectral_model(input_spec)
        elif self.model_type == 'both':
            pool2_flat_map = self.spatial_model(input_map)
            pool2_flat_spec = self.spectral_model(input_spec)
            pool2_flat = tf.concat([pool2_flat_map,pool2_flat_spec],1)

            
        # Dense Layer
        # Densely connected layer with 1024 neurons
        # Input Tensor Shape: [batch_size, 7 * 7 * 64]
        # Output Tensor Shape: [batch_size, 1024]
        dense = tf.layers.dense(inputs=pool2_flat, 
                                units=self.Ndense, 
                                activation=tf.nn.relu,
                                name='dense')
    
        # Add dropout operation; 0.6 probability that element will be kept
        dropout = tf.layers.dropout(
            inputs=dense, rate=self.dropout_rate, training=train)
    
        # Logits layer
        # Input Tensor Shape: [batch_size, 1024]
        # Output Tensor Shape: [batch_size, 10]
        logits = tf.layers.dense(inputs=dropout, units=self.Nlogit,name='logits')
        return logits

    def train_single(self,in_data,train_labels,train_iter=20000,run_test=False,test_data=None,test_labels=None,
                     resume=False,resume_transfer=False):

        # converting labels array to onehot labels
        in_labels = self.convert_to_onehot(train_labels)

        #start training statement
        print('------------------------')
        print('start training')
        print('')
        
        #getting data shape
        [Nsamples, Ndata] = np.shape(in_data)
        
        
        
        # tf Graph input
        x = tf.placeholder("float", [None, Ndata],name='data_in')
        y = tf.placeholder("int32", [None, self.Nlogit],name='labels')
    
        if run_test:
            [Nsamples_t, Ndata_t] = np.shape(test_data)
            # tf Graph test data input
            x_t = tf.placeholder("float", [None, Ndata_t],name='test_data')
            y_t = tf.placeholder("int32", [None, self.Nlogit],name='test_labels')
            
    
        #network model
        if self.model_type == 'spatial':
            y_ = self.model(input_map=x,train=True)
        elif self.model_type == 'spectral':
            y_ = self.model(input_spec=x,train=True)
            
#         y_ = self.model(input_map=x,train=True)
        
        #defining loss function
        cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
        
        #defining optimiser
        train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy)
        
        #initialising saver 
        saver_full = tf.train.Saver() #saves full model
        
        if self.model_type == 'spatial':
            saver_part = tf.train.Saver(self.spatial_var_dict) #saves partial model 
        elif self.model_type == 'spectral':    
            saver_part = tf.train.Saver(self.spectral_var_dict)

        # Initializing the variables
        init = tf.global_variables_initializer()
        
        #starting training session
        
        with tf.Session() as sess:
            # create log writer object
            train_writer = tf.summary.FileWriter(self.logs_path,sess.graph)
            
            sess.run(init) #initialise variables
            
            if resume:
                try:
                    saver_full.restore(sess, self.model_path)
                    print("Model restored from file: %s" % self.model_path)
                except:
                    print('Model not found, not resuming from file...')
                    
            elif resume_transfer: 
                try:
                    saver_part.restore(sess,self.trans_model_path+'_{0}'.format(self.model_type))
                    print("Transfer-leanring model restored from file: {0}_{1} ".format(self.trans_model_path,self.model_type))
                except:
                    print('Model not found, not resuming from file...')
            
            count = 0
            for i in range(train_iter):
#                 avg_cost = 0.
#                 total_batch = int(Ndata/batch_size)
#                 for i in range(total_batch):
                image_batch, label_batch = self.shuffle_data(input_data1=in_data,
                                                             input_labels1=in_labels,
                                                             next_batch=True,
                                                             batch_size=self.batch_size)
     
                    # Run optimization op (backprop) and cost op (to get loss value)
                summary,accuracy = sess.run([train_step,cross_entropy], feed_dict={x: image_batch, y: label_batch})

                self.cross_entropy_loss = accuracy
                #saving checkpoints
                if i % self.checkpoint_iter == 0:
                    # Append the step number to the checkpoint name:
                    saver_full.save(sess, self.model_path+'_checkpoint')
                    
                if count == self.batch_size:
                    if run_test:
                        pred_test = sess.run(y_, feed_dict={x: test_data}) 
                        pred_class = tf.argmax(input=pred_test, axis=1).eval()
                        pred_correct = np.equal(pred_class,test_labels)
                        self.pred_corr_frac = np.sum(pred_correct)/Nsamples_t                       
                        self.update_progress(i,train_iter,accuracy,self.pred_corr_frac) #progress bar   
                        self.log_training(accuracy, self.pred_corr_frac)                      
                    else:
                        self.update_progress(i,train_iter,accuracy) #progress bar 
                        train_writer.add_summary(summary) #writes tensorflow log
                        self.log_training(accuracy)   
                    count = 0
                count += 1
                # Compute average loss
#                     avg_cost += c / total_batch
#                         # Display logs per epoch step
#                 if epoch % display_step == 0:
#                     print("Epoch:", '%04d' % (epoch+1), "cost=", \
#                                 "{:.9f}".format(avg_cost))
            print("\n Training Finished!")
            
            # Test model
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            self.train_accuracy = sess.run(accuracy, feed_dict={x: in_data, y: in_labels})
            if run_test:
                self.test_accuracy = self.pred_corr_frac
            print("Training accuracy: {}".format(self.train_accuracy))
        
#             prediction=tf.argmax(y,1)
#             print("predictions", prediction.eval(feed_dict={x: in_data}, session=sess))
             
            save_path1 = saver_full.save(sess, self.model_path)
            save_path2 = saver_part.save(sess,self.trans_model_path+'_{}'.format(self.model_type))
            print("Model saved in file: %s" % save_path1)
            print("Partial model saved in file: {0}".format(save_path2))
            
    
    def train_both(self,map_data,spec_data,train_labels,run_test=False,map_test=None,spec_test=None,test_labels=None,
                   train_iter=20000,resume=False,resume_transfer=False):


        #converting labels array to onehot labels
        in_labels = self.convert_to_onehot(train_labels)

        #start training statement
        print('------------------------')
        print('start training')
        print('')
        
        #getting data shape
        [Nsamples, Nmap] = np.shape(map_data)
        [Nsamples, Nspec]= np.shape(spec_data)
        
#         print(np.shape(map_data))
#         print(np.shape(spec_data))
        
        # tf Graph input2
        xm = tf.placeholder("float", [None, Nmap])  #map parameters
        xs = tf.placeholder("float", [None, Nspec]) #spec parameters
        y = tf.placeholder("int32", [None, self.Nlogit]) #labels
        
        # test data sets 
        if run_test:
            [Nsamples_t, Nmap_t] = np.shape(map_test)
            [Nsamples_t, Nspec_t]= np.shape(spec_test)
            xm_t = tf.placeholder("float", [None, Nmap_t])  #map test parameters
            xs_t = tf.placeholder("float", [None, Nspec_t]) #spec parameters
            y_t = tf.placeholder("int32", [None, self.Nlogit]) #labels
    
        #network model
        y_ = self.model(input_map=xm,input_spec=xs,train=True)
        
        
        #defining loss function
        cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
        
        #defining optimiser
        train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy)
        
        #initialising saver 
        saver_full = tf.train.Saver() #saves the full model
        saver_spatial = tf.train.Saver(self.spatial_var_dict) #saves partial model     
        saver_spectral = tf.train.Saver(self.spectral_var_dict)
        
        # Initializing the variables
        init = tf.global_variables_initializer()

        #starting training session
        with tf.Session() as sess:
            # create log writer object
            train_writer = tf.summary.FileWriter(self.logs_path,sess.graph)
            
            sess.run(init) #initialise variables
            
            if resume:
                try:
                    saver_full.restore(sess, self.model_path)
                    print("Model restored from file: %s" % self.model_path)
                except:
                    print('Model not found, not resuming from file...')
            
            elif resume_transfer: 
                try:
                    saver_spatial.restore(sess,self.trans_model_path+'_spatial')
                    print(" Spatial transfer-leanring model restored from file: {0}".format(self.trans_model_path+'_spatial'))
                except:
                    print('Spatial transfer model not found, not resuming from file...')
                    
                try:
                    saver_spectral.restore(sess,self.trans_model_path+'_spectral')
                    print(" Spectral transfer-leanring model restored from file: {0}".format(self.trans_model_path+'_spectral'))
                except:
                    print('Spectral transfer model not found, not resuming from file...')
                    
                      
            count = 0
            for i in range(train_iter):
#                 avg_cost = 0.
#                 total_batch = int(Ndata/batch_size)
#                 for i in range(total_batch):
                image_batch, label_batch,spec_batch = self.shuffle_data(input_data1=map_data,
                                                             input_labels1=in_labels,
                                                             input_data2=spec_data,
                                                             next_batch=True,
                                                             batch_size=self.batch_size)
                
     
                    # Run optimization op (backprop) and cost op (to get loss value)
                summary,accuracy = sess.run([train_step,cross_entropy], feed_dict={xm: image_batch, xs:spec_batch, y: label_batch})
    
                self.cross_entropy_loss = accuracy
                #saving checkpoints
                if i % self.checkpoint_iter == 0:
                    # Append the step number to the checkpoint name:
                    saver_full.save(sess, self.model_path+'_checkpoint', global_step=i)
    
                #Printing current statistics per batch
                if count == self.batch_size:
#                     print('Iteration: {0}/{1}  Loss: {2}'.format(i,train_iter,accuracy))
                    if run_test and test_labels is not None:
                        pred_test = sess.run(y_, feed_dict={xm: map_test,xs: spec_test})
                        pred_data = sess.run(y_,feed_dict={xm: map_data,xs: spec_data})

                        pred_class_test = tf.argmax(input=pred_test, axis=1).eval()
                        pred_class_data = tf.argmax(input=pred_data, axis=1).eval()

                        pred_test_correct = np.equal(pred_class_test,test_labels)
                        pred_data_correct = np.equal(pred_class_data,train_labels)

                        pred_corr_test_frac = np.sum(pred_test_correct)/Nsamples_t
                        pred_corr_data_frac = np.sum(pred_data_correct)/Nsamples

                        self.log_training(accuracy, pred_corr_data_frac, pred_corr_test_frac)
                        self.update_progress(i,train_iter,accuracy,pred_corr_data_frac,pred_corr_test_frac) #progress bar

                    else:
                        self.update_progress(i,train_iter,accuracy) #progress bar 
                        train_writer.add_summary(summary) #writes tensorflow log
                        self.log_training(accuracy)  
                    count = 0
                count += 1
                # Compute average loss
#                     avg_cost += c / total_batch
#                         # Display logs per epoch step
#                 if epoch % display_step == 0:
#                     print("Epoch:", '%04d' % (epoch+1), "cost=", \
#                                 "{:.9f}".format(avg_cost))
            print("\n Training Finished!")
            
            # Test model
            
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            self.train_accuracy = sess.run(accuracy, feed_dict={xm: map_data, xs: spec_data, y: in_labels})
            if run_test:
                self.test_accuracy = pred_corr_test_frac
            print("Training accuracy: {}".format(self.train_accuracy))
        
#             prediction=tf.argmax(y,1)
#             print("predictions", prediction.eval(feed_dict={x: in_data}, session=sess))
             
            save_path1 = saver_full.save(sess, self.model_path)
            save_path2 = saver_spatial.save(sess,self.trans_model_path+'_spatial')
            save_path3 = saver_spectral.save(sess,self.trans_model_path+'_spectral')
            print("Model saved in file: %s" % save_path1)
            print("Partial model saved in files: {0} & {1}".format(save_path2,save_path3))
    

    def predict_single(self,data):
        #returns prediction from trained network
        
        #setting up variables
        [Nsamples, Ndata] = np.shape(data)
        x = tf.placeholder("float", [None, Ndata])
        
        #defining model
        y_ = self.model(x)
        
        #initialising stuff
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init) #initialise variables
            
            #restoring model from saved 
            saver.restore(sess, self.model_path)
            pred = sess.run(y_, feed_dict={x: data})
            
            predictions = {
            "classes": tf.argmax(
                input=pred, axis=1).eval(),
            "probabilities": tf.nn.softmax(
                pred, name="softmax_tensor").eval()
                           }
        
        self.predictions = predictions
        return predictions
    
    def predict_both(self,map_data,spec_data):
        #returns prediction from trained network
        
        #setting up variables
        [Nsamples, Nmap] = np.shape(map_data)
        [Nsamples, Nspec] = np.shape(spec_data)
        xm = tf.placeholder("float", [None, Nmap])
        xs = tf.placeholder("float", [None, Nspec])
        
        #defining model
        y_ = self.model(input_map=xm,input_spec=xs,train=False)
        
        #initialising stuff
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init) #initialise variables
            
            #restoring model from saved 
            saver.restore(sess, self.model_path)
            pred = sess.run(y_, feed_dict={xm: map_data,xs: spec_data})

            predictions = {
            "classes": tf.argmax(
                input=pred, axis=1).eval(),
            "probabilities": tf.nn.softmax(
                pred, name="softmax_tensor").eval()
                           }
        
        self.predictions = predictions
        return predictions
    


    def get_cnn_filters(self,plot=False):
        #function retrieving filters and plotting them and returning the arrays 
                
        #initialising stuff
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init) #initialise variables
            
            #restoring model from saved 
            saver.restore(sess, self.model_path)

            out = {}
            out['spatial'] = {} 
            out['spectral'] = {}
            out['spatial']['weights'] ={}
            out['spatial']['biases'] ={}
            out['spectral']['weights'] ={}
            out['spectral']['biases'] ={}
            
            out['spatial']['weights']['conv1'] = sess.graph.get_tensor_by_name("spatial_conv1_w:0").eval()
            out['spatial']['weights']['conv2'] = sess.graph.get_tensor_by_name("spatial_conv2_w:0").eval()
            out['spatial']['biases']['conv1'] = sess.graph.get_tensor_by_name("spatial_conv1_b:0").eval()
            out['spatial']['biases']['conv2'] = sess.graph.get_tensor_by_name("spatial_conv2_b:0").eval()
            
            out['spectral']['weights']['conv1'] = sess.graph.get_tensor_by_name("spectral_conv1_w:0").eval()
            out['spectral']['weights']['conv2'] = sess.graph.get_tensor_by_name("spectral_conv2_w:0").eval()
            out['spectral']['biases']['conv1'] = sess.graph.get_tensor_by_name("spectral_conv1_b:0").eval()
            out['spectral']['biases']['conv2'] = sess.graph.get_tensor_by_name("spectral_conv2_b:0").eval()

            
        if plot:    
            for mode in out.keys():
                for ntype in out[mode].keys():
                    if ntype is 'weights':
                        for layer in out[mode][ntype].keys():
                            data = out[mode][ntype][layer]
                            Ndata = np.shape(data)
                            Nsub5 = np.around(Ndata[-1]/5.0)
                            
                            pl.figure()
                            pl.suptitle('{0} {1} {2}'.format(mode,ntype,layer))
                            for i in range(Ndata[-1]):
                                ax = pl.subplot(5,Nsub5,i+1)
                                ax.imshow(data[:,:,0,i],interpolation='nearest',origin='upper')
                                ax.xaxis.set_major_locator(pl.NullLocator())
                                ax.yaxis.set_major_locator(pl.NullLocator())

        return out
        
        
    
    def shuffle_data(self,input_data1, input_labels1, input_data2=None,next_batch=True,batch_size=100):
        #shuffles input data. If two datasets are provided then they are shuffled in the same order
        
        [Nsamples, Ndata] = np.shape(input_data1)
        idx_shuffle = np.int32(np.arange(Nsamples))
        np.random.shuffle(idx_shuffle)
        if next_batch:
            idx_shuffle = np.int32(idx_shuffle[0:batch_size])
        
        out_data1 = input_data1[idx_shuffle,:]
        out_labels1 = input_labels1[idx_shuffle,:]

        if input_data2 is not None:
            out_data2 = input_data2[idx_shuffle,:]
            
        if input_data2 is not None:
            return out_data1,out_labels1, out_data2
        else:
            return out_data1,out_labels1


    def update_progress(self,iteration,total_num,accuracy,datascore=None,testscore=None):
        barLength = 20 # Modify this to change the length of the progress bar
        status = ""
        progress = iteration/total_num
        if progress == 0:
            status = "Starting...\r\n"
        elif progress >= 1:
            progress = 1
            status = "Done..."
        else:
            if testscore is None and datascore is None:
                status = "| Cross entropy: {0:.3f}".format(accuracy)
            elif testscore and datascore:
                status = "| Cross entropy: {0:.3f}  Data accuracy: {1:.3f}% Test accuracy: {2:.3f}%".format(accuracy,datascore * 100.0,testscore*100.0)
            elif datascore:
                status = "| Cross entropy: {0:.3f}  Data accuracy: {1:.3f}%".format(accuracy, datascore * 100.0)
        block = int(round(barLength*progress))
        text = "\rPercent: [{0}] {1}%  {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
        sys.stdout.write(text)
        sys.stdout.flush()
        
    def log_training(self,entropy,data_acc=None, test_acc=None):
        #logs entropy and test accuracy during training 
        
        self.entropy_log.append(entropy)
        if data_acc is not None:
            self.data_acc_log.append(data_acc)
        if test_acc is not None:
            self.test_acc_log.append(test_acc)
        
    def save_training_log(self,path):
        #converts training log lists to array and saves it 
        
        if len(self.entropy_log) == len(self.test_acc_log):
            self.entropy_log = np.asarray(self.entropy_log)
            self.data_acc_log = np.asarray(self.data_acc_log)
            self.test_acc_log = np.asarray(self.test_acc_log)
            self.log_out = np.transpose(np.vstack((self.entropy_log,self.data_acc_log,self.test_acc_log)))
        else:
            self.log_out = np.asarray(self.entropy_log)
    
        np.savetxt(path,self.log_out)
        
        
    def convert_to_onehot(self,train_labels):
        #converts array of labels e.g. [0,4,1,6,2] to onehot labels
        
        Ntrain = len(train_labels)
        Nmax = np.max(train_labels)+1
        
        onehot = np.zeros((Ntrain, Nmax))
        onehot[np.arange(Ntrain), train_labels] = 1
        
        return onehot
        