#!/usr/bin/python
import os
import lasagne
from lasagne.layers import InputLayer, DenseLayer, batch_norm, ConcatLayer, DropoutLayer, BatchNormLayer, Conv2DLayer, DimshuffleLayer, MaxPool2DLayer, ReshapeLayer, NonlinearityLayer,GRULayer
import time
import theano
import theano.tensor as T
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import sys
import subprocess
import scipy.io
from fuel.datasets import IndexableDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from lasagne.nonlinearities import TemperatureSoftmax
from lasagne.nonlinearities import rectify
from collections import OrderedDict
import h5py
from fuel.datasets import H5PYDataset
from lasagne.regularization import regularize_layer_params, l1
from lasagne.init import GlorotUniform,HeNormal

def save(network, wts_path): 
  print('Saving Model ...')
  np.savez(wts_path, *lasagne.layers.get_all_param_values(network))

##project path - make a new one each time
#one folder for the weigths
project_path='/misc/scratch03/reco/bhattaga/data/i-vectors/dnn_projects/MSCOCO/convAE-transposed2/'
wts_path = os.path.join(project_path,'weights')
epoch_path = os.path.join(project_path,'epoch_weights')

logfile = os.path.join(project_path,'uttRNN-train.log')

if os.path.exists(project_path):
  print('Project folder exits. Deleting...')
  command00 = "rm -r" +" "+ project_path
  process0 = subprocess.check_call(command00.split())
      
  command0 = "mkdir -p" +" "+ project_path
  process = subprocess.check_call(command0.split())
  command1 = "mkdir -p" +" "+ wts_path
  process1 = subprocess.check_call(command1.split())
  command2 = "mkdir -p" +" "+ epoch_path
  process2 = subprocess.check_call(command2.split())
else:
  print('Creating Project folder')
  command0 = "mkdir -p" +" "+ project_path
  process = subprocess.check_call(command0.split())
  command1 = "mkdir -p" +" "+ wts_path
  process1 = subprocess.check_call(command1.split())
  command2 = "mkdir -p" +" "+ epoch_path
  process2 = subprocess.check_call(command2.split())


X = T.tensor4(name='features',dtype='float32')
Targets = T.tensor4(name='targets',dtype='float32')

input_var = X.transpose((0,3,1,2))

encoder_size=300
num_units=64

network={}

print("Building network ...")
network['input'] = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),input_var=input_var)


network['conv1'] = lasagne.layers.Conv2DLayer(network['input'], num_filters=num_units, filter_size=(5, 5),W=lasagne.init.GlorotUniform())
network['pool1'] = lasagne.layers.MaxPool2DLayer(network['conv1'], pool_size=(2, 2))

network['conv2'] = lasagne.layers.Conv2DLayer(network['pool1'], num_filters=num_units, filter_size=(3, 3))
network['pool2'] = lasagne.layers.MaxPool2DLayer(network['conv2'], pool_size=(2, 2))

network['conv3'] = lasagne.layers.Conv2DLayer(network['pool2'], num_filters=num_units, filter_size=(3, 3))
network['pool3'] = lasagne.layers.MaxPool2DLayer(network['conv3'], pool_size=(2, 2))

network['conv4'] = lasagne.layers.Conv2DLayer(network['pool3'], num_filters=encoder_size, filter_size=(6, 6))

network['deconv1'] = lasagne.layers.TransposedConv2DLayer(network['conv4'], num_filters=num_units, filter_size=(6, 6), stride=(1, 1))
network['deconv2'] = lasagne.layers.TransposedConv2DLayer(network['deconv1'], num_filters=num_units, filter_size=(5, 5), stride=(2,2))
network['deconv3'] = lasagne.layers.TransposedConv2DLayer(network['deconv2'], num_filters=3, filter_size=(4, 4), stride=(2,2), nonlinearity=lambda x: x.clip(0., 1.))

network_output = lasagne.layers.get_output(network['deconv3'])
network_op = network_output.transpose((0,2,3,1))

total_cost = lasagne.objectives.squared_error(network_op, Targets) #+ L1_penalty*1e-7
mean_cost = total_cost.mean()


#Get parameters of both encoder and decoder
all_parameters = lasagne.layers.get_all_params(network['deconv3'], trainable=True)

print("Trainable Model Parameters")
print("-"*40)
for param in all_parameters:
    print(param, param.get_value().shape)
print("-"*40)

all_grads = T.grad(mean_cost, all_parameters)
Learning_rate = 0.001
learn_rate = theano.shared(np.array(Learning_rate, dtype='float32'))
lr_decay = np.array(0.1, dtype='float32')

updates = lasagne.updates.adam(all_grads, all_parameters, learn_rate)

train_func = theano.function([X, Targets], [mean_cost], updates=updates)


#Load MSCOCO Dataset
train_set = H5PYDataset('/misc/data15/reco/bhattgau/Rnn/code/Code_/mscoco/cocotrain/inpainting/mscoco-train.hdf5',which_sets=('train',))

epoch=0
num_epochs=50

print("Starting training...")
    # We iterate over epochs:
while 'true':
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0

    h1=train_set.open()

    scheme = ShuffledScheme(examples=train_set.num_examples, batch_size=512)

    train_stream = DataStream(dataset=train_set, iteration_scheme=scheme)

    start_time = time.time()

    for data in train_stream.get_epoch_iterator():

        t_data,t_target = data
        #t_data = np.reshape(t_data,(t_data.shape[0],3,64,64))
        ntars = np.empty((t_target.shape[0],32,32,3),dtype='float32')
        for ind,tt in enumerate(t_target):
          center = (int(np.floor(tt.shape[0] / 2.)), int(np.floor(tt.shape[1] / 2.)))
          tar = tt[center[0]-16:center[0]+16, center[1] - 16:center[1]+16]
          ntars[ind] = tar
        #t_target = np.reshape(t_target,(t_data.shape[0],3,64,64)) 

        terr = train_func(t_data, ntars)
        train_err += terr[0]
        train_batches += 1
    
    epoch+=1
    train_set.close(h1)
    
    print("Epoch {} of {} took {:.3f}s Learning Rate {}".format(
          epoch, num_epochs, time.time() - start_time, learn_rate.get_value()))
    print("training loss:{:.6f}".format((train_err / train_batches)))
     
    flog1 = open(logfile,'ab')
    flog1.write("Epoch {} of {} took {:.3f}s Learning rate {}\n".format(
        epoch, num_epochs, time.time() - start_time, learn_rate.get_value()))
    flog1.write("training loss:{:.6f}\n".format((train_err / train_batches)))
      
    flog1.write("\n")
    flog1.close()
    
    min_loss_network = network['deconv3'] 
      
    mname = 'ConvAE-weights-epoch-%d'%(epoch+1)
    spth = os.path.join(epoch_path,mname+'.npz')
    save(min_loss_network,spth)

    
    if epoch == num_epochs: 
      spth = os.path.join(wts_path,'ConvAE_minloss_nepoch.npz')
      save(min_loss_network, spth)
      break
