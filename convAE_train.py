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
project_path='/misc/scratch03/reco/bhattaga/data/i-vectors/dnn_projects/MSCOCO/convAE-sigmoid/'
wts_path = os.path.join(project_path,'weights')
epoch_path = os.path.join(project_path,'epoch_weights')

logfile = os.path.join(project_path,'uttRNN-train.log')

if os.path.exists(project_path):
  print 'Project folder exits. Deleting...'
  command00 = "rm -r" +" "+ project_path
  process0 = subprocess.check_call(command00.split())
      
  command0 = "mkdir -p" +" "+ project_path
  process = subprocess.check_call(command0.split())
  command1 = "mkdir -p" +" "+ wts_path
  process1 = subprocess.check_call(command1.split())
  command2 = "mkdir -p" +" "+ epoch_path
  process2 = subprocess.check_call(command2.split())
else:
  print 'Creating Project folder'
  command0 = "mkdir -p" +" "+ project_path
  process = subprocess.check_call(command0.split())
  command1 = "mkdir -p" +" "+ wts_path
  process1 = subprocess.check_call(command1.split())
  command2 = "mkdir -p" +" "+ epoch_path
  process2 = subprocess.check_call(command2.split())

network={}

X = T.tensor4(name='features',dtype='float32')
Targets = T.tensor4(name='targets',dtype='float32')


print("Building network ...")
network['input'] = lasagne.layers.InputLayer(shape=(None,3, 64, 64),input_var=X)
batch,_,_,_ = network['input'].input_var.shape

#network['reshape'] = lasagne.layers.ReshapeLayer(network['input'],(batch,3,64,64))

network['conv1'] = lasagne.layers.Conv2DLayer(network['input'], num_filters=96,filter_size=(5, 5),pad='same',nonlinearity=leaky_rectify)
network['pool1'] = lasagne.layers.MaxPool2DLayer(network['conv1'], pool_size=(2, 2))

network['conv2'] = lasagne.layers.Conv2DLayer(network['pool1'], num_filters=64,filter_size=(5, 5),pad='same',nonlinearity=leaky_rectify)
network['pool2'] = lasagne.layers.MaxPool2DLayer(network['conv2'], pool_size=(2, 2))

network['conv3'] = lasagne.layers.Conv2DLayer(network['pool2'], num_filters=64,filter_size=(5, 5),pad='same',nonlinearity=leaky_rectify)
network['pool3'] = lasagne.layers.MaxPool2DLayer(network['conv3'], pool_size=(2, 2))

#network['fc1'] = lasagne.layers.DenseLayer(network['pool2'],num_units=200,nonlinearity=lasagne.nonlinearities.leaky_relu)
#network['fc2'] = lasagne.layers.DenseLayer(network['fc1'],num_units=64*16*16)
#network['reshape1'] = lasagne.layers.ReshapeLayer(network, (batch, 64, 8, 8))

# Deconv
network['deconv1'] = lasagne.layers.Conv2DLayer(network['pool3'], num_filters=64, filter_size=(5,5),pad='same',nonlinearity=leaky_rectify)
network['up1'] = lasagne.layers.Upscale2DLayer(network['deconv1'], 2)

network['deconv2'] = lasagne.layers.Conv2DLayer(network['up1'], num_filters=64, filter_size=(5,5),pad='same',nonlinearity=leaky_rectify)
network['up2'] = lasagne.layers.Upscale2DLayer(network['deconv2'], 2)

network['deconv3'] = lasagne.layers.Conv2DLayer(network['up2'], num_filters=96, filter_size=(5,5),pad='same', nonlinearity=leaky_rectify)
network['up3'] = lasagne.layers.Upscale2DLayer(network['deconv3'], 2)

network['deconv4'] = lasagne.layers.Conv2DLayer(network['up3'], num_filters=3, filter_size=(5,5),pad='same',nonlinearity=lasagne.nonlinearities.sigmoid)

network_output = lasagne.layers.get_output(network['deconv4'])

val_prediction = lasagne.layers.get_output(network['deconv4'], deterministic=True)

total_cost = lasagne.objectives.squared_error(network_output, Targets) #+ L1_penalty*1e-7
mean_cost = total_cost.mean()

#accuracy function
val_cost = lasagne.objectives.squared_error(val_prediction, Targets) #+ L1_penalty*1e-7
val_mcost = val_cost.mean()

#Get parameters of both encoder and decoder
all_parameters = lasagne.layers.get_all_params(network['deconv3'], trainable=True)

print("Trainable Model Parameters")
print("-"*40)
for param in all_parameters:
    print(param, param.get_value().shape)
print("-"*40)

all_grads = T.grad(mean_cost, all_parameters)
Learning_rate = 0.005
learn_rate = theano.shared(np.array(Learning_rate, dtype='float32'))
lr_decay = np.array(0.1, dtype='float32')

updates = lasagne.updates.adam(all_grads, all_parameters, learn_rate)

train_func = theano.function([X, Targets], [mean_cost], updates=updates)

val_func = theano.function([X, Targets], [val_mcost])
  
#function to return the softmax posterior
#posterior = theano.function([X,Masks], val_prediction)
#dvector = theano.function([X,Masks], hidden_output)

#Load MSCOCO Dataset
train_set = H5PYDataset('/misc/data15/reco/bhattgau/Rnn/code/Code_/mscoco/cocotrain/inpainting/mscoco-train.hdf5',which_sets=('train',),subset=slice(0,75000))
valid_set = H5PYDataset('/misc/data15/reco/bhattgau/Rnn/code/Code_/mscoco/cocotrain/inpainting/mscoco-train.hdf5',which_sets=('train',),subset=slice(75000,80262))

min_val_loss = np.inf
val_prev = 1000

patience=0#patience counter
val_counter=0 
epoch=0
num_epochs=50

print("Starting training...")
    # We iterate over epochs:
while 'true':
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0

    h1=train_set.open()
    h2=valid_set.open()

    scheme = ShuffledScheme(examples=train_set.num_examples, batch_size=512)
    scheme1 = SequentialScheme(examples=valid_set.num_examples, batch_size=128)

    train_stream = DataStream(dataset=train_set, iteration_scheme=scheme)
    valid_stream = DataStream(dataset=valid_set, iteration_scheme=scheme1)

    start_time = time.time()

    for data in train_stream.get_epoch_iterator():
        
        t_data,t_target = data

        t_data = np.reshape(t_data,(t_data.shape[0],3,64,64))
        t_target = np.reshape(t_target,(t_data.shape[0],3,64,64)) 
        
        terr = train_func(t_data, t_target)
        train_err += terr[0]
        train_batches += 1

    val_err = 0
    val_batches = 0

    for data in valid_stream.get_epoch_iterator():
        v_data,v_target = data    
        v_data = np.reshape(v_data,(v_data.shape[0],3,64,64))
        v_target = np.reshape(v_target,(v_data.shape[0],3,64,64)) 
        
        err = val_func(v_data, v_target)
        val_err += err[0]
        val_batches += 1
    
    epoch+=1
    train_set.close(h1)
    valid_set.close(h2)
    
    print("Epoch {} of {} took {:.3f}s Learning Rate {}".format(
          epoch, num_epochs, time.time() - start_time, learn_rate.get_value()))
    print("  training loss:{:.6f}, validation loss:{:.6f}".format((train_err / train_batches), (val_err / val_batches)))
     
    flog1 = open(logfile,'ab')
    flog1.write("Epoch {} of {} took {:.3f}s Learning rate {}\n".format(
        epoch, num_epochs, time.time() - start_time, learn_rate.get_value()))
    flog1.write("  training loss:{:.6f}, validation loss:{:.6f}\n".format((train_err / train_batches), (val_err / val_batches)))
      
    flog1.write("\n")
    flog1.close()
    
    valE = val_err/val_batches
    min_loss_network = network['deconv4'] 
    if valE <= min_val_loss:

      #save the network parameters corresponding to this loss
      min_loss_network = network['deconv4'] 
      patience=0
      min_val_loss = valE
      mloss_epoch=epoch+1
      
      mname = 'ConvAE-weights-epoch-%d'%(epoch+1)
      spth = os.path.join(epoch_path,mname+'.npz')
      save(min_loss_network,spth)

    #Patience / Early stopping
    else:
      #increase the patience counter
      patience+=1
      #decrease the learning rate
      learn_rate.set_value(learn_rate.get_value()*lr_decay)
      spth = os.path.join(wts_path,'ConvAE_minloss_valincr.npz')
      save(min_loss_network,spth)

    if patience==5:
      break
  #Decay the learning rate based on performance on validation set 
    if val_prev - valE <= 0.001:
      learn_rate.set_value(learn_rate.get_value()*lr_decay)
      val_counter+=1
    
    val_prev = valE

    if val_counter==5:
      spth = os.path.join(wts_path,'ConvAE_minloss_valincr.npz')
      save(min_loss_network,spth)
      break #break out
    
    if epoch == num_epochs: 
      spth = os.path.join(wts_path,'ConvAE_minloss_nepoch.npz')
      save(min_loss_network, spth)
      break

 

