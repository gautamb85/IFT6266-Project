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
from lasagne.nonlinearities import leaky_rectify
from collections import OrderedDict
import h5py
from fuel.datasets import H5PYDataset
from lasagne.regularization import regularize_layer_params, l1
from lasagne.init import GlorotUniform,HeNormal
from PIL import Image

def build(net_input=None):
  network={}
  print("Building network ...")
  network['input'] = lasagne.layers.InputLayer(shape=(None,3, 64, 64),input_var=net_input)
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

  network['deconv4'] = lasagne.layers.Conv2DLayer(network['up3'], num_filters=3, filter_size=(5,5),pad='same',nonlinearity=None)
  
  return network


def generate(weights,data_in):
  X = T.tensor4(name='masked_images')
  network=build(net_input=X)
  set_weights(network,weights)
  
  predicted_image = lasagne.layers.get_output(network['deconv4'], deterministic=True)
  get_image = theano.function([X],predicted_image.transpose((0,2,3,1)))
  Y = get_image(data_in)
  #Y = np.reshape(Y,(Y.shape[0],64,64,3))
  return Y

def set_weights(network,weights):

  print("Loading trained network")
  with np.load(weights) as f:
     param_values=[f['arr_%d' % i] for i in range(len(f.files))]

  lasagne.layers.set_all_param_values(network['deconv4'],param_values)
  
  all_parameters = lasagne.layers.get_all_params(network['deconv4'], trainable=True)

  print("Model Parameters")
  print("-"*40)
  for param in all_parameters:
      print(param, param.get_value().shape)
  print("-"*40)

def save_images(images,save_path):
  for ind,i in enumerate(images):
    name='gen_image-%d'%(ind+1)
    name=name+'.npy'
    sfile = os.path.join(save_path,name) 
    np.save(sfile,i)

def main():
  #make savedir
  spth='/misc/data15/reco/bhattgau/Rnn/code/Code_/mscoco/cocotrain/inpainting/convAE_imgs'
  if not os.path.exists(spth):
    command0 = "mkdir -p" +" "+ spth
    process = subprocess.check_call(command0.split())
 
  
  #Load MSCOCO Dataset
  train_set = H5PYDataset('/misc/data15/reco/bhattgau/Rnn/code/Code_/mscoco/cocotrain/inpainting/mscoco-train.hdf5',which_sets=('train',),subset=slice(0,10))
  h= train_set.open()
  DATA = train_set.get_data(h,slice(0,train_set.num_examples))

  data_in = DATA[0]
  print(data_in.shape)
  data_in=np.reshape(data_in,(data_in.shape[0],3,64,64))
  print(data_in.shape)

  weights= '/misc/scratch03/reco/bhattaga/data/i-vectors/dnn_projects/MSCOCO/convAE-3/epoch_weights/ConvAE-weights-epoch-6.npz'
  images = generate(weights,data_in)
  save_images(images,spth)

if __name__=='__main__':
  main()
