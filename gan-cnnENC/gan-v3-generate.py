from __future__ import print_function
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import h5py
from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from torchvision import models

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
#parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass
opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 300, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        #cnn encoder
        self.layer1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, 32, 5, 1, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 96, 5, 1, 2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(2))
        
        self.fc = nn.Linear(8*8*96,300)
        
    def forward(self, input):
        cond = self.layer1(input)
        cond = self.layer2(cond)
        cond = self.layer3(cond)
        cond = cond.view(cond.size(0),-1)
        cond = self.fc(cond)

        cond = torch.unsqueeze(torch.unsqueeze(cond,2),3)
        #cond_input = torch.cat((input,cond),1)
        
        return self.main(cond)

netG = _netG()
#set generator weights
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

conditioning = torch.FloatTensor(opt.batchSize,3,opt.imageSize,opt.imageSize)

#move generator and discriminator to the GPU
netG.cuda()
conditioning = conditioning.cuda()

#convert to autograd
conditioning = Variable(conditioning)

#data stuff
train_set = H5PYDataset('/misc/data15/reco/bhattgau/Rnn/code/Code_/mscoco/cocotrain/inpainting/mscoco-valid.hdf5',which_sets=('train',),subset=slice(0,800))
scheme = ShuffledScheme(examples=train_set.num_examples, batch_size=8)

train_stream = DataStream(dataset=train_set, iteration_scheme=scheme)

tot_files = train_set.num_examples
nbatches = tot_files/opt.batchSize


for i, Data in enumerate(train_stream.get_epoch_iterator()):
    # train with real
    Data1 = Data[1] 
    t_data = torch.from_numpy(Data1)
    
    c_data  = torch.from_numpy(Data[0])
    
    outer_part = np.copy(Data[0])

    real_cpu = t_data

    #embed the conditioning data
    batch_size = real_cpu.size(0)
    conditioning.data.resize_(c_data.size()).copy_(c_data) 

    gen_image = netG(conditioning)
    generated = gen_image.data.cpu().numpy()
    center = (int(np.floor(generated.shape[2] / 2.)), int(np.floor(generated.shape[3] / 2.)))
    outer_part[:,:,center[0]-16:center[0]+16, center[1]-16:center[1]+16] = generated[:,:,center[0]-16:center[0]+16, center[1]-16:center[1]+16]

    final_image = torch.from_numpy(outer_part)
  
    vutils.save_image(final_image,'%s/gen_images%d.png' % (opt.outf, i))
    vutils.save_image(real_cpu,'%s/real_images%d.png' % (opt.outf, i))

