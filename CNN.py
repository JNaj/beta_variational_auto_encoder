#-*-encoding:UTF-8-*-
"""
Beta Variational Auto-Encoder
"""

#%% Librairies
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms

from framework import modVAE1D
from framework import CNN_VAE
from framework.utils import to_var,zdim_analysis

from toyDataset import dataset as dts
import matplotlib.pyplot as plt
from numpy.random import randint
import numpy as np

import librosa
import os.path as pa
import pickle

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

#%% PARAMETERS
# Parameters, dataset
N_FFT = 100
BATCH_SIZE = 100
#LEN_EXAMPLES = 38400
LEN_EXAMPLES = 8000

# Net parameters
Z_DIM, H_DIM = 15, 500
KERNEL_SIZE = 5

FS = 16000

#%% Importing DATASET
# Creating dataset
DATASET_FILEPATH = 'data/datasets/DATASET.obj'

# If there is no archive of the dataset, it needs to be rendered
if not pa.isfile(DATASET_FILEPATH):
    print 'Generating Dataset \n\n'
    DATASET = dts.toyDataset(length_sample=LEN_EXAMPLES,
                            n_bins=N_FFT,
                            Fe_Hz=FS,
                            data='cqt')
    obj = DATASET
    file_obj = open(DATASET_FILEPATH, 'w')
    pickle.dump(obj, file_obj)
    print 'File is ' + DATASET_FILEPATH
else:
    # Otherwise, load the pickled archive
    print 'Importing dataset at location' + DATASET_FILEPATH
    DATASET = pickle.load(open(DATASET_FILEPATH,'rb'))

IMG_LENGTH = np.shape(DATASET.__getitem__(9)[0])[1]

for i in xrange(25, 75):
    if (IMG_LENGTH % i) == 0:
        BATCH_SIZE = i
        print('Mini_batch size is %d'%(BATCH_SIZE))
        break


SOUND_LENGTH = np.shape(DATASET.__getitem__(9)[0])[0]


DATA_LOADER = torch.utils.data.DataLoader(dataset=DATASET,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)

#%% Saving original image
# Saving an item from the dataset to debug
DATA_ITER = iter(DATA_LOADER)
FIXED_X,_ = next(DATA_ITER)

FIXED_X = torch.Tensor(FIXED_X.type(torch.FloatTensor))
FIXED_X = FIXED_X.view(BATCH_SIZE, N_FFT, -1).unsqueeze(1)
_,_,HEIGHT,WIDTH = FIXED_X.size()


FIXED_X = to_var(FIXED_X)
torchvision.utils.save_image(FIXED_X.data.cpu(), './data/CNN/real_images.png')

#%% CREATING THE Beta-VAE
reload(CNN_VAE)
betaVAE = CNN_VAE.CNN(height=HEIGHT,
                      width=WIDTH,
                      h_dim=H_DIM, 
                      z_dim=Z_DIM,
                      kernel_size=5)


# BETA: Regularisation factor
# 0: Maximum Likelihood
# 1: Bayes solution
BETA = 4


# GPU computing if available
if torch.cuda.is_available():
    betaVAE.cuda()
    print('GPU acceleration enabled')

# OPTIMIZER
OPTIMIZER = torch.optim.RMSprop(betaVAE.parameters())

ITER_PER_EPOCH = len(DATASET)/BATCH_SIZE
NB_EPOCH = 500;



""" TRAINING """
for epoch in range(NB_EPOCH):    
    # Epoch
    print(' ')
    print('\t \t  /=======================================\\')
    print('\t \t | - | - | - | EPOCH [%d/%d] | - | - | - | '%(epoch+1, NB_EPOCH))
    print('\t \t  \\=======================================/')
    print(' ')
    for i,(images,_) in enumerate(DATA_LOADER):

        # Formatting
        #images = images/max(images);
        images = images.type(torch.FloatTensor)
        images = images.view(images.size(0), N_FFT, -1).unsqueeze(1)
        images = to_var(torch.Tensor(images))
        
        # Input in the net
        out, mu, log_var = betaVAE(images)
        images = images.squeeze(1)

        # Loss
        reconst_loss = -0.5*torch.sum(2*np.pi*log_var)
        reconst_loss -= torch.sum(torch.sum((images-out).pow(2))/((2*log_var.exp())))
        
        kl_divergence = torch.sum(0.5 * (mu**2
                                         + torch.exp(log_var)
                                         - log_var -1))

        # Backprop + Optimize
        total_loss = -reconst_loss + BETA*kl_divergence
        OPTIMIZER.zero_grad()
        total_loss.backward()
        OPTIMIZER.step()

        # PRINT
        # Prints stats at each epoch
        if i%100 == 0:
            print ("Step [%d/%d] \t Total Loss: %.2f \t Reconst Loss: %.2f \t KL Div: %.3f"
                   %(i,
                     ITER_PER_EPOCH,
                     total_loss.data[0],
                     reconst_loss.data[0],
                     kl_divergence.data[0]))

    # Save the reconstructed images
    reconst_images, _, _ = betaVAE(FIXED_X)
    reconst_images = reconst_images.view(reconst_images.size(0), 1, N_FFT, -1)
    torchvision.utils.save_image(reconst_images.data.cpu(),
                                 './data/CNN/reconst_images_%d.png' %(epoch+1))

#%% SAMPLING FROM LATENT SPACE

FIXED_Z = zdim_analysis(BATCH_SIZE, Z_DIM, 50, -10, 10)

FIXED_Z = torch.Tensor(FIXED_Z.contiguous()).unsqueeze(1)

# Sampling from model, reconstructing from spectrogram
sampled_image = betaVAE.sample(FIXED_Z)
reconst_images = sampled_image.view(BATCH_SIZE, 1, N_FFT, -1)
torchvision.utils.save_image(reconst_images.data.cpu(), './data/CNN/sampled_images.png')

#%%
obj = betaVAE
MODEL_FILEPATH = 'data/models/CNN-2D_beta4_H-DIM750_kernel5_ZDIM_20_STRIDE_1_full_dataset_small.model'
file_obj = open(MODEL_FILEPATH, 'w')
pickle.dump(obj, file_obj)
print 'File is ' + MODEL_FILEPATH
