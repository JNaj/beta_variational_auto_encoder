"""
Beta Variational Auto-Encoder
Derived from Pytorch tutorial at
https://github.com/yunjey/pytorch-tutorial
"""

#%% Librairies
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms

from framework import modVAE
from framework.utils import to_var,zdim_analysis

from toyDataset import dataset as dts

import framework.CNN_VAE as CNN_VAE
import numpy as np
#%% Importing DATASET

# MNIST dataset
DATASET = datasets.MNIST(root='./data',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)    
# Data loader
DATA_LOADER = torch.utils.data.DataLoader(dataset=DATASET,
                                           batch_size=100,
                                           shuffle=True)

#%%
""" TRAINING THE VAE MODEL """

ITER_PER_EPOCH = len(DATA_LOADER)
DATA_ITER = iter(DATA_LOADER)

# fixed inputs for debugging
FIXED_X, _ = next(DATA_ITER)
torchvision.utils.save_image(FIXED_X.cpu(), './data/MNIST/real_images.png')
#FIXED_X = to_var(FIXED_X.view(FIXED_X.size(0), -1))
FIXED_X = to_var(torch.Tensor(FIXED_X)).view(FIXED_X.size(0),28, -1).unsqueeze(1)

DATA_ITER = enumerate(DATA_LOADER)    

#%% CREATING THE Beta-VAE
_,_,HEIGHT,WIDTH=FIXED_X.size()

H_DIM = 400
Z_DIM = 10
CONV1D_OUT = 10
reload(CNN_VAE)
betaVAE = CNN_VAE.CNN(height=28,
                      width=28,
                      h_dim=H_DIM, 
                      z_dim=Z_DIM,
                      kernel_size=5)

# BETA: Regularisation factor
# 0: Maximum Likelihood
# 1: Bayes solution
BETA = 12

# GPU computing if available
if torch.cuda.is_available():
    betaVAE.cuda()
    print('GPU acceleration enabled')

# OPTIMIZER
OPTIMIZER = torch.optim.Adam(betaVAE.parameters(), lr=0.001)

ITER_PER_EPOCH = len(DATA_LOADER)
NB_EPOCH = 10;

#%%
""" TRAINING """

for epoch in range(NB_EPOCH):
    # Epoch
    for i,(images,params) in enumerate(DATA_LOADER):
        
        # Formatting
        images = to_var(torch.Tensor(images)).view(images.size(0),28, -1).unsqueeze(1)
        out, mu, log_var = betaVAE(images)

        # Compute reconstruction loss and KL-divergence
#        reconst_loss = F.binary_cross_entropy(out, images, size_average=False)
        reconst_loss = -0.5*784*torch.sum(2*np.pi*log_var)
        reconst_loss -= torch.sum(torch.sum((images-out).pow(2))/((2*log_var.exp())))
        kl_divergence = torch.sum(0.5 * (mu**2
                                         + torch.exp(log_var)
                                         - log_var -1))

        # Backprop + Optimize
        total_loss = - reconst_loss + BETA*kl_divergence
        OPTIMIZER.zero_grad()
        total_loss.backward()
        OPTIMIZER.step()

        # PRINT
        # Prints stats at each epoch
        if i % 100 == 0:
            print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
                   "Reconst Loss: %.4f, KL Div: %.7f"
                   %(epoch+1,
                     NB_EPOCH,
                     i+1,
                     ITER_PER_EPOCH,
                     total_loss.data[0],
                     reconst_loss.data[0],
                     kl_divergence.data[0])
                  )

    # Save the reconstructed images
    reconst_images, _, _ = betaVAE(FIXED_X)
    reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
    torchvision.utils.save_image(reconst_images.data.cpu(),
                                 './data/MNIST/reconst_images_%d.png' %(epoch+1))
#%% SAMPLING 
# Creating a z vector to decode from

for i in xrange(Z_DIM):
    Z_DIM_SEL = i
    # 11
    FIXED_Z = zdim_analysis(100, Z_DIM, Z_DIM_SEL, -10, 20)
    FIXED_Z = to_var(FIXED_Z)
    # Sampling from model
    sampled_images = betaVAE.sample(FIXED_Z)
    
    # Saving
    sampled_images = sampled_images.view(sampled_images.size(0), 1, 28, 28)
    torchvision.utils.save_image(sampled_images.data.cpu(),
                                 './data/MNIST/sampled_zdim_%d.png'%(Z_DIM_SEL))
