#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 23:09:39 2017

@author: judy
"""


from framework import modAttentiondef
from framework.utils import to_var, zdim_analysis
import torch
import torchvision

NB_FEN = 32
N_FFT = 100
BATCH_SIZE = 50
#LEN_EXAMPLES = 38400
LEN_EXAMPLES = 2000
# Net parameters
Z_DIM, H_DIM = 20, 400

betaVAE = modAttentiondef.AttentionRnn(sample_size=N_FFT, h_dim=H_DIM,
                                       z_dim=Z_DIM)
for i in xrange(Z_DIM):
    Z_DIM_SEL = i+1
    FIXED_Z = zdim_analysis(BATCH_SIZE, Z_DIM, Z_DIM_SEL, -10, 20)
    FIXED_Z = to_var(torch.Tensor(FIXED_Z.contiguous()))
    FIXED_Z = FIXED_Z.repeat(NB_FEN, 1)
    FIXED_Z = FIXED_Z.view(NB_FEN, BATCH_SIZE, Z_DIM)

    # Sampling from model, reconstructing from spectrogram
    sampled_image = betaVAE.sample(FIXED_Z)
    sampled_image = sampled_image.transpose(0, 2)
    sampled_image = torch.chunk(sampled_image, N_FFT, 0)
    sampled_image = torch.cat(sampled_image, 2).squeeze()
    sampled_image = sampled_image.view(BATCH_SIZE, 1, N_FFT, -1)
    torchvision.utils.save_image(sampled_image.data.cpu(),
                                 './data/spectro/sampled_images_%d.png'%(Z_DIM_SEL))