# Librairies
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


# ------------------------------------------------------------------------
def to_var(tensor):
    """ Converts the given tensor into a torch Variable. Loads the data onto 
    the GPU if available
    
    Arguments:
        - torch.tensor: Data to be converted
    
    Returns:
        - Variable object
    """ 
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)


# ------------------------------------------------------------------------
    
def zdim_analysis(BATCH_SIZE, TOTAL_Z_DIM, Z_DIM, start=0., stop=100.):
    """ Returns a matrix where only one dimension changes 
    
    Arguments:
        - BATCH_SIZE: size of the batch to input
        - TOTAL_Z_DIM: number of latent dim
        - Z_DIM: which z_dim to variate
        - start: start value
        - stop: stop value
        -inc: increment
        
    Returns:
        - sampling_data: desired matrix [BATCH_SIZE * TOTAL_Z_DIM]
    """
    #Z_DIM = Z_DIM + 1
    BATCH_SIZE, start, stop = BATCH_SIZE, start, stop
    STEP = float(stop-start)/float(BATCH_SIZE-1)
    
    a = torch.arange(start, stop, step=STEP)
    
    if Z_DIM == 1:        
        c = torch.zeros(BATCH_SIZE, TOTAL_Z_DIM-1)
        out = torch.cat( (a.unsqueeze(1),c),dim=1)        
    elif Z_DIM-1 == TOTAL_Z_DIM:
        b = torch.zeros(BATCH_SIZE, TOTAL_Z_DIM-1)
        out = torch.cat((b,a),dim=1)
    else:
        b = torch.zeros(BATCH_SIZE, Z_DIM)
        c = torch.zeros(BATCH_SIZE, TOTAL_Z_DIM-Z_DIM-1)
        out = torch.cat((b,a,c),dim=1)

    return out
