#-*-encoding:UTF-8-*-
""" ---------------------------------------------
VAE with a 2D convnet
"""

# Librairies
import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
from framework.utils import to_var
import logging

class CNN(nn.Module):
    """ Variational Audo-Encoder (VAE) class, with a 1D convolutional net """

    def __init__(self, height=784, width=100, kernel_size=5, h_dim=400, z_dim=5):
        """ Creates a VAE net object. 
        
        Arguments:
            - image_size (int, def: 784): size of the flattened image (1D)
            - h_dim (int, def:400): size of the hidden layer
            - z_dim (int, def:5): number of latent dimensions

        Returns:
            - nn.Module object
        """
        super(CNN, self).__init__()
        self.sound_length=height*width
        self.height = height
        self.width = width
        self.kernel_size = kernel_size
        
        # ENCODER
        self.encoder = nn.Sequential(
            nn.Linear(32*self.sound_length/((self.kernel_size-1)**2),1200),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(1200, h_dim),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim*2),
            nn.ReLU())  # 2 for mean and variance.

        # DECODER
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1200),
            nn.ReLU6(),
            nn.Linear(1200, self.sound_length*32/((kernel_size-1)**2)))
        
        # Convolutional layer 1 & 2
        self.layer1_forward = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=self.kernel_size, padding=2),
            nn.Dropout2d(),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        
        self.layer2_forward = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=self.kernel_size, padding=2),
            nn.Dropout2d(),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        # Deconvolutional layers 1 & 2
        self.layer1_backward = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=self.kernel_size, padding=2),
                nn.ReLU())
        
        self.layer2_backward = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.ConvTranspose2d(16, 8, kernel_size=2,
                                    stride=2,))
                                # padding=2))
        
        self.layer3_backward = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm2d(8),
                nn.ConvTranspose2d(8, 1, kernel_size=2,stride=2))#, padding=2))

        # Initializing weights
        self.encoder.apply(self.init_weight)
        self.decoder.apply(self.init_weight)
        
        # Weight normalization
        self.w_enc = []
        for module in self.encoder:
            if type(module) is nn.Linear:
                self.w_enc.append(nn.utils.weight_norm(module))

        self.w_dec = []
        for module in self.decoder:
            if type(module) is nn.Linear:
                self.w_dec.append(nn.utils.weight_norm(module))


    def init_weight(self, module):
        """ This function initialize the weight of a linear layer

        Arguments:
            - module: the sequential to initialize weights

        """
        if type(module) == nn.Linear:
            nn.init.xavier_uniform(module.weight)

    def reparametrize(self, mu, log_var):
        """" Does the reparametrize trick on the beta-VAE:
            $z = mean + eps * sigma$ where 'eps' is sampled from N(0, 1).

        Arguments: 
            - mu: mean of the distribution
            - log_var: log value of the variance

        Returns:
            - output: Reparametrized data (mu + eps * exp(log_var/2))
        
        """
        eps = to_var(torch.randn(mu.size(0), mu.size(1)))
        output = mu + eps * torch.exp(log_var/2)    # 2 for convert var to std
        return output

    def forward(self, data):
        """ Do the forward pass in the VAE model: encodes, reparameterize and
        decodes the input data.
        
        Arguments:
            - data (array[SAMPLE_INDEX][DATA_VECTOR]):

        Returns:
            - output (array like data): data encoded, reparametrized then decoded
            - mu: mean 
            - log_var: log value of the variance
        
        """
        # Convolution 
        outconv_1 = self.layer1_forward(data)
        outconv_2 = self.layer2_forward(outconv_1)

        # return out
        outconv_2 = outconv_2.view(outconv_2.size(0), -1)
        
        # Encoder
        encoded_vec = self.encoder(outconv_2)
        mu, log_var = torch.chunk(encoded_vec, 2, dim=1)  # mean and log variance.
        
        
        # Reparametrization
        reparam = self.reparametrize(mu, log_var)
        reparam = reparam.unsqueeze(1)

        # Decoder
        decoded = self.decoder(reparam)
        decoded_unsq = decoded.unsqueeze(1)
        decoded_unsq = decoded_unsq.view(decoded_unsq.size(0),
                             32, 
                             self.height/(self.kernel_size-1), 
                             self.width/(self.kernel_size-1))
        
        # Return output
        output_back1 = self.layer1_backward(decoded_unsq)
        output_back2 = self.layer2_backward(output_back1)
        output_back3 = self.layer3_backward(output_back2)
        
        return output_back3, mu, log_var

    def sample(self, latent_vec):
        """ Decodes the given latent vector.
        
        Arguments:
            - latent_vec: matrix containing a number of encoded vectors

        Returns:
            - output: decoded vector
        """
        decoded = self.decoder(latent_vec)
        decoded_unsq = decoded.unsqueeze(1)
        decoded_unsq = decoded_unsq.view(decoded_unsq.size(0),
                             32, 
                             self.height/(self.kernel_size-1), 
                             self.width/(self.kernel_size-1))
        
        # Return output
        output_back1 = self.layer1_backward(decoded_unsq)
        output_back2 = self.layer2_backward(output_back1)
        output_back3 = self.layer3_backward(output_back2)
        return output_back3


