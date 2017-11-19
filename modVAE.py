""" modVAE.py
Implementation of a VAE model """

# Librairies
import torch
import torch.nn as nn
from torch.autograd import Variable

def to_var(tensor):
    """ to_var(tensor):
            Converts data into a torch tensor, and puts it on the GPU if
            available"""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)

# VAE model
class VAE(nn.Module):
    """ Variational audo-encoder class """

    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()

        # ENCODER
        # Linear, LeakyReLU, Linear
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, z_dim*2))  # 2 for mean and variance.

        # DECODER
        # Linear, ReLU, linear, Sigmoid
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, image_size),
            nn.Sigmoid())

    def reparametrize(self, mu, log_var):
        """"z = mean + eps * sigma where eps is sampled from N(0, 1)."""
        eps = to_var(torch.randn(mu.size(0), mu.size(1)))
        output = mu + eps * torch.exp(log_var/2)    # 2 for convert var to std
        return output

    def forward(self, data):
        """ forward(x):
            Do the forward pass in the VAE model """
        h = self.encoder(data)
        mu, log_var = torch.chunk(h, 2, dim=1)  # mean and log variance.
        reparam = self.reparametrize(mu, log_var)
        output = self.decoder(reparam)
        return output, mu, log_var

    def sample(self, hiddenVec):
        """ sample(hiddenVec):
            Decodes data from the z space """
        return self.decoder(hiddenVec)


