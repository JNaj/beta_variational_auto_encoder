# largement inspiré de https://wiseodd.github.io/techblog/2017/01/24/vae-pytorch/#

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data

class VAE(torch.nn.Module):
    def __init__(self, D_in, H, D_out, N):
        super(VAE, self).__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.LeakyReLU(0.2), torch.nn.Linear(H, D_out*2))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(D_out, H), torch.nn.ReLU(), torch.nn.Linear(H, D_in), torch.nn.Sigmoid())
        
    def repar(self, mu, log_var):
        eps = Variable(torch.randn(N, D_out))
        return mu + torch.exp(log_var / 2) * eps
    
    def forward(self, x):
        x_enc = self.encoder(x)
        mu, log_var = torch.chunk(x_enc, 2, dim=1)
        x_rep = self.repar(mu, log_var)
        x_dec = self.decoder(x_rep)
        return x_dec, mu, log_var
        
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
D_in = mnist.train.images.shape[1]
D_out = 100
H = 128
N = 64
c = 0
lr = 1e-3
beta = 1

# Construct our model by instantiating the class defined above

betaVAE = VAE(D_in,H,D_out,N)
params = list(betaVAE.parameters())
solver = optim.Adam(params, lr=lr)

for t in range(300):
    x, _ = mnist.train.next_batch(N)
    x = Variable(torch.from_numpy(x))
    
    # Forward pass
    x_dec, mu, log_var = betaVAE(x)

    # Compute loss
    recon_loss = torch.nn.functional.binary_cross_entropy(x_dec, x, size_average=False) / N
    kl_divergence = torch.mean(0.5 * torch.sum(torch.exp(log_var) + mu**2 - 1. - log_var, 1))
    loss = recon_loss + beta*kl_divergence

    # Zero gradients, perform a backward pass, and update the weights.
    solver.zero_grad()
    loss.backward()
    solver.step()
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())
            
    #affichage       
    if t % 20 == 0:
        print('Iter-{}; Loss: {:.4}'.format(t, loss.data[0]))

        samples = x_dec.data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
        c += 1
plt.close(fig)
