# frameworks

We implemented several frameworks. In order of developpement:
* **VAE** `modVAE.py`: Simple VAE model, with 2 fully connected linear
  layers, and non-linear transformations
* **1D convolutional VAE** `modVAE1D.py`: an attempt of using 1D convolutions
  on pure sound database
* **2D convolutional VAE** `CNN_VAE.py`: a more serious attempt. Uses two
  layers of 2D convolution, then inputs it into a three linear layer encoder.
* **Recurrent Neural Network** `modAttentiondef.py`: Our most serious model,
  using an attention RNN model.


`utils.py` is just a function container, utilities mostly.

## Link to other folders
Usually, every net has its own data folder (like `data/RNN` for the attention RNN model). You will find a `original_image.png`, that is a fixed input from our dataset, and the consequent reconstructed images `reconst_image_<epoch>.png`. 

