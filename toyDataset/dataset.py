#-*-encoding:UTF-8-*-
""" Toy Dataset class:
    Create a toyDataset object. 
    Calls the modules
    toydataset.generateParameterSpace and toydataset.audioEngine. The first
    makes a cardinal product of the specficied parameters. The second contains
    the method to synthesize the samples.
    """

# Librairies
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import librosa as lib
import generateParameterSpace as gps
import audioEngine as aud

class toyDataset(Dataset):
    def __init__(self, Fe_Hz=8000, 
                 length_sample=64000, 
                 batch_size=100, 
                 n_bins=1024, 
                 data='spectro',
                 affich=True):
        """ ToyDataset object. When initialized, synthesized the dataset with
        the specifications defined in the parameter_space objet 

        Arguments:
            -(opt) Fs: samplerate (def: 16kHz)
            -(opt) length_sample: number of sample per sound example (def:64000)
            -(opt) batch_size: OBSOLETE size of the dataset (def: 100)
            -(opt) n_bins: order of the n_bins analysis (def: 1024)

        Returns:
            - toyDataset object
        """
        if affich:
            print('--- DATASET ---')
            print(' ')

        # --- INIT ---
        # Scalar variables
        self.Fs = Fe_Hz
        self.batch_size = batch_size
        self.length_sample = length_sample
        self.n_bins = n_bins
        self.data = data
        self.affich = affich # if True, displays debugging messages

        # Data structures
        self.sound_data = [];
        self.spectrograms = [];
        self.cqt = [];

        ### EXERNAL OBJECTS
        # Parameter space
        self.paramSpace = gps.parameterSpace()
        self.paramSpace.generate_parameter_space()

        # Audio engine for audio rendering
        self.audio_engine = aud.audioEngine(Fs_Hz=self.Fs,n_bins=self.n_bins)

        ### RENDERING DATASET
        # Render the dataset
        if affich:
            print('Building dataset ...')

        self.render_dataset()

        if affich:
            print(' ')
            print('Dataset Built')
            print('--- END --- ')

    def render_dataset(self):
        """ Render the whole parameter space from the parameter space instance.

        Returns:
            - self.sound_data: array containing all the sampled
            - self.spectrograms: array of spectrograms
        """
        ### INITIALISATION
        # Allocating memory
        self.sound_data = np.zeros((self.paramSpace.N_samples, self.length_sample))

        if self.affich:
            print('Rendering dataset...')
        ### RENDERING DATASET
        # --- Audio
        # for loop on the parameter space
        for i in xrange(self.paramSpace.N_samples):
            params = self.paramSpace.param_dataset_dict[i]
            self.sound_data[i] = \
            self.audio_engine.render_sound(params, self.length_sample)
        
        # --- Building representation
        if self.data == 'spectro':
            self.spectrograms = self.audio_engine.spectrogram(self.sound_data)
        elif self.data == 'cqt':
            self.cqt = self.audio_engine.cqt(self.sound_data) 
        elif self.data == 'sound':
            print 'No representation builded'
        else:
            raise ValueError('Expected spectro or cqt, but got {}. Select \
            correct representation name'%(self.data))

        # --- END ---


    def __getitem__(self, index):
        """ Returns a tuple containing a data (sound, spectrogram or cqt)
        and the parameters that were used to generate the sound (label).

        Arguments:
            - index: index corresponding to a sample in the dataset

        Returns:
            - sample: dict {'image':[array: spectrogram], 'parameters':[parameters]}

        UNIT TEST:
            - Vérifier que la fonction retourne bien un dico avec un numpy array 
            pour la première clé et un ndarray de taille [1xparam]
        """
        param = self.paramSpace.permutations_array[index]

        # Returns the wanted data
        if self.data == 'spectro':
            data = self.spectrograms[index].reshape(1,-1)
            data = data/np.max(np.abs(data))
        elif self.data == 'sound':
            data = self.sound_data[index]
        elif self.data == 'cqt':
            data = self.cqt[index].reshape(1, -1)
            data = data/np.max(data)
            data = np.array(data, dtype=np.dtype('float'))
        else:
            raise ValueError('Expected spectro or cqt, but got ' + self.data + '. Select a\
            correct representation name')
        
        return data,param

    def __len__(self):
        """ Returns the length of the Dataset, e.g. the number of samples
        
        Returns
            - len (int): number of sounds

        UNIT TEST:
            - Vérifier qu'il renvoit bien un int > 0
        """
        return len(self.sound_data)
