#-*-encoding:UTF-8-*-
""" AudioEngine module, part of package toyDataset 
ATIAM 2017 """

import librosa as lib
import numpy as np

class audioEngine:
    def __init__(self,Fs_Hz=16000, n_fft=1024):
        self.Fs = Fs_Hz
        self.n_fft = n_fft
        
    def render_sound(self, params, sound_length):
        """ Render the sample from a dictionnary of parameters

        INPUT: 
            - Dictionnary of parameters

        OUTPUT:
            - Numpy array of size (N x 1) containing the sample
        """

        return np.random.rand((sound_length))


    def spectrogram(self, data):
        """ Returns the spectrograms of the array of sounds 'data'
        
        INPUT:
            data: array of n sounds (n*SOUND_LENGTH)
            
        OUTPUT:
            output: array of spectrograms
        
        """
        # M: number of sounds
        # N: number of samples
        (M,N) = np.shape(data)
        
        # Allocating the memory needed
        spectrograms = [lib.stft(data[1], n_fft=self.n_fft) * 0 for i in xrange(M)]

        # FOR LOOP: computing spectrogram
        for i in xrange(M):
            spectrograms[i] = np.abs( lib.stft(data[i], n_fft=self.n_fft) )

        return spectrograms
    
        # La fonction prend en paramètres un spectrogramme S    
        # et le nombre de pts de la NFFT désirée.
        # Elle retourne un vecteur correspondant à l'audio reconstruit
    
    def griffinlim(self, S):
        """ Returns a sound, reconstructed from a spectrogram with NFFT points.
        Griffin and lim algorithm
        
        INPUT:
            - S: spectrogram (array)
        
        OUTPUT:
            - x: signal """
        # ---- INIT ----
        # Create empty STFT & Back from log amplitude
        n_fft = S.shape[0]
        S = np.log1p(np.abs(S))

        #a = np.zeros_like(S)
        a = np.exp(S) - 1
        
        # Phase reconstruction
        p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
        
        # LOOP: iterative reconstruction
        for i in range(100):
            S = a * np.exp(1j*p)
            x = lib.istft(S)
            p = np.angle(lib.stft(x, self.n_fft))
    
        return x
