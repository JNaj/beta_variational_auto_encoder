#-*-encoding:UTF-8-*-
""" AudioEngine module, part of package toyDataset 
ATIAM 2017 """

import librosa as lib
import numpy as np
import torch as to

class audioEngine:
    """ audioEngine class
    This class contains the methods that handles synthesis from a
    parameterSpace object 
    """
    def __init__(self,Fs_Hz=16000, n_bins=1024):
        self.Fs = Fs_Hz
        self.n_bins = n_bins
        
    def render_sound(self, params, sound_length, n_modes=10):
        """ Render the sample from a dictionnary of parameters

        Arguments: 
            - Dictionnary of parameters

        Returns:
            - Numpy array of size (N x 1) containing the sample
        """
        ### INITIALIZATION
        # Retrieve parameters from dict
        f0 = params['f0']
        slope = params['PS']
        harmonicPresence = params['PH']
        inharmonicity = params['inh']
        time_decay = params['decay']
        
        # Sampling parameters
        Fs = self.Fs # Sampling rate
        Ts = 1.0/Fs
        length = sound_length/Fs # length of the signal (in secondes)
        N = sound_length # number of samples
        t = np.arange(0,N,1)*Ts # time vector
        
        # Additive synthesis
        if harmonicPresence == 0: # tous les modes
            modes = np.arange(1, n_modes+1)
        elif harmonicPresence == 1: # que les harmoniques impaires
            modes = np.arange(1, n_modes+1, 2)
            n_modes = n_modes/2
        elif harmonicPresence == 2: # que des harmoniques paires
            modes = np.arange(2, n_modes+1, 2)
            modes = np.concatenate((np.array([1]), modes))
            n_modes = n_modes/2 + 1        
        
        freq = modes * f0 * np.sqrt(1 + inharmonicity * pow(modes, 2))
        amp = (modes-1)*slope + n_modes
        amp = (amp/float(n_modes)).clip(min=0)
        x = np.zeros(N) # signal
        
        for k in range(n_modes-1):
            x = x + amp[k]*np.sin(2*np.pi*freq[k]*t)
            
        # time decay
        decay_signal = np.exp(-time_decay*t)
        x = x*decay_signal
                
        
        ### FORMATTING
        y = x
        y = y/max(y)
        y = np.ndarray.astype(y, np.float,copy=False)
        
        return np.real(y)


    def spectrogram(self, data):
        """ Returns the absolute value of the stft of the array of sounds 'data.
        Size : (N_fft, N_frame)
        
        Arguments:
            data: array of n sounds (n*SOUND_LENGTH)
            
        Returns:
            output: absolute value of the stft of every sound in the sound array
        
        """
        # --- INIT ---
        # M: number of sounds
        # N: number of samples
        (M,N) = np.shape(data)
        
        # Allocating the memory needed
        temp_spec = lib.stft(data[0], n_bins=self.n_bins)
        spectrograms = [np.zeros_like(temp_spec) for i in xrange(M)]

        # FOR LOOP: computing spectrogram
        for i in xrange(M):
            spectrograms[i] = np.abs(lib.stft(data[i], n_bins=self.n_bins))
    
        return spectrograms
 
    def cqt(self, data):
        """ Returns the absolute value of the CQT of the array of sounds 'data'
        
        Arguments:
            data: array of n sounds (n*SOUND_LENGTH)
            
        Returns:
            output: CQT
        
        """
        # --- INIT ---
        # M: number of sounds
        # N: number of samples
        (M,N) = np.shape(data)
        
        # Allocating the memory needed
        temp_cqt = lib.core.cqt(data[0], n_bins=self.n_bins)
        cqt = [np.zeros_like(temp_cqt) for i in xrange(M)]

        # cleaning memory
        del(temp_cqt)

        # FOR LOOP: computing spectrogram
        for i in xrange(M):
            cqt[i] = np.abs(lib.core.cqt(data[i], n_bins=self.n_bins),
                            dtype=np.dtype('float'))
    
        return cqt
   
    def griffinlim(self, S, N_iter=100):
        """ Returns a sound, reconstructed from a spectrogram with NFFT points.
        Griffin and lim algorithm
        
        Arguments:
            - S: spectrogram (array) (absolute value of the stft)
            - N_iter (def: 100): number of iteration for the reconstruction
        
        Returns:
            - x: signal """
        # ---- INIT ----
        # Create empty STFT & Back from log amplitude
        n_bins = S.shape[0]
        S = np.log10(np.abs(S))

        a = np.exp(S) - 1
        
        # Phase reconstruction
        p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
        
        # LOOP: iterative reconstruction
        for i in range(N_iter):
            S = a * np.exp(1j*p)
            x = lib.istft(S)
            p = np.angle(lib.stft(x, self.n_bins))
    
        return np.real(x)
