''' AudioClass and AudioDataset class, 
which wrapps up related operations on an audio
'''

if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    
import numpy as np 
import cv2
import librosa
import matplotlib.pyplot as plt 
from collections import namedtuple

import torch
from torch.utils.data import Dataset

if 1: # my lib
    import utils.lib_proc_audio as lib_proc_audio
    import utils.lib_plot as lib_plot
    import utils.lib_io as lib_io
    import utils.lib_commons as lib_commons

class AudioDataset(Dataset):
    def __init__(self, 
                 data_folder="", classes_txt="",
                 files_name=[], files_label=[],
                 transform=None):
        
        assert (data_folder and classes_txt) or (files_name, files_label) # input either one
        
        # Get all data's filename and label
        if files_name and files_label:
            self.files_name, self.files_label = files_name, files_label
        else:
            func = AudioDataset.load_filenames_and_labels
            self.files_name, self.files_label = func(data_folder, classes_txt)
            
        # Store
        self.transform = transform
        self.files_label = torch.tensor(self.files_label, dtype=torch.int64)

    @staticmethod
    def load_filenames_and_labels(data_folder, classes_txt):
        # Load classes
        with open(classes_txt, 'r') as f:
            classes = [l.rstrip() for l in f.readlines()]
        
        # Based on classes, load all filenames from data_folder
        files_name = []
        files_label = []
        for i, label in enumerate(classes):
            folder = data_folder + "/" + label + "/"
            
            names = lib_commons.get_filenames(folder, file_types="*.wav")
            labels = [i] * len(names)
            
            files_name.extend(names)
            files_label.extend(labels)
        
        return files_name, files_label
            
    def __len__(self):
        return len(self.files_name)

    def __getitem__(self, idx):
        
        # Load audio as a class instance
        filename=self.files_name[idx]
        audio_c = AudioClass(filename=filename)
        print(f"Load file: {filename}")

        # Do transform (augmentation)        
        if self.transform:
            self.transform(audio_c.audio)

        # Compute mfcc feature
        audio_c.compute_mfcc(n_mfcc=12) # return mfcc

        # Compose X, Y
        X = torch.tensor(audio_c.mfcc.T, dtype=torch.float32) # (time_len, feature_dim)
        Y = self.files_label[idx]
        return (X, Y)
    
class AudioClass(object):
    def __init__(self, 
                 audio=None, sample_rate=None, filename=None,
                 n_mfcc=12):
        if filename:
            self.audio, self.sample_rate = lib_io.read_audio(filename)
        elif (audio and sample_rate):
            self.audio, self.sample_rate = audio, sample_rate
        else:
            assert 0, "Invalid input"
            
        self.mfcc = None
        self.n_mfcc = n_mfcc # feature dimension of mfcc 
        self.mfcc_image = None 
        self.mfcc_histogram = None
    
    def _check_and_compute_mfcc(self):
        if self.mfcc is None:
            self.compute_mfcc()
            
    def compute_mfcc(self, n_mfcc=None):
        # https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
       
        # Check input
        if n_mfcc is None:
            n_mfcc = self.n_mfcc
        if self.n_mfcc is None:
            self.n_mfcc = n_mfcc
            
        # Compute
        self.mfcc = librosa.feature.mfcc(
            y=self.audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,        
        )
    
    def compute_mfcc_histogram(
            self, bins=10, binrange=(-50, 200), col_divides=5,
        ): 
        ''' Function:
                Divide mfcc into $col_divides columns.
                For each column, find the histogram of each feature (each row),
                    i.e. how many times their appear in each bin.
            Return:
                features: shape=(feature_dims, bins*col_divides)
        '''
        self._check_and_compute_mfcc()
        self.mfcc_histogram = lib_proc_audio.calc_histogram(
            self.mfcc, bins, binrange, col_divides)
        
        self.args_mfcc_histogram = ( # record parameters
            bins, binrange, col_divides,)
        
    def compute_mfcc_image(
            self, row=200, col=400,
            mfcc_min=-200, mfcc_max=200,
        ):
        ''' Convert mfcc to an image by converting it to [0, 255]'''        
        self._check_and_compute_mfcc()
        self.mfcc_img = lib_proc_audio.mfcc_to_image(
            self.mfcc, row, col, mfcc_min, mfcc_max)
    

    # It's difficult to set this threshold, better not use this funciton.
    def remove_silent_prefix(self, threshold=50, padding_s=0.5):
        ''' Remove the silence at the beginning of the audio. '''
         
        l0 = len(self.audio) / self.sample_rate
        
        func = lib_proc_audio.remove_silent_prefix_by_freq_domain
        self.audio, self.mfcc = func(
            self.audio, self.sample_rate, self.n_mfcc, 
            threshold, padding_s, 
            return_mfcc=True
        )
        
        l1 = len(self.audio) / self.sample_rate
        print(f"Audio after removing silence: {l0} s --> {l0} s")
        
    # --------------------------- Plotting ---------------------------
    def plot_audio(self):
        lib_plot.plot_audio(self.audio, self.sample_rate)
        
    def plot_mfcc(self, method='librosa'):
        self._check_and_compute_mfcc()
        lib_plot.plot_mfcc(self.mfcc, self.sample_rate, method)

    def plot_mfcc_histogram(self):
        if self.mfcc_histogram is None:
            self.compute_mfcc_histogram()
            
        lib_plot.plot_mfcc_histogram(
            self.mfcc_histogram, *self.args_mfcc_histogram)

    def plot_mfcc_image(self):
        if self.mfcc_image is None:
            self.compute_mfcc_image()
        plt.show(self.mfcc_img)
        plt.title("mfcc image")

def test_Class_AudioData():
    audio = AudioClass(filename="test_data/audio_1.wav")
    audio.plot_audio()
    audio.plot_mfcc()
    audio.plot_mfcc_histogram()
    
    # plot
    plt.show()

def main():
    test_Class_AudioData()
    pass 

if __name__ == "__main__":
    main()