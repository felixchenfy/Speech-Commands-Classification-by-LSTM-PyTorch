''' 

` class AudioClass
wraps up related operations on an audio

` class AudioDataset
a dataset for loading audios and labels from folder, for training by torch

` def synthesize_audio
API to synthesize one audio

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
import copy 
from gtts import gTTS
import subprocess

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
                 transform=None,
                 bool_cache_audio=False,
                 bool_cache_XY=True, # cache features
                 ):
        
        assert (data_folder and classes_txt) or (files_name, files_label) # input either one
        
        # Get all data's filename and label
        if files_name and files_label:
            self.files_name, self.files_label = files_name, files_label
        else:
            func = AudioDataset.load_filenames_and_labels
            self.files_name, self.files_label = func(data_folder, classes_txt)
        self.files_label = torch.tensor(self.files_label, dtype=torch.int64)
        self.transform = transform

        # Cache computed data
        self.bool_cache_audio = bool_cache_audio
        self.cached_audio = {} # idx : audio
        self.bool_cache_XY = bool_cache_XY
        self.cached_XY = {} # idx : (X, Y). By default, features will be cached
        
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
        
        print("Load data from: ", data_folder)
        print("\tClasses: ", ", ".join(classes))
        return files_name, files_label
            
    def __len__(self):
        return len(self.files_name)

    def get_audio(self, idx):
        if idx in self.cached_audio: # load from cached 
            audio = copy.deepcopy(self.cached_audio[idx]) # copy from cache
        else:  # load from file
            filename=self.files_name[idx]
            audio = AudioClass(filename=filename)
            # print(f"Load file: {filename}")
            self.cached_audio[idx] = copy.deepcopy(audio) # cache a copy
        return audio 
    
    def __getitem__(self, idx):
        
        timer = lib_commons.Timer()
        
        
        # -- Load audio
        if self.bool_cache_audio:
            audio = self.get_audio(idx)
            print("{:<20}, len={}, file={}".format("Load audio from file", audio.get_len_s(), audio.filename))
        else: # load audio from file
            if (idx in self.cached_XY) and (not self.transform): 
                # if (1) audio has been processed, and (2) we don't need data augumentation,
                # then, we don't need audio data at all. Instead, we only need features from self.cached_XY
                pass 
            else:
                filename=self.files_name[idx]
                audio = AudioClass(filename=filename)
        
        # -- Compute features
        read_features_from_cache = (not self.bool_cache_XY) and (idx in self.cached_XY) and (not self.transform)
        
        # Read features from cache: 
        #   If already computed, and no augmentatation (transform), then read from cache
        if read_features_from_cache:
            X, Y = self.cached_XY[idx]
            
        # Compute features:
        #   if (1) not loaded, or (2) need new transform
        else: 
            # Do transform (augmentation)        
            if self.transform:
                audio = self.transform(audio)
                # self.transform(audio) # this is also good. Transform (Augment) is done in place.

            # Compute mfcc feature
            audio.compute_mfcc(n_mfcc=12) # return mfcc
            
            # Compose X, Y
            X = torch.tensor(audio.mfcc.T, dtype=torch.float32) # shape=(time_len, feature_dim)
            Y = self.files_label[idx]
            
            # Cache 
            if self.bool_cache_XY:
                self.cached_XY[idx] = (X, Y)
            
        # print("{:>20}, len={:.3f}s, file={}".format("After transform", audio.get_len_s(), audio.filename))
        # timer.report_time(event="Load audio", prefix='\t')
        return (X, Y)
    
class AudioClass(object):
    def __init__(self, 
                 data=None, sample_rate=None, filename=None,
                 n_mfcc=12):
        if filename:
            self.data, self.sample_rate = lib_io.read_audio(filename, dst_sample_rate=None)
        elif (len(data) and sample_rate):
            self.data, self.sample_rate = data, sample_rate
        else:
            assert 0, "Invalid input. Use keyword to input either (1) filename, or (2) data and sample_rate"
            
        self.mfcc = None
        self.n_mfcc = n_mfcc # feature dimension of mfcc 
        self.mfcc_image = None 
        self.mfcc_histogram = None
        
        # Record info of original file
        self.filename = filename
        self.original_length = len(self.data)

    def get_len_s(self): # audio length in seconds
        return len(self.data)/self.sample_rate
    
    def _check_and_compute_mfcc(self):
        if self.mfcc is None:
            self.compute_mfcc()
    
    def resample(self, new_sample_rate):
        self.data = librosa.core.resample(self.data, self.sample_rate, new_sample_rate)
        self.sample_rate = new_sample_rate
        
    def compute_mfcc(self, n_mfcc=None):
        # https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
       
        # Check input
        if n_mfcc is None:
            n_mfcc = self.n_mfcc
        if self.n_mfcc is None:
            self.n_mfcc = n_mfcc
            
        # Compute
        self.mfcc = lib_proc_audio.compute_mfcc(self.data, self.sample_rate, n_mfcc)
    
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
        ''' Remove the silence at the beginning of the audio data. '''
         
        l0 = len(self.data) / self.sample_rate
        
        func = lib_proc_audio.remove_silent_prefix_by_freq_domain
        self.data, self.mfcc = func(
            self.data, self.sample_rate, self.n_mfcc, 
            threshold, padding_s, 
            return_mfcc=True
        )
        
        l1 = len(self.data) / self.sample_rate
        print(f"Audio after removing silence: {l0} s --> {l0} s")
        
    # --------------------------- Plotting ---------------------------
    def plot_audio(self, plt_show=False):
        lib_plot.plot_audio(self.data, self.sample_rate)
        if plt_show: plt.show()
            
    def plot_mfcc(self, method='librosa', plt_show=False):
        self._check_and_compute_mfcc()
        lib_plot.plot_mfcc(self.mfcc, self.sample_rate, method)
        if plt_show: plt.show()

    def plot_mfcc_histogram(self, plt_show=False):
        if self.mfcc_histogram is None:
            self.compute_mfcc_histogram()
            
        lib_plot.plot_mfcc_histogram(
            self.mfcc_histogram, *self.args_mfcc_histogram)
        if plt_show: plt.show()

    def plot_mfcc_image(self, plt_show=False):
        if self.mfcc_image is None:
            self.compute_mfcc_image()
        plt.show(self.mfcc_img)
        plt.title("mfcc image")
        if plt_show: plt.show()

    # --------------------------- Input / Output ---------------------------
    def write_to_file(self, filename):
        lib_io.write_audio(filename, self.data, self.sample_rate)
    
    def play_audio(self):
        lib_io.play_audio(data=self.data, sample_rate=self.sample_rate)
        
def synthesize_audio(
        text, sample_rate=16000, 
        lang='en', tmp_filename = ".tmp_audio_from_SynthesizedAudio.wav",
        PRINT=False):
        
    # Create audio
    assert lang in ['en', 'en-uk', 'en-au', 'en-in'] # 4 types of acsents to choose
    if PRINT: print("Synthesizing audio ...", end=' ')
    tts = gTTS(text=text, lang=lang)
    
    # Save to file and load again
    tts.save(tmp_filename)
    data, sample_rate = librosa.load(tmp_filename) # has to be read by librosa, not soundfile
    subprocess.call(["rm", tmp_filename])
    if PRINT: print("Done!")

    # Convert to my audio class
    audio = AudioClass(data=data, sample_rate=sample_rate)
    audio.resample(sample_rate)
    
    return audio


def test_Class_AudioData():
    audio = AudioClass(filename="test_data/audio_1.wav")
    audio.plot_audio()
    audio.plot_mfcc()
    audio.plot_mfcc_histogram()
    
    plt.show()
    # audio.play_audio()

def test_synthesize_audio():
    audio = synthesize_audio("none", PRINT=True)
    audio.play_audio()
    audio.write_to_file("synthesized_audio.wav")
    
def main():
    # test_Class_AudioData()
    test_synthesize_audio()

if __name__ == "__main__":
    main()