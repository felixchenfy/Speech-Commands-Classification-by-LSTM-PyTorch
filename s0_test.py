
import numpy as np 
import matplotlib.pyplot as plt 
import soundfile as sf

import librosa
import librosa.display

from utils.lib_io import * 
from utils.lib_plot import * 
from utils.lib_feature_proc import *
from utils.lib_commons import *

def test_mfcc_features(filename):
    # Load data
    data, sample_rate = read_audio(filename)
    data = remove_silent_prefix(data, sample_rate)
    data = data_augment(data, sample_rate)
    # play_audio(data=data, sample_rate=sample_rate)

    # Proc data
    features = data_to_features(data, sample_rate)
    cv2.imwrite('data/data_tmp/audio_tmp.jpg', cv2_image_f2i(features))

    # Plot mfccs
    if 1:
        fig = plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1) # plot data
        plot_audio(data, sample_rate)
        plt.subplot(2, 1, 2) # plot mfcc
        # plot_mfcc(features, sample_rate)
        plot_mfcc(features, sample_rate, method="hist")
        plt.show()
    if 0:
        plot_mfcc(features, sample_rate, method="hist")
        plt.show()
    write_audio('data/data_tmp/audio_tmp.wav', data, sample_rate=sample_rate//1)

    
filename = '0ac15fe9_nohash_0.wav'
# filename = 'data/data_train/one/audio_05-08-23-02-56-132dgaljnm_.wav'

test_mfcc_features(filename)

# filename = 'audio_2.wav'
# test_mfcc_features(filename)

# 4 s, 383
# 2.8 s, 271
# -300 200