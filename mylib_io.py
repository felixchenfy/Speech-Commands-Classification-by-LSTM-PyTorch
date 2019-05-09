
import numpy as np 
import matplotlib.pyplot as plt 
import soundfile as sf
import librosa
import librosa.display
import json

def read_audio(filename, PRINT=False):
    data, sample_rate = sf.read(filename)
    if PRINT:
        print("Read audio file: {}. Audio len = {:.2}s, sample rate = {}, num points = {}".format(
            filename, data.size / sample_rate, sample_rate, data.size))
    return data, sample_rate

def write_audio(filename, data, sample_rate):
    sf.write(filename, data, sample_rate)

def write_list(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)
        # What's in file: [[2, 3, 5], [7, 11, 13, 15]]

def load_list(filename):
    with open(filename) as f:
        data = json.load(f)
    return data