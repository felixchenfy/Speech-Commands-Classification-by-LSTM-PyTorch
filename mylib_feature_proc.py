
import numpy as np 
import cv2

import librosa
import librosa.display


def rand_num(val): # [-val, val]
    return (np.random.random()-0.5)*2*val 

def filter_audio_by_average(data, sample_rate, window_seconds):
    # For en element j, use the mean(data[i:j]) to replace it

    sums = [0]*len(data)
    for i in range(1, len(data)):
        sums[i] = sums[i-1] + data[i]

    window_size = int(window_seconds*sample_rate)
    res = [0]*len(data)
    for i in range(1, len(data)):
        prev = max(0, i - window_size)
        res[i] = (sums[i] - sums[prev]) / (i - prev)
    return res

def remove_prefix(mfcc, threshold=80, padding=0):
    # threshold: if voice is larger than this, the speech is truly begin
    # padding: pad some noise at the left
    if 0:
        voices = mfcc[0]
    else:
        voices = mfcc[0] + mfcc[1]
        threshold = 10
    start_idx = np.argmax(voices>threshold)
    start_idx = max(0, start_idx - padding)
    return mfcc[:, start_idx:]

def remove_data_prefix(data, sample_rate, threshold=0.25, padding=0.1):

    if 0: # Threshold on time domain
        def running_mean(x, N):
            cumsum = np.cumsum(np.insert(x, 0, 0)) 
            return (cumsum[N:] - cumsum[:-N]) / N

        window_size = int(0.01*sample_rate)
        trend = running_mean(abs(data), window_size)
        start_idx = np.argmax(trend > threshold)
        start_idx = max(0, start_idx + window_size//2 - int(padding*sample_rate))
        return data[start_idx:]

    else: # Threshold on frequency domain
        mfcc0 = compute_mfcc(data, sample_rate)
        l0 = mfcc0.shape[1]
        mfcc1 = remove_prefix(mfcc0)
        l1 = mfcc1.shape[1]
        
        start_idx = int(data.size * (1 - l1 / l0))
        start_idx = max(0, start_idx - int(padding*sample_rate))
        return data[start_idx:]

def add_padding(mfcc, goal_len=100, vmin=-200):
    rows, cols = mfcc.shape
    if cols >= goal_len:
        mfcc = mfcc[:, :-(cols - goal_len)] # crop the end of data
    else:
        n = goal_len - cols
        zeros = lambda n: np.zeros((rows, n)) + vmin
        if 0: # Add paddings to both side
            n1, n2 = n//2, n - n//2
            mfcc = np.hstack(( zeros(n1), mfcc, zeros(n2)))
        else: # Add paddings to left only
            mfcc = np.hstack(( zeros(n), mfcc))
    return mfcc

def scale_mfcc(mfcc, vmin=-200, vmax=200):
    mfcc = 256 * (mfcc - vmin) / (vmax - vmin)
    mfcc[mfcc>255] = 255
    mfcc[mfcc<0] = 0
    mfcc = mfcc.astype(np.uint8)
    return mfcc 

def mfcc_to_image(mfcc, row=200, col=400):
    img = scale_mfcc(mfcc)
    # The image shape might be something like (10, 383)
    # Resize to make it more like an image
    img = cv2.resize(img, (col, row))
    return img

def calc_histogram(mfcc, bins=10, binrange=(-50, 200), col_divides = 3): 
    rows, cols = mfcc.shape
    cc = cols//col_divides # cols / num_hist = size of each hist
    def calc_hist(row, cl, cr):
        hist, bin_edges = np.histogram(mfcc[row, cl:cr], bins=bins, range=binrange)
        return hist/(cr-cl)
    features = []
    for row in range(rows):
        row_hists = [calc_hist(row, j*cc, (j+1)*cc) for j in range(col_divides)]
        row_hists = np.hstack(row_hists)
        features += [row_hists]
    return np.vstack(features)

def data_augment(data, sample_rate):
    # https://www.kaggle.com/CVxTz/audio-data-augmentation

    def add_noise(data, noise=0.005):
        return data + np.random.random(data.shape) * noise
    
    def shift_data(data, sample_rate, time=0.2):
        d = abs(int(time*sample_rate))
        z = np.zeros(d)
        if time>0:
            return np.concatenate((z, data[:-d]))
        else:
            return np.concatenate((data[d:], z))

    def stretch(data, rate=1):
        input_length = len(data)
        data = librosa.effects.time_stretch(data, rate)
        
        if 0: # Make data the same length as original
            if len(data)>input_length:
                data = data[:input_length]
            else:
                data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

        return data

    def strength(data, rate=1):
        return data * rate 
    
    data = add_noise(data, noise=rand_num(0.005))
    data = shift_data(data, sample_rate, time=rand_num(val=0.1))
    data = stretch(data, rate=1+rand_num(0.4))
    data = strength(data, rate=1+rand_num(0.4))

    return data

def compute_mfcc(data, sample_rate):
    # Extract MFCC features
    # https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
    mfcc = librosa.feature.mfcc(
        y=data,
        sr=sample_rate,
        n_mfcc=8)
    return mfcc 

def data_to_features(data, sample_rate):
    mfcc = compute_mfcc(data, sample_rate)
    # print("features shape = ", mfcc.shape)

    # mfcc = remove_prefix(mfcc)
    mfcc = calc_histogram(mfcc)

    # mfcc = add_padding(mfcc)
    # mfcc = mfcc_to_image(mfcc)
    return mfcc 
