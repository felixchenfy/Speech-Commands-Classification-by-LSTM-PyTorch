
import numpy as np 
import cv2

import librosa
import librosa.display


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

def remove_prefix(mfcc, threshold=0, padding=15):
    # threshold: if voice is larger than this, the speech is truly begin
    # padding: pad some noise at the left
    voices = mfcc[0] + mfcc[1] # 120 + -50
    start_idx = np.argmax(voices>threshold)
    start_idx = max(0, start_idx - padding)
    return mfcc[:, start_idx:]

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

def calc_histogram(mfcc, bins=10, binrange=(-50, 200)):
    def calc_hist(row):
        hist, bin_edges = np.histogram(mfcc[row], bins=bins, range=binrange)
        return hist/mfcc.shape[1]
    features = []
    for row in range(mfcc.shape[0]):
        features += [calc_hist(row)]
    return np.array(features)

def data_to_features(data, sample_rate):
    # Extract MFCC features
    # https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
    mfcc = librosa.feature.mfcc(
        y=data,
        sr=sample_rate,
        n_mfcc=8)
    # print("features shape = ", mfcc.shape)

    mfcc = remove_prefix(mfcc)
    features = calc_histogram(mfcc)
    # mfcc = add_padding(mfcc)
    # mfcc = mfcc_to_image(mfcc)
    return features 