''' Functions for processing audio and audio mfcc features '''

if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    
import numpy as np 
import cv2
import librosa
import warnings


# How long is a mfcc data frame ?        
MFCC_RATE = 50 # TODO: It's about 1/50 s, I'm not sure.

# ----------------------------------------------------------------------
if 1: # Basic maths
    def rand_num(val): # random [-val, val]
        return (np.random.random()-0.5)*2*val
    
    def integral(arr):
        ''' sums[i] = sum(arr[0:i]) '''
        sums = [0]*len(arr)
        for i in range(1, len(arr)):
            sums[i] = sums[i-1] + arr[i]
        return sums
    
    def filter_by_average(arr, N):
        ''' Average filtering a data sequency by window size of N '''
        cumsum = np.cumsum(np.insert(arr, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / N  

# ----------------------------------------------------------------------

if 1: # Time domain processings
    def filter_audio_by_average(audio, sample_rate, window_seconds):
        ''' Replace audio[j] with np.mean(audio[i:j]) '''
        ''' 
        Output:
            audio with same length
        '''
        
        window_size = int(window_seconds * sample_rate)
        
        if 1: # Compute integral arr, then find interval sum
            sums = integral(audio)
            res = [0]*len(audio)
            for i in range(1, len(audio)):
                prev = max(0, i - window_size)
                res[i] = (sums[i] - sums[prev]) / (i - prev)
        else: # Use numpy built-in
            filter_by_average(audio, window_size)
            
        return res


    def remove_silent_prefix_by_freq_domain(
            audio, sample_rate, n_mfcc, threshold, padding_s=0.2,
            return_mfcc=False):
        
        # Compute mfcc, and remove silent prefix
        mfcc_src = compute_mfcc(audio, sample_rate, n_mfcc)
        mfcc_new = remove_silent_prefix_of_mfcc(mfcc_src, threshold, padding_s)

        # Project len(mfcc) to len(audio)
        l0 = mfcc_src.shape[1]
        l1 = mfcc_new.shape[1]
        start_idx = int(audio.size * (1 - l1 / l0))
        new_audio = audio[start_idx:]
        
        # Return
        if return_mfcc:        
            return new_audio, mfcc_new
        else:
            return new_audio
            
    def remove_silent_prefix_by_time_domain(
            audio, sample_rate, threshold=0.25, window_s=0.1, padding_s=0.2):
        ''' Remove silent prefix of audio, by checking voice intensity in time domain '''
        ''' 
            threshold: voice intensity threshold. Voice is in range [-1, 1].
            window_s: window size (seconds) for averaging.
            padding_s: padding time (seconds) at the left of the audio.
        '''
        window_size = int(window_s * sample_rate)
        trend = filter_by_average(abs(audio), window_size)
        start_idx = np.argmax(trend > threshold)
        start_idx = max(0, start_idx + window_size//2 - int(padding_s*sample_rate))
        return audio[start_idx:]



# ----------------------------------------------------------------------
if 1: # Frequency domain processings (on mfcc)
    
    def compute_mfcc(audio, sample_rate, n_mfcc=12):
        # Extract MFCC features
        # https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=n_mfcc,   # How many mfcc features to use? 12 at most.
                        # https://dsp.stackexchange.com/questions/28898/mfcc-significance-of-number-of-features
        )
        return mfcc 

    def remove_silent_prefix_of_mfcc(mfcc, threshold, padding_s=0.2):
        '''
        threshold:  Audio is considered started at t0 if mfcc[t0] > threshold
        padding: pad data at left (by moving the interval to left.)
        '''
        
        # Set voice intensity
        voice_intensity = mfcc[1]
        if 1:
            voice_intensity += mfcc[0]
            threshold += -100
        
        # Threshold to find the starting index
        start_indices = np.nonzero(voice_intensity > threshold)[0]
        
        # Return sliced mfcc
        if len(start_indices) == 0:
            warnings.warn("No audio satisifies the given voice threshold.")
            warnings.warn("Original data is returned")
            return mfcc
        else:
            start_idx = start_indices[0]
            # Add padding
            start_idx = max(0, start_idx - int(padding_s * MFCC_RATE))
            return mfcc[:, start_idx:]
    
    def mfcc_to_image(mfcc, row=200, col=400,
                    mfcc_min=-200, mfcc_max=200):
        ''' Convert mfcc to an image by converting it to [0, 255]'''
        
        # Rescale
        mfcc_img = 256 * (mfcc - mfcc_min) / (mfcc_max - mfcc_min)
        
        # Cut off
        mfcc_img[mfcc_img>255] = 255
        mfcc_img[mfcc_img<0] = 0
        mfcc_img = mfcc_img.astype(np.uint8)
        
        # Resize to desired size
        img = cv2.resize(mfcc_img, (col, row))
        return img
    
    def pad_mfcc_to_fix_length(mfcc, goal_len=100, pad_value=-200):
        feature_dims, time_len = mfcc.shape
        if time_len >= goal_len:
            mfcc = mfcc[:, :-(time_len - goal_len)] # crop the end of audio
        else:
            n = goal_len - time_len
            zeros = lambda n: np.zeros((feature_dims, n)) + pad_value
            if 0: # Add paddings to both side
                n1, n2 = n//2, n - n//2
                mfcc = np.hstack(( zeros(n1), mfcc, zeros(n2)))
            else: # Add paddings to left only
                mfcc = np.hstack(( zeros(n), mfcc))
        return mfcc
    
    def calc_histogram(mfcc, bins=10, binrange=(-50, 200), col_divides=5): 
        ''' Function:
                Divide mfcc into $col_divides columns.
                For each column, find the histogram of each feature (each row),
                    i.e. how many times their appear in each bin.
            Return:
                features: shape=(feature_dims, bins*col_divides)
        '''
        feature_dims, time_len = mfcc.shape
        cc = time_len//col_divides # cols / num_hist = size of each hist
        def calc_hist(row, cl, cr):
            hist, bin_edges = np.histogram(mfcc[row, cl:cr], bins=bins, range=binrange)
            return hist/(cr-cl)
        features = []
        for j in range(col_divides):
            row_hists = [calc_hist(row, j*cc, (j+1)*cc) for row in range(feature_dims) ]
            row_hists = np.vstack(row_hists) # shape=(feature_dims, bins)
            features.append(row_hists)
        features = np.hstack(features)# shape=(feature_dims, bins*col_divides)
        return features 
    
        # if 0: # deprecated code
        #     for j in range(col_divides):
        #         row_hists = [calc_hist(row, j*cc, (j+1)*cc) for row in range(feature_dims) ]
        #         row_hists = np.vstack(row_hists) # shape=(feature_dims, bins)
        #         features += [row_hists.reshape((1, -1))] # shape=(feature_dims * bins, 1)
        #     features = np.vstack(features).ravel() # shape=(feature_dims * bins * col_divides, )
        #     return features 

if __name__ == "__main__":
    # Test cases are in "lib_datasets.py"
    pass
