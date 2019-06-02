''' Data augmentation on audio.
Written in the form of a set of Classes. 
'''

if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    
import numpy as np 
import cv2

import librosa

# Import all my libraries
import utils.lib_io as lib_io
import utils.lib_commons as lib_commons
from utils.lib_datasets import AudioClass
import utils.lib_ml as lib_ml
import utils.lib_rnn as lib_rnn

# ----------------------------------------------------------------------

def rand_uniform(bound, size=None):
    l, r = bound[0], bound[1]
    return np.random.uniform(l, r, size=size)

def is_list_or_tuple(obj):
    return isinstance(obj, list) or isinstance(obj, tuple)

def to_tuple(val, left_bound=None):
    if isinstance(val, tuple):
        return val 
    if isinstance(val, list):
        return (val[0], val[1])
    else:
        if left_bound is None:
            left_bound = -val
            assert val>=0, "should be >0, so that (-val, val)"
            return (-val, val)
        else:
            return (left_bound, val)
    
class Augmenter(object):
    ''' A wrapper for a serials of transformations '''
    
    def __init__(self, transforms, prob_to_aug=1):
        self.transforms = transforms 
        self.prob_to_aug = prob_to_aug
        
    def __call__(self, audio):
        if np.random.random()>self.prob_to_aug:
            return audio # not augment. direct return.
        else:
            for transform in self.transforms:
                audio = transform(audio)
            return audio 

    # Add simple noise to audio
    class SimpleNoise(object):
        def __init__(self, intensity=(-0.1, 0.1)):
            self.intensity = to_tuple(intensity)
            
        def __call__(self, audio):
            data = audio.data
            noise = rand_uniform(self.intensity, size=data.shape)
            data = data + noise 
            
            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio

    # Add noise to audio, where noise come from file
    class Noise(object):
        def __init__(self, noise_folder, prob_noise=0.5, intensity=(0, 0.5)):
            self.intensity = to_tuple(intensity)
            
            # Load noises that we will use
            fnames = lib_commons.get_filenames(noise_folder)
            noises = []
            for name in fnames:
                noise, rate = lib_io.read_audio(filename=name)
                noise = librosa.util.normalize(noise) # normalize noise
                noise = self.repeat_pad_to_time(noise, rate, time=10)
                noises.append(noise)
            self.noises = noises
            self.prob_noise = prob_noise
            
        def __call__(self, audio):
            if np.random.random() > self.prob_noise: # no noise
                return audio
            
            data = audio.data
            
            # add noise
            noise = self.randomly_pick_a_noise() * rand_uniform(self.intensity)
            data = data + self.random_crop(noise, len(data))
            data[data>+1] = +1
            data[data<-1] = -1
                
            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio
        
        def randomly_pick_a_noise(self):
            i = np.random.randint(len(self.noises))
            return self.noises[i]
        
        def repeat_pad_to_time(self, noise, sample_rate, time):
            # repeat the noise data, to make it longer than time
            N = time * sample_rate
            n = len(noise)
            if n < N:
                noise = np.tile(noise, 1+(N//n))
            return noise
        
        def random_crop(self, arr, N): 
            # crop N-len subarray from array
            n = len(arr)
            if n < N:
                arr = np.tile(arr, 1+(N//n))
            n = len(arr) # e.g. n=10, N=9, n-N=1, left=[0, 1]=0, right=[9, 10]
            left = np.random.randint(n - N + 1)
            right = left + N
            return arr[left:right]
            ''' Test case:
            for i in range(5):
                x = np.arange(10)
                print(random_crop(x, 10))
            '''
            

    # Shift audio by some time or ratio (>0, to right; <0, to left)
    class Shift(object):
        def __init__(self, time=None, rate=None, keep_size=False):
            self.rate, self.time = None, None
            if rate: # shift time = rate*len(audio)
                self.rate = to_tuple(rate)
            elif time: # shift time = time
                self.time = to_tuple(time)
            else:
                assert 0
            self.keep_size = keep_size
            
        def __call__(self, audio):
            if self.rate:
                rate = rand_uniform(self.rate)
                time = rate * audio.get_len_s() # rate * seconds
            elif self.time:
                time = rand_uniform(self.time) # seconds
                
            n = abs(int(time * audio.sample_rate)) # count shift
            data = audio.data
            assert n < len(data), "Shift amount should be smaller than data length."
            
            # Shift audio data
            if time > 0 or n == 0: # move audio data to right
                data = data[n:]
            else:
                data = data[:-n]
            
            # Add padding
            if self.keep_size:
                z = np.zeros(n)
                if time>0: # pad at left
                    data = np.concatenate((z, data))
                else:
                    data = np.concatenate((data, z))

            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio
         
    

    # Pad zeros randomly at left or right by a time or rate >= 0
    class PadZeros(object):
        def __init__(self, time=None, rate=None):
            self.rate, self.time = None, None
            if rate: # shift time = rate*len(audio)
                self.rate = to_tuple(rate, left_bound=0)
            elif time: # shift time = time
                self.time = to_tuple(time, left_bound=0)
            else:
                assert 0
            
        def __call__(self, audio):
            if self.rate:
                rate = rand_uniform(self.rate)
                time = rate * audio.get_len_s() # rate * seconds
            elif self.time:
                time = rand_uniform(self.time) # seconds
                
            n = abs(int(time * audio.sample_rate)) # count padding
            data = audio.data
            
            # Shift audio data
            if np.random.random() < 0.5:
                data = np.concatenate(( data, np.zeros(n, ) ))
            else:
                data = np.concatenate(( np.zeros(n, ), data ))

            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio
        
    # PlaySpeed audio by a rate (e.g., longer or shorter)
    class PlaySpeed(object):
        def __init__(self, rate=(0.9, 1.1), keep_size=False):
            assert is_list_or_tuple(rate)
            self.rate = rate
            self.keep_size = keep_size
            
        def __call__(self, audio):
            data = audio.data
            rate = rand_uniform(self.rate)
            len0 = len(data) # record original length
            
            # PlaySpeed
            data = librosa.effects.time_stretch(data, rate)
            
            # Pad
            if self.keep_size:
                if len(data)>len0:
                    data = data[:len0]
                else:
                    data = np.pad(data, (0, max(0, len0 - len(data))), "constant")
            
            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio
        
    # Amplify audio by a rate (e.g., louder or lower)
    class Amplify(object):
        def __init__(self, rate=(0.2, 2)):
            assert is_list_or_tuple(rate)
            self.rate = to_tuple(rate)
            '''
            Test result: For an audio with a median voice,
            if rate=0.2, I could still here it.
            if rate=2, it becomes a little bit loud.
            '''            
        def __call__(self, audio):
            rate = rand_uniform(self.rate)
            data = audio.data * rate
            if rate > 1: # cutoff (Default range for an audio is [-1, 1]).
                # I've experimented this by amplifying an audio, 
                #   saving it file and load again using soundfile library,
                #   and then I found that abs(voice)>1 was cut off to 1.
                data[data>+1] = +1
                data[data<-1] = -1
                
            # return
            audio.data = data
            assert len(audio.data) > 0
            return audio


def test_augmentation_effects():
    from utils.lib_io import read_audio, write_audio, play_audio

    filename = 'test_data/audio_3.wav'
    output_name = 'test_data/tmp_audio.wav'
    audio = AudioClass(filename=filename)
    print("Audio length = ", audio.get_len_s(), " seconds")
    
    if 1: # Augment
        import copy
        for i in range(5):
            print(i)
            
            # Set up augmenter
            aug = Augmenter([
                Augmenter.PadZeros(time=(0.1, 0.3)),
                # Augmenter.Noise(noise_folder="data/noises/", 
                #                 prob_noise=0.6, intensity=(0.1, 0.4)),
                # Augmenter.Shift(rate=0.5, keep_size=False),
                # Augmenter.PlaySpeed(rate=(1.4, 1.5), keep_size=True),
                # Augmenter.Amplify(rate=(2, 2)),
            ])

            # Augment    
            audio_copy = copy.deepcopy(audio)
            aug(audio_copy)
            audio_copy.play_audio()

    audio = audio_copy
    
    # -- Write to file and play. See if its good
    # audio.plot_audio(plt_show=True)
    # audio.write_to_file(output_name)
    
    # -- Load again
    # audio2 = AudioClass(filename=output_name)
    # audio2.plot_audio(plt_show=True)
    
def main():
    test_augmentation_effects()

if __name__ == "__main__":
    main()
     