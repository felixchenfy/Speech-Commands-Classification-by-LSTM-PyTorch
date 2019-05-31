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
# ----------------------------------------------------------------------

def rand_uniform(bound, size=None):
    l, r = bound[0], bound[1]
    return np.random.uniform(l, r, size=size)

def is_list_or_tuple(obj):
    return isinstance(obj, list) or isinstance(obj, tuple)

def to_tuple(val):
    if isinstance(val, tuple):
        return val 
    if isinstance(val, list):
        return (val[0], val[1])
    else:
        assert val>=0, "should be >0, so that (-val, val)"
        return (-val, val)
        
class Augmenter(object):
    ''' A wrapper for a serials of transformations '''
    sample_rate = -1
    
    def __init__(self, sample_rate):
        Augmenter.sample_rate = sample_rate
        
    def set_transforms(self, transforms):
        self.transforms = transforms 
        
    def __call__(self, audio):
        for transform in self.transforms:
            audio = transform(audio)
        return audio 

    # Add noise to audio
    class Noise(object):
        def __init__(self, intensity=(-0.1, 0.1)):
            self.intensity = to_tuple(intensity)
            
        def __call__(self, audio):
            noise = rand_uniform(self.intensity, size=audio.shape)
            return audio + noise 
        
    # Shift audio by some time
    class Shift(object):
        def __init__(self, time=(-0.2, 0.2), keep_size=False):
            self.time = to_tuple(time)
            self.keep_size = keep_size
            
        def __call__(self, audio):
            time = rand_uniform(self.time)
            n = abs(int(time * Augmenter.sample_rate)) # count shift
            
            # Shift audio
            if time > 0: # move audio to right
                audio = audio[n:]
            else:
                audio = audio[:-n]
            
            # Add padding
            if self.keep_size:
                z = np.zeros(n)
                if time>0: # pad at left
                    audio = np.concatenate((z, audio))
                else:
                    audio = np.concatenate((audio, z))
            return audio
         
    
    # Stretch audio by a rate (e.g., longer or shorter)
    class Stretch(object):
        def __init__(self, rate=(0.9, 1.1), keep_size=False):
            assert is_list_or_tuple(rate)
            self.rate = rate
            self.keep_size = keep_size
            
        def __call__(self, audio):
            rate = rand_uniform(self.rate)
            len0 = len(audio) # record original length
            
            # Stretch
            audio = librosa.effects.time_stretch(audio, rate)
            
            # Pad
            if self.keep_size:
                if len(audio)>len0:
                    audio = audio[:len0]
                else:
                    audio = np.pad(audio, (0, max(0, len0 - len(audio))), "constant")
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
            return audio * rate 

def test_augmentation_effects():
    from utils.lib_io import read_audio, write_audio, play_audio
    
    filename = 'test_data/audio_1.wav'
    output_name = 'test_data/tmp_audio.wav'
    audio, sample_rate = read_audio(filename)
    
    if 1: # Augment
        
        # Set up augmenter
        aug = Augmenter(sample_rate)
        aug.set_transforms([
            aug.Noise(intensity=0),
            # aug.Shift(time=0.5, keep_size=True),
            # aug.Stretch(rate=(0.4, 1), keep_size=True),
            aug.Amplify(rate=(2, 2)),
        ])

        # Augment    
        audio = aug(audio)

    # Write to file and play. See if its good
    write_audio(output_name, audio, sample_rate)
    play_audio(output_name)

def main():
    test_augmentation_effects()

if __name__ == "__main__":
    main()
     