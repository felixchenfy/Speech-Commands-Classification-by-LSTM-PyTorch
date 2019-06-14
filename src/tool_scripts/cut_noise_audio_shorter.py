# Randomly cut audios shorter

if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    
import glob 
import os 
import soundfile as sf
import librosa
import utils.lib_datasets as lib_datasets 
import utils.lib_augment as lib_augment
import utils.lib_commons as lib_commons
import copy 

def get_filenames(folder, file_type):
    return glob.glob(folder + "/" + file_type)

def reset_audio_sample_rate(filename, dst_sample_rate=16000):
    data, sample_rate = sf.read(filename) 
    if (dst_sample_rate is not None) and (dst_sample_rate != sample_rate):
        data = librosa.core.resample(data, sample_rate, dst_sample_rate)
        sample_rate = dst_sample_rate
    sf.write(filename, data, sample_rate)
    print(f"Reset sample rate to {dst_sample_rate} for the file: {filename}")
    
folder = "./data/noises/"
fnames = get_filenames(folder, file_type="*.wav")

Aug = lib_augment.Augmenter
aug = Aug([
        Aug.Crop(time=(0.6, 2.0)),
        Aug.PadZeros(time=(0, 0.3)),
        Aug.PlaySpeed(rate=(0.7, 1.5), keep_size=False),
        Aug.Amplify(rate=(0.6, 1.5)),
    ])
    
for name in fnames:
    audio0 = lib_datasets.AudioClass(filename=name)
    for i in range(5):
        print(i)
        audio = copy.deepcopy(audio0)
        aug(audio)
        name_new = lib_commons.add_idx_suffix(name, i).split('/')[-1]
        audio.write_to_file("data/data_tmp/" + name_new)
        audio.play_audio()
        
