
if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    
import numpy as np 
import cv2
import librosa
import matplotlib.pyplot as plt 
from collections import namedtuple
import types
import time 

import torch 
import torch.nn as nn

if 1: # my lib
    import utils.lib_commons as lib_commons
    import utils.lib_rnn as lib_rnn
    import utils.lib_augment as lib_augment
    import utils.lib_datasets as lib_datasets
    import utils.lib_ml as lib_ml
    import utils.lib_io as lib_io
    from utils.lib_record_audio import *


# ---------------------------------------------
# Settings ------------------------------------

save_audio_to = './data/data_tmp/'
classes = lib_io.read_list("config/classes.names")
load_model_from = 'models_good/my.ckpt'
DO_INFERENCE = True

# ---------------------------------------------
# load classifier model -----------------------

def setup_classifier():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device for RNN: ", device)
    
    args = lib_rnn.set_default_args()
    classifier_model = lib_rnn.create_RNN_model(args, load_model_from)
    
    if 0: # random test
        label = classifier_model.predict(np.random.random((66, 12)))
        print("Label of a random feature: ", label, ", label's data type = ", type(label))
        exit("Complete test.")
    return classifier_model

if DO_INFERENCE:
    classifier_model = setup_classifier()
    
# start record audio ---------------------------------------------
if __name__ == '__main__':
    
    # Start keyboard listener
    recording_state = Value('i', 0)
    board = KeyboardMonitor(recording_state, PRINT=False)
    board.start_listen(run_in_new_thread=True)

    # Set up audio recorder
    recorder = AudioRecorder()

    # Others
    tprinter = TimerPrinter() # for print

    # Start loop
    cnt_voice = 0
    while True:
        tprinter.print("Usage: keep pressing down 'R' to record audio", T_gap=20)

        board.update_key_state()
        if board.has_just_pressed():
            cnt_voice += 1
            print("Record {}th voice".format(cnt_voice))
            
            # start recording
            recorder.start_record(folder=save_audio_to) 

            # wait until key release
            while not board.has_just_released():
                board.update_key_state()
                time.sleep(0.001)

            # stop recording
            recorder.stop_record()

            # Do inference
            if DO_INFERENCE:

                # Load audio
                audio = lib_datasets.AudioClass(filename=recorder.filename)
                audio.compute_mfcc()
                
                X = audio.mfcc.T 
                
                predicted_idx = classifier_model.predict(X)
                predicted_label = classes[predicted_idx]
                
                print("\nAll word labels: {}".format(classes))
                print("\nPredicted label: {}".format(predicted_label))

                # Play the video of the predicted label
                lib_io.play_audio(filename="data/examples/" + predicted_label + ".wav")
                
            # reset for better printing
            print("\n")
            tprinter.reset()
        
        time.sleep(0.1)