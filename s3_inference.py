

import numpy as np 
import time
from mylib.mylib_sklearn import *
from mylib.mylib_plot import *
from mylib.mylib_io import *
from mylib.mylib_feature_proc import *
import pickle
from mylib.mylib_record_audio import *

from mylib.mylib_rnn import RNN, create_RNN_model

import torch 
import torch.nn as nn

# load classifier model ---------------------------------------------
save_audio_to = './data_tmp/'
classes = read_list("classes.csv")

DO_INFERENCE = True

if DO_INFERENCE:

    # Choose a classifier
    
    MODEL_TO_USE = ["sklearn", "rnn"][1]
    print("Using the classifer of: ", MODEL_TO_USE)

    if MODEL_TO_USE == "sklearn": # sklearn
        model_path = './models/sklearn_model2.pickle'
        with open(model_path, 'rb') as f:
            classifier_model = pickle.load(f)

    elif MODEL_TO_USE == "rnn": # RNN
        
        model_path = 'models/rnn_0512_ep11_ac98.ckpt'
        # model_path = 'models/rnn_0512_ep08_ac99.ckpt'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device for RNN: ", device)
        classifier_model = create_RNN_model(model_path, device)
        if 0: # random test
            label = classifier_model.predict(np.random.random((66, 8)))
            print("Label of a random feature: ", label, ", label's data type = ", type(label))

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
                data, sample_rate = read_audio(recorder.filename)
                
                # preprocess and extract features
                data = remove_data_prefix(data, sample_rate)
                features = data_to_features(data, sample_rate)

                # Predict
                if MODEL_TO_USE == "sklearn":
                    X = np.ravel(features)
                elif MODEL_TO_USE == "rnn":
                    X = features
                predicted_idx = classifier_model.predict(X)
                predicted_label = classes[predicted_idx]
                print("\nAll word labels: {}".format(classes))
                print("\nPredicted label: {}".format(predicted_label))

                # Play the video of the predicted label
                # play_audio(filename="data_train/result_is.wav")
                play_audio(filename="data_train/" + predicted_label + ".wav")
                
            # reset for better printing
            print("\n")
            tprinter.reset()
        
        time.sleep(0.1)