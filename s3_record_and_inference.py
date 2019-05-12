

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
classes = read_list("classes.csv")

MODEL_TO_USE = ["sklearn", "rnn"][1]

if MODEL_TO_USE == "sklearn": # sklearn
    # path = './models/m1.pickle'
    path = './models/good_model2.pickle'
    with open(path, 'rb') as f:
        model2 = pickle.load(f)

elif MODEL_TO_USE == "rnn": # RNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device for RNN: ", device)
    LOAD_PRETRAINED_PATH = 'models/good_model_ep14_ac98.ckpt'
    model2 = create_RNN_model(LOAD_PRETRAINED_PATH, device)
    if 0: # random test
        label = model2.predict(np.random.random((66, 8)))
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
    while True:
        tprinter.print("Usage: keep pressing down 'R' to record audio", T_gap=20)

        board.update_key_state()
        if board.has_just_pressed():

            # start recording
            recorder.start_record(folder='./data_tmp/') 

            # wait until key release
            while not board.has_just_released():
                board.update_key_state()
                time.sleep(0.001)

            # stop recording
            recorder.stop_record()

            # Load audio
            data, sample_rate = read_audio(recorder.filename)
            
            data = remove_data_prefix(data, sample_rate)
            features = data_to_features(data, sample_rate)
            # play_audio(data=data, sample_rate=sample_rate)

            # Predict
            if MODEL_TO_USE == "sklearn":
                X = np.ravel(features)
            elif MODEL_TO_USE == "rnn":
                X = features
            predicted_idx = model2.predict(X)
            predicted_label = classes[predicted_idx]
            print("\nWord labels: {}".format(classes))
            print("\nPredicted label: {}".format(predicted_label))
            
            # reset for better printing
            print("\n")
            tprinter.reset()

        time.sleep(0.1)