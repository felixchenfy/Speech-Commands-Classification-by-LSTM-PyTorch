
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

import torch 
import torch.nn as nn

if 1: # my lib
    import utils.lib_commons as lib_commons
    from utils.lib_datasets import AudioDataset
    from utils.lib_ml import split_train_eval_test
    import utils.lib_rnn as lib_rnn


# ------------------------- Arguments
args = types.SimpleNamespace()
args.input_size = 12  # In a sequency of features, the feature dimensions == input_size
args.batch_size = 1
args.hidden_size = 64
args.num_layers = 3
args.num_classes = 10
args.num_epochs = 15
args.learning_rate = 0.0005
args.weight_decay = 0.00
args.classes_txt = "classes.txt" 
args.data_folder = "data/data_train/"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Settings -------------------------- 
IF_TRAIN_MODEL = True
IF_LOAD_FROM_PRETRAINED = not IF_TRAIN_MODEL
SAVE_MODEL_NAME = 'models/model.ckpt'

# LOAD_PRETRAINED_PATH = 'models/rnn_0510_ep14_ac98.ckpt'
# LOAD_PRETRAINED_PATH = 'models/rnn_0512_ep08_ac99.ckpt'
LOAD_PRETRAINED_PATH = 'models/rnn_0512_ep11_ac98.ckpt'
    
# -------------------------- Torch dataset class

# Get data filenames and labels
files_name, files_label = AudioDataset.load_filenames_and_labels(
    args.data_folder, args.classes_txt)

# Train/Eval/Test split
tr_X, tr_Y, ev_X, ev_Y, te_X, te_Y = split_train_eval_test(
    X=files_name, Y=files_label, ratios=[0.8, 0.1, 0.1], dtype='list')

train_dataset = AudioDataset(
    files_name=tr_X, files_label=tr_Y, transform=None)
    
eval_dataset = AudioDataset(
    files_name=ev_X, files_label=ev_Y, transform=None)

test_dataset = AudioDataset(
    files_name=te_X, files_label=te_Y, transform=None)

# -------------------------- Torch data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size, 
                                           shuffle=True)

eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                            batch_size=args.batch_size, 
                                            shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size, 
                                            shuffle=False)

# --------------------------------------------------------------
# Create model
model = lib_rnn.create_RNN_model(args, device)

# Start training
if IF_TRAIN_MODEL:
    lib_rnn.train_model(model, args, train_loader, eval_loader, SAVE_MODEL_NAME)

# Eval the model
if IF_LOAD_FROM_PRETRAINED:
    model.load_state_dict(torch.load(LOAD_PRETRAINED_PATH))

model.eval()    

with torch.no_grad():
    lib_rnn.evaluate_model(model, test_loader)