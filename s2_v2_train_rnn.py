

import numpy as np 
import time
from utils.lib_io import *
from utils.lib_rnn import *

import torch 
import torch.nn as nn

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

# Load data -------------------------- 
train_X = read_list('data/train_X.csv') # list[list]
train_Y = read_list('data/train_Y.csv') # list
if 1: # Test1 Dataset: 99% accu
    test_X = read_list('data/test_X.csv') # list[list]
    test_Y = read_list('data/test_Y.csv') # list
else: # Test2 Dataset: 20% accu
    test_X = read_list('data/test2_X.csv') # list[list]
    test_Y = read_list('data/test2_Y.csv') # list
classes = read_list("classes.csv") # list

if 1: # evaluation set from part of the training set 
    # (Use this if there is no data augmentation. Because we want the train and eval to be different)
    train_X, eval_X, train_Y, eval_Y = split_data(train_X, train_Y, eval_size=0.1, USE_ALL=False, dtype='list')
else: # evaluation set same as the test set
    eval_X, eval_Y = test_X, test_Y 

# -------------------------- Torch dataset class
train_dataset = AudioDataset(train_X, train_Y, input_size,)
eval_dataset = AudioDataset(eval_X, eval_Y, input_size,)
test_dataset = AudioDataset(test_X, test_Y, input_size,)

# -------------------------- Torch data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# --------------------------------------------------------------
# Create model
model = RNN(input_size, hidden_size, num_layers, num_classes, device).to(device)


# Start training
if IF_TRAIN_MODEL:
    train_model(model, train_loader, eval_loader, SAVE_MODEL_NAME)

# Eval the model

if IF_LOAD_FROM_PRETRAINED:
    model.load_state_dict(torch.load(LOAD_PRETRAINED_PATH))

model.eval()    

with torch.no_grad():
    evaluate_model(model, test_loader)