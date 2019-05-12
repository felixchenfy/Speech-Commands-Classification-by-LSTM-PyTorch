

import numpy as np 
import time
from mylib.mylib_clf import *
# from mylib.mylib_plot import *
from mylib.mylib_io import *
# from mylib.mylib_commons import *
from mylib.mylib_rnn import *

import torch 
import torch.nn as nn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# -------------------------- Settings
IF_TRAIN_MODEL = False
IF_LOAD_FROM_PRETRAINED = not IF_TRAIN_MODEL
SAVE_MODEL_NAME = 'models/model.ckpt'

# LOAD_PRETRAINED_PATH = 'models/model_039.ckpt'
# LOAD_PRETRAINED_PATH = 'models/model_024.ckpt'
LOAD_PRETRAINED_PATH = 'models/good_model_ep14_ac98.ckpt'

# -------------------------- Load data
train_X = read_list('train_X.csv') # list[list]
train_Y = read_list('train_Y.csv') # list
classes = read_list("classes.csv") # list
tr_X, ev_X, tr_Y, ev_Y = split_data(train_X, train_Y, USE_ALL=False, dtype='list')


te_X = read_list('test_X.csv') # list[list]
te_Y = read_list('test_Y.csv') # list

# -------------------------- Torch dataset class
train_dataset = AudioDataset(
    tr_X, tr_Y, input_size,
    )

eval_dataset = AudioDataset(
    ev_X, ev_Y, input_size,
    )

test_dataset = AudioDataset(
    te_X, te_Y, input_size,
    )

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