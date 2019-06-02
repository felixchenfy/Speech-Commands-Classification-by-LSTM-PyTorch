
if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    
import numpy as np 
import torch 

if 1: # my lib
    import utils.lib_io as lib_io
    import utils.lib_commons as lib_commons
    import utils.lib_datasets as lib_datasets
    import utils.lib_augment as lib_augment
    import utils.lib_ml as lib_ml
    import utils.lib_rnn as lib_rnn

# ------------------------------------------------------------------------

# Arguments ------------------------- 
args = lib_rnn.set_default_args()
args.classes_txt = "config/classes_kaggle.names" 
args.load_model_from="./models/kaggle_accu_914/model_025.ckpt"

# Create model
model = lib_rnn.create_RNN_model(args, args.load_model_from)
