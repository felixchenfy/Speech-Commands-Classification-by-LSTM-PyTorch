
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


# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------


# Set arguments ------------------------- 
args = lib_rnn.set_default_args()

args.learning_rate = 0.001
args.num_epochs = 25
args.learning_rate_decay_interval = 5 # decay for every 3 epochs
args.learning_rate_decay_rate = 0.5 # lr = lr * rate
args.do_data_augment = True
args.train_eval_test_ratio=[0.9, 0.1, 0.0]
args.data_folder = "data/kaggle/"
args.classes_txt = "config/classes_kaggle.names" 
args.load_weights_from = None

# Dataset -------------------------- 

# Get data's filenames and labels
files_name, files_label = lib_datasets.AudioDataset.load_filenames_and_labels(
    args.data_folder, args.classes_txt)

# if 1: # DEBUG: use only a subset of all data
#     GAP = 1000
#     files_name = files_name[::GAP]
#     files_label = files_label[::GAP]
#     args.num_epochs = 5
    
# Set data augmentation
if args.do_data_augment:
    Aug = lib_augment.Augmenter # rename
    aug = Aug([        
        Aug.Shift(rate=0.2, keep_size=False), 
        Aug.PadZeros(time=(0, 0.3)),
        Aug.Amplify(rate=(0.2, 1.5)),
        # Aug.PlaySpeed(rate=(0.7, 1.3), keep_size=False),
        Aug.Noise(noise_folder="data/noises/", 
                        prob_noise=0.7, intensity=(0, 0.7)),
    ], prob_to_aug=0.8)
else:
    aug = None

# Split data into train/eval/test
tr_X, tr_Y, ev_X, ev_Y, te_X, te_Y = lib_ml.split_train_eval_test(
    X=files_name, Y=files_label, ratios=args.train_eval_test_ratio, dtype='list')
train_dataset = lib_datasets.AudioDataset(files_name=tr_X, files_label=tr_Y, transform=aug)
eval_dataset = lib_datasets.AudioDataset(files_name=ev_X, files_label=ev_Y, transform=None)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=True)

# Create model and train -------------------------------------------------
model = lib_rnn.create_RNN_model(args, load_weights_from=args.load_weights_from) # create model
lib_rnn.train_model(model, args, train_loader, eval_loader)
