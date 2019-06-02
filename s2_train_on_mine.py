
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

args.num_epochs = 15
args.learning_rate = 0.0001
args.train_eval_test_ratio=[0.7, 0.3, 0.0]
args.do_data_augment = True
args.data_folder = "data/data_train/"
args.classes_txt = "config/classes.names" 
args.load_model_from = "models_good/kaggle.ckpt"
args.finetune_model = True # If true, fix all parameters except the fc layer
args.save_model_to = 'models/' # Save model and log file

# Dataset -------------------------- 

# Get data's filenames and labels
files_name, files_label = lib_datasets.AudioDataset.load_filenames_and_labels(
    args.data_folder, args.classes_txt)

if 0: # DEBUG: use only a subset of all data
    GAP = 20
    files_name = files_name[::GAP]
    files_label = files_label[::GAP]
    args.num_epochs = 5
    
# Set data augmentation
if args.do_data_augment:
    Aug = lib_augment.Augmenter # rename
    aug = Aug([        
        Aug.Shift(rate=0.2, keep_size=False), 
        Aug.PadZeros(time=(0, 0.3)),
        Aug.Amplify(rate=(0.5, 1.2)),
        # Aug.PlaySpeed(rate=(0.7, 1.3), keep_size=False),
        # Aug.Noise(noise_folder="data/noises/", prob_noise=0.8, intensity=(0.1, 0.4)),
        #       There is already strong white noise in most of my data. No need to add noise.
    ], prob_to_aug=0.8)
else:
    aug = None

# Split data into train/eval/test
tr_X, tr_Y, ev_X, ev_Y, te_X, te_Y = lib_ml.split_train_eval_test(
    X=files_name, Y=files_label, ratios=args.train_eval_test_ratio, dtype='list')
train_dataset = lib_datasets.AudioDataset(files_name=tr_X, files_label=tr_Y, transform=aug, bool_cache_XY=False)
eval_dataset = lib_datasets.AudioDataset(files_name=ev_X, files_label=ev_Y, transform=None, bool_cache_XY=False)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=True)

# Create model and train -------------------------------------------------
model = lib_rnn.create_RNN_model(args, load_model_from=args.load_model_from) # create model
lib_rnn.train_model(model, args, train_loader, eval_loader)
