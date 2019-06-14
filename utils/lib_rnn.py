if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    
import numpy as np 
import time
import types
import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


if 1: # my lib
    import utils.lib_proc_audio as lib_proc_audio
    import utils.lib_plot as lib_plot
    import utils.lib_io as lib_io
    import utils.lib_commons as lib_commons
    import utils.lib_ml as lib_ml

def set_default_args():
    
    args = types.SimpleNamespace()

    # model params
    args.input_size = 12  # == n_mfcc
    args.batch_size = 1
    args.hidden_size = 64
    args.num_layers = 3

    # training params
    args.num_epochs = 100
    args.learning_rate = 0.0001
    args.learning_rate_decay_interval = 5 # decay for every 5 epochs
    args.learning_rate_decay_rate = 0.5 # lr = lr * rate
    args.weight_decay = 0.00
    args.gradient_accumulations = 16 # number of gradient accums before step
    
    # training params2
    args.load_weights_from = None
    args.finetune_model = False # If true, fix all parameters except the fc layer
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # data
    args.data_folder = "data/data_train/"
    args.train_eval_test_ratio=[0.9, 0.1, 0.0]
    args.do_data_augment = False

    # labels
    args.classes_txt = "config/classes.names" 
    args.num_classes = None # should be added with a value somewhere, like this:
    #                = len(lib_io.read_list(args.classes_txt))

    # log setting
    args.plot_accu = True # if true, plot accuracy for every epoch
    args.show_plotted_accu = False # if false, not calling plt.show(), so drawing figure in background
    args.save_model_to = 'checkpoints/' # Save model and log file
        #e.g: model_001.ckpt, log.txt, log.jpg
    
    return args 

def load_weights(model, weights, PRINT=False):
    # Load weights into model.
    # If param's name is different, raise error.
    # If param's size is different, skip this param.
    # see: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
    
    for i, (name, param) in enumerate(weights.items()):
        model_state = model.state_dict()
        
        if name not in model_state:
            print("-"*80)
            print("weights name:", name) 
            print("RNN states names:", model_state.keys()) 
            assert 0, "Wrong weights file"
            
        model_shape = model_state[name].shape
        if model_shape != param.shape:
            print(f"\nWarning: Size of {name} layer is different between model and weights. Not copy parameters.")
            print(f"\tModel shape = {model_shape}, weights' shape = {param.shape}.")
        else:
            model_state[name].copy_(param)
        
def create_RNN_model(args, load_weights_from=None):
    ''' A wrapper for creating a 'class RNN' instance '''
    
    
    # Update some dependent args
    args.num_classes = len(lib_io.read_list(args.classes_txt)) # read from "config/classes.names"
    args.save_log_to = args.save_model_to + "log.txt"
    args.save_fig_to = args.save_model_to + "fig.jpg"
    
    # Create model
    device = args.device
    model = RNN(args.input_size, args.hidden_size, args.num_layers, args.num_classes, device).to(device)
    
    # Load weights
    if load_weights_from:
        print(f"Load weights from: {load_weights_from}")
        weights = torch.load(load_weights_from)
        load_weights(model, weights)
    
    return model

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device, classes=None):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
        self.classes = classes

    def forward(self, x):
        # Set initial hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device) 
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # shape = (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, x):
        '''Predict one label from one sample's features'''
        # x: feature from a sample, LxN
        #   L is length of sequency
        #   N is feature dimension
        x = torch.tensor(x[np.newaxis, :], dtype=torch.float32)
        x = x.to(self.device)
        outputs = self.forward(x)
        _, predicted = torch.max(outputs.data, 1)
        predicted_index = predicted.item()
        return predicted_index
    
    def set_classes(self, classes):
        self.classes = classes 
    
    def predict_audio_label(self, audio):
        idx = self.predict_audio_label_index(audio)
        assert self.classes, "Classes names are not set. Don't know what audio label is"
        label = self.classes[idx]
        return label

    def predict_audio_label_index(self, audio):
        audio.compute_mfcc()
        x = audio.mfcc.T # (time_len, feature_dimension)
        idx = self.predict(x)
        return idx
    
def evaluate_model(model, eval_loader, num_to_eval=-1):
    ''' Eval model on a dataset '''
    device = model.device
    correct = 0
    total = 0
    for i, (featuress, labels) in enumerate(eval_loader):

        featuress = featuress.to(device) # (batch, seq_len, input_size)
        labels = labels.to(device)

        # Predict
        outputs = model(featuress)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # stop
        if i+1 == num_to_eval:
            break
    eval_accu = correct / total
    print('  Evaluate on eval or test dataset with {} samples: Accuracy = {}%'.format(
        i+1, 100 * eval_accu)) 
    return eval_accu

def fix_weights_except_fc(model):
    not_fix = "fc"
    for name, param in model.state_dict().items():
        if not_fix in name:
            continue
        else:
            print(f"Fix {name} layer", end='. ')
            param.requires_grad = False
    print("")

def train_model(model, args, train_loader, eval_loader):

    device = model.device
    logger = lib_ml.TrainingLog(training_args=args)
    if args.finetune_model:
        fix_weights_except_fc(model)
        
    # -- create folder for saving model
    if args.save_model_to:
        if not os.path.exists(args.save_model_to):
            os.makedirs(args.save_model_to)
            
    # -- Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # -- For updating learning rate
    def update_lr(optimizer, lr):    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # -- Train the model
    total_step = len(train_loader)
    curr_lr = args.learning_rate
    cnt_batches = 0
    for epoch in range(1, 1+args.num_epochs):
        cnt_correct, cnt_total = 0, 0
        for i, (featuress, labels) in enumerate(train_loader):
            cnt_batches += 1

            ''' original code of pytorch-tutorial:
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            # we can see that the shape of images should be: 
            #    (batch_size, sequence_length, input_size)
            '''
            featuress = featuress.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(featuress)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward() # error
            if cnt_batches % args.gradient_accumulations == 0:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # Record result
            _, argmax = torch.max(outputs, 1)
            cnt_correct += (labels == argmax.squeeze()).sum().item()
            cnt_total += labels.size(0)
            
            # Print accuracy
            train_accu = cnt_correct/cnt_total
            if (i+1) % 50 == 0 or (i+1) == len(train_loader):
                print ('Epoch [{}/{}], Step [{}/{}], Loss = {:.4f}, Train accuracy = {:.2f}' 
                    .format(epoch, args.num_epochs, i+1, total_step, loss.item(), 100*train_accu))
            continue
        print(f"Epoch {epoch} completes")
        
        # -- Decay learning rate
        if (epoch) % args.learning_rate_decay_interval == 0:
            curr_lr *= args.learning_rate_decay_rate # lr = lr * rate
            update_lr(optimizer, curr_lr)
    
        # -- Evaluate and save model
        if (epoch) % 1 == 0 or (epoch) == args.num_epochs:
            eval_accu = evaluate_model(model, eval_loader, num_to_eval=-1)
            if args.save_model_to:
                name_to_save = args.save_model_to + "/" + "{:03d}".format(epoch) + ".ckpt"
                torch.save(model.state_dict(), name_to_save)
                print("Save model to: ", name_to_save)

            # logger record
            logger.store_accuracy(epoch, train=train_accu, eval=eval_accu)
            logger.save_log(args.save_log_to)
            
            # logger Plot
            if args.plot_accu and epoch == 1:
                plt.figure(figsize=(10, 8))
                plt.ion()
                if args.show_plotted_accu:
                    plt.show()
            if (epoch == args.num_epochs) or (args.plot_accu and epoch>1):
                logger.plot_train_eval_accuracy()
                if args.show_plotted_accu:
                    plt.pause(0.01) 
                plt.savefig(fname=args.save_fig_to)
        
        # An epoch end
        print("-"*80 + "\n")
    
    # Training end
    return
            
            
'''
# ==========================================================================================
# == Test 
# ==========================================================================================

def test_model_on_a_random_dataset():

    args = types.SimpleNamespace()
    args.input_size = 12  # In a sequency of features, the feature dimensions == input_size
    args.batch_size = 1
    args.hidden_size = 64
    args.num_layers = 3
    args.num_classes = 10
    args.num_epochs = 15
    args.learning_rate = 0.0005
    args.weight_decay = 0.00
 
    class Simple_AudioDataset(Dataset):
        def __init__(self, X, Y, input_size, transform=None):
            self.input_size = input_size
            self.transform = transform
            self.Y = torch.tensor(Y, dtype=torch.int64)
            self.X = X # list[list]

        def __len__(self):
            return len(self.Y)

        def __getitem__(self, idx):
            x = torch.tensor(self.X[idx], dtype=torch.float32)
            x = x.reshape(-1, self.input_size)
            if self.transform:
                sample = self.transform(x)
            return (x, self.Y[idx])
        
    # Create random data
    num_samples = 13
    sequence_length = 23
    train_X = np.random.random((num_samples, sequence_length, args.input_size))
    train_Y = np.random.randint(low=0, high=args.num_classes, size=(num_samples, ))

    # Convert data to list, which is the required data format
    train_X = [train_X[i].flatten().tolist() for i in range(num_samples)]
    train_Y = train_Y.tolist()

    # Construct dataset
    train_dataset = Simple_AudioDataset(train_X, train_Y, args.input_size,)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size, 
        shuffle=True)
    import copy
    eval_loader = copy.deepcopy(train_loader)
    test_loader = copy.deepcopy(train_loader)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_RNN_model(args, device)

    # Train model on training set
    train_model(model, args, train_loader, eval_loader)

    # Test model on test set
    model.eval()    
    with torch.no_grad():
        evaluate_model(model, test_loader)

if __name__ == "__main__":
    test_model_on_a_random_dataset()
'''
