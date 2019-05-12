
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np 
import time
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../")
from mylib.mylib_commons import *

# Hyper-parameters
FEATURE_WITH_DIFFERENT_LENGTH = True
if FEATURE_WITH_DIFFERENT_LENGTH:
    input_size = 8  # In a sequency of features, the dimension of each feature == input_size
    batch_size = 1
    hidden_size = 64
    num_layers = 3
    num_classes = 10
    num_epochs = 80
    learning_rate = 0.0005
    weight_decay = 0.00
else:
    input_size = 80
    batch_size = 32
    # sequency_len = 5
    hidden_size = 32
    num_layers = 3
    num_classes = 10
    num_epochs = 80
    learning_rate = 0.01
    weight_decay = 0.01

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device

    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) # 2, 100, 128
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) # 2, 100, 128
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size): 100, 28, 28
        
        # Decode the hidden state of the last time step
        # out[:, -1, :] is the last block in sequence
        out = self.fc(out[:, -1, :]) # 100, 10
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

def create_RNN_model(model_path, device):
    model = RNN(input_size, hidden_size, num_layers, num_classes, device).to(device)
    model.load_state_dict(torch.load(model_path))
    return model 

class AudioDataset(Dataset):
    def __init__(self, X, Y, input_size, transform=None):

        self.num_samples = len(X)
        self.input_size = input_size
        self.transform = transform

        # Set up Y
        self.Y = torch.tensor(Y, dtype=torch.int64)

        # Set up X
        if FEATURE_WITH_DIFFERENT_LENGTH:
            self.X = X # list[list]
        else:
            X = np.array(X)
            self.sequence_length = X.shape[1] // input_size
            X = X.reshape(-1, self.sequence_length, self.input_size).astype(np.float32)
            self.X = torch.from_numpy(X)
        return 

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):

        if FEATURE_WITH_DIFFERENT_LENGTH:
            x = torch.tensor(self.X[idx], dtype=torch.float32)
            feature_dim = len(self.X[idx])
            time_len = feature_dim//self.input_size
            x = x.reshape(time_len, self.input_size)
        else:
            x = self.X[idx]
        sample = (x, self.Y[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample

def evaluate_model(model, eval_loader, num_to_eval=-1):
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
    print('  Eval Accuracy on {} eval samples: {} %'.format(
        i+1, 100 * correct / total)) 

def train_model(model, train_loader, eval_loader, name_to_save_model=None):

    device = model.device

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # For updating learning rate
    def update_lr(optimizer, lr):    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate

    for epoch in range(num_epochs):
        cnt_correct, cnt_total = 0, 0
        for i, (featuress, labels) in enumerate(train_loader):

            ''' original code:
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record result
            _, argmax = torch.max(outputs, 1)
            cnt_correct += (labels == argmax.squeeze()).sum().item()
            cnt_total += labels.size(0)
            continue

        # Decay learning rate
        if (epoch+1) % 10 == 0:
            curr_lr /= 2
            update_lr(optimizer, curr_lr)

        # Print accuracy
        print ('Epoch [{}/{}], Step [{}/{}], Loss = {:.4f}, Accuracy = {:.2f}' 
            .format(epoch+1, num_epochs, i+1, total_step, loss.item(), 100*cnt_correct/cnt_total))

        # Save model
        if (epoch+1) % 5 == 0 or (epoch+1) == num_epochs:
            evaluate_model(model, eval_loader, num_to_eval=-1)
            if name_to_save_model is not None:
                name_to_save = add_idx_suffix(name_to_save_model, int2str(epoch, len=3))
                torch.save(model.state_dict(), name_to_save)
                print("-"*80)
                print("Save model to: ", name_to_save)
                print("-"*80)
                print("\n")
            print("")

def test_model_on_a_random_dataset():

    # Create random data
    num_samples = 13
    sequence_length = 23
    train_X = np.random.random((num_samples, sequence_length, input_size))
    train_Y = np.random.randint(low=0, high=num_classes, size=(num_samples, ))

    # Convert data to list, which is the required data format
    train_X = [train_X[i].flatten().tolist() for i in range(num_samples)]
    train_Y = train_Y.tolist()

    # Construct dataset
    train_dataset = AudioDataset(train_X, train_Y, input_size,)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size, 
        shuffle=True)
    import copy
    eval_loader = copy.deepcopy(train_loader)
    test_loader = copy.deepcopy(train_loader)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNN(input_size, hidden_size, num_layers, num_classes, device).to(device)

    # Train model on training set
    train_model(model, train_loader, eval_loader, name_to_save_model=None)

    # Test model on test set
    model.eval()    
    with torch.no_grad():
        evaluate_model(model, test_loader)

if __name__ == "__main__":
    test_model_on_a_random_dataset()