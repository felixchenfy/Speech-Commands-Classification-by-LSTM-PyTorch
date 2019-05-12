

import numpy as np 
import time
from mylib.mylib_clf import *
from mylib.mylib_plot import *
from mylib.mylib_io import *

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyper-parameters
sequence_length = 8
input_size = 30  # feature dimension
hidden_size = 128
num_layers = 3
num_classes = 10
batch_size = 32
num_epochs = 300
learning_rate = 0.001


# Load data
train_X = read_list('train_X.csv')
train_Y = read_list('train_Y.csv')
train_X, train_Y = np.array(train_X), np.array(train_Y).astype(np.int)
# train_X = np.random.random((train_X.shape[0], 784))

classes = read_list("classes.csv")
tr_X, te_X, tr_Y, te_Y = split_data(train_X, train_Y, USE_ALL=False)

class AudioDataset(Dataset):
    def __init__(self, X, Y, input_size, transform=None):
        
        self.num_samples = X.shape[0]
        self.input_size = input_size
        self.sequence_length = X.shape[1] // input_size

        self.transform = transform

        self.X = X.reshape(-1, self.sequence_length, self.input_size).astype(np.float32)
        self.Y = Y

        self.X = torch.from_numpy(self.X)
        self.Y = torch.tensor(self.Y, dtype=torch.int64)

        return 

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        sample = (self.X[idx], self.Y[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample

train_dataset = AudioDataset(
    tr_X, tr_Y, input_size,
    )

test_dataset = AudioDataset(
    te_X, te_Y, input_size,
    )

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # 2, 100, 128
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # 2, 100, 128
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size): 100, 28, 28
        
        # Decode the hidden state of the last time step
        # out[:, -1, :]: last block in sequence
        out = self.fc(out[:, -1, :]) # 100, 10
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (featuress, labels) in enumerate(train_loader):

        ''' original code:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        # we can see that the shape of images should be: 
        #    (batch_size, sequence_length, input_size)
        '''
        featuress = featuress.to(device) # (batch, seq_len, input_size)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(featuress)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for featuress, labels in test_loader:

        featuress = featuress.to(device) # (batch, seq_len, input_size)
        labels = labels.to(device)

        # Predict
        outputs = model(featuress)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')