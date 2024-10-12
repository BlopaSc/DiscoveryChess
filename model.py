import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_3layers(nn.Module):
    '''
    input tensor would be N*8*8
    '''
    def __init__(self, state_vector = 22):
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(64*4*4 + state_vector, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        '''
        input: x would be N*8*8
        output: N*1
        '''
        out = F.relu(self.conv1(x)) # Output N*8*8
        out = self.pool1(out)

        out = F.relu(self.conv2(out)) # Output N*8*8
        out = self.pool1(out)

        out = F.relu(self.conv3(out)) # Output N*4*4
        out = self.pool2(out)
        
        out = out.view(-1, 64*4*4) # Flatten for FC layer
        
        out = self.fc1(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out