import torch
import torch.nn as nn
import torch.nn.functional as F
    
class CNN(nn.Module):
    '''
    input tensor would be N*12*8*8
    state_vector: shape(22,)
    '''
    def __init__(self, input_channels=12, output_channels=[24, 48, 96], kernel_size=3, state_vector_size=22, num_layers=3, activation='relu'):
        super(CNN, self).__init__() 
        
        self.convlayers = nn.ModuleList()
        current_channels = input_channels
        for out_channel in output_channels:
            self.convlayers.append(
                
                nn.Conv2d(current_channels, out_channel, kernel_size=kernel_size, stride=1, padding='same')
            )
            current_channels = out_channel
        
        # Pooling layer for the output of the last Conv. layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dropout
        self.dropout = nn.Dropout(0.5)
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'lrelu':
            self.activation = F.leaky_relu
        
        # Fully Connected Layer
        last_conv_out_size = output_channels[-1] * 4 * 4
        
        self.fc1 = nn.Linear(last_conv_out_size + state_vector_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x, state_vector):
        '''
        input: x would be N*12*8*8
        output: N*1
        '''
        for conv in self.convlayers:
            x = self.activation(conv(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1) # Flatten for FC layer
        x = torch.cat((x, state_vector), dim=1)
        
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x