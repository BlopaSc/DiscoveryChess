import torch
import torch.nn as nn
import torch.nn.functional as F

# class CNN_3layers(nn.Module):
#     '''
#     input tensor would be N*12*8*8
#     state_vector: shape(12,)
#     '''
#     def __init__(self, state_vector_size):       
#         super(CNN_3layers, self).__init__() 
#         self.conv1 = nn.Conv2d(12, 32, kernel_size=3, stride=1, padding='same') 
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same') 
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same') 
        
#         # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         self.dropout = nn.Dropout(0.5)
        
#         self.fc1 = nn.Linear(128*4*4 + state_vector_size, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 1)
    
#     def forward(self, x, state_vector):
#         '''
#         input: x would be N*12*8*8
#         output: N*1
#         '''
#         print(x.size())
#         out = F.relu(self.conv1(x)) # Output N*8*8
#         # out = self.pool1(out)
#         # print(out.size())

#         out = F.relu(self.conv2(out)) # Output N*8*8
#         # out = self.pool1(out)
#         # print(out.size)

#         out = F.relu(self.conv3(out)) # Output N*4*4
#         out = self.pool(out)
#         # print(out.size)
        
#         out = out.view(out.size(0), -1) # Flatten for FC layer
#         out = torch.cat((out, state_vector), dim=1)
#         # print(out.size)
        
#         out = self.fc1(out)
#         out = self.dropout(out)
        
#         out = self.fc2(out)
#         out = self.dropout(out)
        
#         out = self.fc3(out)
        
#         return out
    
class CNN(nn.Module):
    '''
    input tensor would be N*12*8*8
    state_vector: shape(12,)
    '''
    def __init__(self, input_channels, output_channels=[32, 64, 128], kernel_size=3, state_vector_size=12, num_layers=3):       
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
            x = F.relu(conv(x))
            print(x.size())

        x = self.pool(x)
        # print(out.size)
        
        x = x.view(x.size(0), -1) # Flatten for FC layer
        x = torch.cat((x, state_vector), dim=1)
        # print(out.size)
        
        x = self.fc1(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x