import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        output1 = self.fc1(x)
        output2 = F.relu(output1)
        output3 = self.out(output2)
        return output3