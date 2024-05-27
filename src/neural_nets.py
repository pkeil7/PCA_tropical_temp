import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNeuralNet(nn.Module):
    def __init__(self, in_features, out_features, hl = [64, 64, 64], dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hl[0])
        self.fc2 = nn.Linear(hl[0], hl[1])
        self.fc3 = nn.Linear(hl[1], hl[2])
        self.fc4 = nn.Linear(hl[2], out_features)
        self.do = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.do(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
