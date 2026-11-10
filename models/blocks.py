import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
def conv1d(in_channels,out_channels,kernel_size=3,stride=1,padding=1):
    return nn.Conv1d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False)

class compress_block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1):
        super(compress_block,self).__init__()
        self.conv = conv1d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        # self.bn = nn.BatchNorm2d(out_channels)
        # JS3C didnt have bn, CCHARQ/SCHARQ did
        # self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        out = self.conv(x)
        out = self.relu(out)
        return out

class fc_ResBlock(nn.Module):
    def __init__(self, Nin, Nout):
        super(fc_ResBlock, self).__init__()
        Nh = Nin*2
        self.use_fc3 = False
        self.fc1 = nn.Linear(Nin, Nh)
        self.fc2 = nn.Linear(Nh, Nout)
        self.relu = nn.ReLU()
        if Nin != Nout:
            self.use_fc3 = True
            self.fc3 = nn.Linear(Nin, Nout)
    def forward(self, x): 
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        if self.use_fc3 == True:
            x = self.fc3(x)
        out = out+x
        out = self.relu(out)
        return out 

class AF_block(nn.Module):
    def __init__(self, Nin, Nh, No):
        super(AF_block, self).__init__()
        self.fc1 = nn.Linear(Nin+1+1, Nh)
        self.fc2 = nn.Linear(Nh, No)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, snr, ratio):
        # out = F.adaptive_avg_pool2d(x, (1,1)) 
        # out = torch.squeeze(out)
        # out = torch.cat((out, snr), 1)
        if snr.shape[0]>1:
            snr = snr.squeeze()
            snr = snr.unsqueeze(1)
        if ratio.shape[0] > 1:
            ratio = ratio.squeeze()
            ratio = ratio.unsqueeze(1)
        mu = torch.mean(x, dim=2)
        out = torch.cat((mu, snr, ratio), 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2)
        out = out*x
        return out


