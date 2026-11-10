import torch
import torch.nn as nn
import numpy as np
import random
from .blocks import fc_ResBlock
class Estimator_LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_first = nn.Linear(256,48)
        self.rfcb1 = fc_ResBlock(3456,3072)
        self.rfcb2 = fc_ResBlock(3072,3072)
        self.com_hidden = nn.Linear(64,128)
        self.snr_hidden = nn.Embedding(16,128)
        self.rate_hidden = nn.Linear(1,128)

        self.fc = nn.Linear(3072,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
    
    def forward(self,x,snr,ratio,compress_x):
        B = x.shape[0]
        snr_cuda = torch.tensor(snr, dtype=torch.long).to(x.device)
        rate_cuda = torch.tensor(ratio, dtype=torch.float).to(x.device)
        # rate_cuda = torch.tensor(0.1,dtype=torch.float).to(x.device)
        snr_batch = snr_cuda.expand(B)
        rate_batch = rate_cuda.unsqueeze(0).expand(B, -1)
        device = x.device
        feature = self.fc_first(x)
        features = self.flatten(feature).to(device)
        # ratio = torch.tensor([ratio],device=device)
        # ratio = ratio.expand(B,-1)
        rate_batch = self.rate_hidden(rate_batch)
        snr_batch = self.snr_hidden(snr_batch)
        xcom = self.com_hidden(compress_x)
        all_features = torch.cat([features,snr_batch,rate_batch,xcom],dim=1).to(device)

        out = self.rfcb1(all_features)
        out = self.rfcb2(out)
        out = self.sigmoid(self.fc(out))
        return out


