import torch
import torch.nn as nn
import numpy as np
import random
from .blocks import compress_block, AF_block
class Compression_Network_IB_Attn(nn.Module):
    #compress feature from B,3072 to B,48
    def __init__(self):
        super().__init__()
        self.conv1 = compress_block(64,256,kernel_size=5,stride=1,padding=2)
        self.conv2 = compress_block(256,256,kernel_size=5,stride=1,padding=2)
        self.conv3 = compress_block(256,256,kernel_size=5,stride=1,padding=2)
        self.conv4 = compress_block(256,256,kernel_size=5,stride=1,padding=2)
        self.conv5 = compress_block(256,64,kernel_size=5,stride=1,padding=2)
        self.AF1 = AF_block(256,128,256)
        self.AF2 = AF_block(64,32,64)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3072,1024)
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256,64)
        self.fc_first = nn.Linear(256,48)
        self.fc_mu = nn.Sequential(nn.Linear(64, 64),
                                   nn.Tanh())
        self.fc_sigma = nn.Sequential(nn.Linear(64, 64),
                                    nn.Sigmoid())
    
    def KL_loss_function(self, mu1, sigma1, sigma2 = 1):
        batch_size = mu1.size()[0]
        J = mu1.size()[1]

        mu_diff = (mu1) ** 2
        var1 = sigma1 ** 2
        var2 = sigma2 ** 2

        var_frac = var1 / var2
        diff_var_frac = mu_diff / var2

        term1 = torch.sum(torch.log(var_frac)) / batch_size
        term2 = torch.sum(var_frac) / batch_size
        term3 = torch.sum(diff_var_frac) / batch_size

        return - 0.5 * (term1 - term2 -term3 + J)

    def IB_cal(self, mu, sigma):
        B = mu.shape[0]
        mu = mu.view(B, -1)
        sigma = sigma.view(B, -1)
        KL_loss = self.KL_loss_function(mu, sigma)
        if self.training:
            eps = torch.randn_like(mu)
            x = (mu + torch.mul(eps,sigma))
        else:
            x = mu
        return x, KL_loss
        
    def forward(self,x,snr,ratio):
        B = x.shape[0]
        x = x.view(B,64,256)
        x = self.fc_first(x)
        snr_cuda = torch.tensor(snr, dtype=torch.float).to(x.device)
        rate_cuda = torch.tensor(ratio, dtype=torch.float).to(x.device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        rate_batch = rate_cuda.unsqueeze(0).expand(B, -1)
        out = self.conv1(x)
        out = self.AF1(out,snr_batch,rate_batch)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.AF2(out,snr_batch,rate_batch)
        out = self.flatten(out)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        # VIB
        out_mu = self.fc_mu(out)
        out_sigma = self.fc_sigma(out)
        out, KL_loss = self.IB_cal(out_mu, out_sigma)
        return out, KL_loss


