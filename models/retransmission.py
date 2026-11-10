import torch
import torch.nn as nn
import numpy as np
import random
from .blocks import compress_block, AF_block
class Temporal_Entropy_model(nn.Module):
    def __init__(self,latent_dim=64,feature_dim=256):
        super().__init__()
        self.fusion_fc = nn.Sequential(
            nn.Linear(feature_dim*2,feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim,feature_dim)
        )

        self.compressor = nn.Sequential(
            nn.Conv1d(latent_dim,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv1d(128,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv1d(128,latent_dim,kernel_size=3,padding=1),
            nn.Linear(feature_dim,feature_dim)
        )

        self.fc_mu = nn.Sequential(
            nn.Linear(feature_dim,feature_dim),
            nn.Tanh())
        
        self.fc_sigma = nn.Sequential(
            nn.Linear(feature_dim,feature_dim),
            nn.Sigmoid()
        )
    
    def KL_loss_function(self,mu1,sigma1,sigma2=1):
        batch_size = mu1.size()[0]
        J = mu1.size()[1]
        mu_diff = (mu1)**2
        var1 = sigma1 ** 2 
        var2 = sigma2 ** 2 
        var_frac = var1/var2
        diff_var_frac = mu_diff / var2
        term1 = torch.sum(torch.log(var_frac))/batch_size
        term2 = torch.sum(var_frac) / batch_size
        term3 = torch.sum(diff_var_frac) / batch_size

        return -0.5 * (term1 - term2 - term3 + J)
    
    def IB_cal(self,mu,sigma):
        B = mu.shape[0]
        mu = mu.view(B,-1)
        sigma = sigma.view(B,-1)
        KL_loss = self.KL_loss_function(mu,sigma)
        if self.training:
            eps = torch.randn_like(mu)
            x = (mu + torch.mul(eps,sigma))
        else:
            x = mu
        return x,KL_loss

    def forward(self,x1,x2):
        feature = torch.cat([x1,x2],dim=2)
        new_feature = self.fusion_fc(feature)
        new_x = self.compressor(new_feature)
        mu = self.fc_mu(new_x)
        sigma = self.fc_sigma(new_x)
        out,KL_loss = self.IB_cal(mu,sigma)
        return out,KL_loss

class Second_Compression_Network_IB_Attn(nn.Module):
    #compress feature from B,3072 to B,64
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
        self.fc_multi = nn.Linear(3200,3072)
        self.fc1 = nn.Linear(3072,1024)
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256,64)
        self.fc_first = nn.Linear(256,48)
        self.fc_second = nn.Linear(256,48)
        self.est_hidden = nn.Linear(1,128)
        self.x_fc = nn.Linear(96,48)
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
        
    def forward(self,x,x2,psnr_est,snr,ratio):
        B = x.shape[0]
        x = x.view(B,64,256)
        x2 = x.view(B,64,256)
        psnr_est = psnr_est.clone().detach().unsqueeze(0).view(B,-1)
        est_feature = self.est_hidden(psnr_est).to(x.device)
        x = self.fc_first(x)
        x2 = self.fc_second(x2)
        x_feature = torch.cat([x,x2],dim=2)
        x_feature = self.x_fc(x_feature)
        snr_cuda = torch.tensor(snr, dtype=torch.float).to(x.device)
        rate_cuda = torch.tensor(ratio, dtype=torch.float).to(x.device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        rate_batch = rate_cuda.unsqueeze(0).expand(B, -1)
        out = self.conv1(x_feature)
        out = self.AF1(out,snr_batch,rate_batch)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.AF2(out,snr_batch,rate_batch)
        out = self.flatten(out)
        out = torch.cat([out,est_feature],dim=1)
        out = self.fc_multi(out)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        # VIB
        out_mu = self.fc_mu(out)
        out_sigma = self.fc_sigma(out)
        out, KL_loss = self.IB_cal(out_mu, out_sigma)
        return out, KL_loss

class PreConcator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_first = nn.Linear(256,128)
        self.fc_second = nn.Linear(256,128)
        self.com_hidden_one = nn.Linear(1,48)
        self.com_hidden_second = nn.Linear(1,48)
        self.final_linear = nn.Linear(352,256)
    
    def forward(self,x1,x2,xcom1,xcom2):
        B = x1.shape[0]
        feature1 = self.fc_first(x1)
        feature2 = self.fc_second(x2)
        com1 = xcom1.unsqueeze(2)
        com2 = xcom2.unsqueeze(2)
        com1 = self.com_hidden_one(com1)
        com2 = self.com_hidden_second(com2)
        all_features = torch.cat([feature1,com1,feature2,com2],dim=2).to(x1.device)
        decoder_feature = self.final_linear(all_features)
        return decoder_feature


