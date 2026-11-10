import torch
import torch.nn as nn
import numpy as np
import random

class ChannelMixin:
    def Channel_AWGN(self,feature,std):
        device = feature.device
        noise_real = torch.normal(mean=0.0,std=std,size=feature.shape,device=device)
        noise_imag = torch.normal(mean=0.0,std=std,size=feature.shape,device=device)
        noise = noise_real + 1j*noise_imag
        return feature + noise

    def Channel_AWGN_Real(self,feature,std):
        device = feature.device
        noise_real = torch.normal(mean=0.0,std=std,size=feature.shape,device=device)
        noise = noise_real
        return feature + noise

    def Channel_Rayleigh(self,feature,std):
        device=feature.device
        noise_real = torch.normal(mean=0.0,std=std,size=feature.shape,device=device)
        noise_imag = torch.normal(mean=0.0,std=std,size=feature.shape,device=device)
        noise = noise_real + 1j * noise_imag
        h = torch.sqrt(torch.normal(mean=0.0,std=1,size=feature.shape)**2 + torch.normal(mean=0.0,std=1,size=feature.shape)**2)/np.sqrt(2)
        h = h.to(device)
        return feature*h + noise

    def Channel_Rayleigh_Real(self,feature,std):
        device=feature.device
        noise_real = torch.normal(mean=0.0,std=std,size=feature.shape,device=device)
        noise = noise_real
        h = torch.sqrt(torch.normal(mean=0.0,std=1,size=feature.shape)**2 + torch.normal(mean=0.0,std=1,size=feature.shape)**2)/np.sqrt(2)
        h = h.to(device)
        return feature*h + noise

    def Pass_Channel(self,feature,std,Channel_type="AWGN"):
        input_shape = feature.shape
        channel_in = feature.reshape(-1)
        L = channel_in.shape[0]
        channel_in = channel_in[:L//2] + channel_in[L//2:] *1j
        if Channel_type == "AWGN":
            noisy_feature = self.Channel_AWGN(channel_in,std)
        elif Channel_type == "Rayleigh":
            noisy_feature = self.Channel_Rayleigh(channel_in,std)
        else:
            noisy_feature = feature
        
        channel_out = torch.cat([torch.real(noisy_feature),torch.imag(noisy_feature)])
        channel_out = channel_out.reshape(input_shape)
        return channel_out

    def Pass_Channel_Real(self,feature,std,Channel_type="AWGN"):
        input_shape = feature.shape
        channel_in = feature.reshape(-1)
        L = channel_in.shape[0]
        if Channel_type == "AWGN":
            noisy_feature = self.Channel_AWGN_Real(channel_in,std)
        elif Channel_type == "Rayleigh":
            noisy_feature = self.Channel_Rayleigh_Real(channel_in,std)
        else:
            noisy_feature = feature
        
        channel_out = noisy_feature.reshape(input_shape)
        return channel_out

    def complex_normalize(self,feature):
        B = feature.shape[0]
        feature = feature.view(B,-1)
        P_current = torch.mean(torch.abs(feature)**2*2,dim=1)
        P_target = 1 
        k = torch.sqrt(torch.div(P_target,(P_current+1e-6)))
        normalized_feature = feature*k.view(B,-1)
        return normalized_feature

    def real_normalize(self,feature):
        B = feature.shape[0]
        feature = feature.view(B,-1)
        P_current = torch.mean(torch.abs(feature)**2,dim=1)
        P_target = 1 
        k = torch.sqrt(torch.div(P_target,(P_current+1e-6)))
        normalized_feature = feature*k.view(B,-1)
        return normalized_feature

    def snr2std(self,snr):
        # snr = 10 ** (snr/10)
        # noise_std = 1/torch.sqrt(2*snr)
        # return noise_std
        snr = 10 ** (snr / 10)
        noise_std = 1 / np.sqrt(2 * snr)
        return noise_std    


