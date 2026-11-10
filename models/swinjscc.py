import torch
import torch.nn as nn
import numpy as np
import random
from . import encoder_swinvit, decoder_swinvit
from channel import ChannelMixin
class SwinJSCCcomplex(ChannelMixin, nn.Module):
    def __init__(self):
        super().__init__() 
        image_dims = (3, 32, 32)
        # Encoder 
        embed_dims = [128,256]
        self.embed_dims=embed_dims
        self.hidden_dim = int(self.embed_dims[len(embed_dims)-1] * 1.5)
        self.layer_num=layer_num=7
        self.num_layers = 2
        ## Change C to change compress ratio
        ## Out: [B,64,C] if u wanna add CNN, first transform it to [B,C,64],then reshape it to [B,C,8,8] for further conv or other operate
        self.semantic_encoder = encoder_swinvit.SwinJSCC_Encoder(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=96,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True)
                
        # Decoder 
        self.semantic_decoder = decoder_swinvit.SwinJSCC_Decoder(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=96,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,)

        self.sm_list = nn.ModuleList()
        self.bm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(embed_dims[len(embed_dims)-1],self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = embed_dims[len(embed_dims)-1]
            else:
                outdim = self.hidden_dim
            self.sm_list.append(nn.Linear(self.hidden_dim,outdim))
            self.bm_list.append(encoder_swinvit.AdaptiveModulator(self.hidden_dim))
        self.sigmoid = nn.Sigmoid()
        self.sigmoid1 = nn.Sigmoid()


        self.bm_list1 = nn.ModuleList()
        self.sm_list1 = nn.ModuleList()
        self.sm_list1.append(nn.Linear(embed_dims[len(embed_dims)-1], self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = embed_dims[len(embed_dims)-1]
            else:
                outdim = self.hidden_dim
            self.bm_list1.append(encoder_swinvit.AdaptiveModulator(self.hidden_dim))
            self.sm_list1.append(nn.Linear(self.hidden_dim, outdim))
        # Digital data stream transmission
        self.bit_flip_ratio = 0.0125
    

    def AdaptiveMasking(self,x,snr,rate):
        B = x.shape[0]
        ch_num = round(rate * 96)
        snr_cuda = torch.tensor(snr, dtype=torch.float).to(x.device)
        rate_cuda = torch.tensor(ch_num, dtype=torch.float).to(x.device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        rate_batch = rate_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.layer_num):
            if i == 0:
                temp = self.sm_list1[i](x.detach())
            else:
                temp = self.sm_list1[i](temp)

            bm = self.bm_list1[i](snr_batch).unsqueeze(1).expand(-1, 32*32 // (self.num_layers ** 4), -1)
            temp = temp * bm
        mod_val1 = self.sigmoid1(self.sm_list1[-1](temp))
        x = x * mod_val1

        for i in range(self.layer_num):
            if i == 0:
                temp = self.sm_list[i](x.detach())
            else:
                temp = self.sm_list[i](temp)

            bm = self.bm_list[i](rate_batch).unsqueeze(1).expand(-1, 32*32 // (self.num_layers ** 4), -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_list[-1](temp))
        x = x * mod_val
        mask = torch.sum(mod_val, dim=1)
        sorted, indices = mask.sort(dim=1, descending=True)
        c_indices = indices[:, :ch_num]
        add = torch.Tensor(range(0, B * x.size()[2], x.size()[2])).unsqueeze(1).repeat(1, ch_num)
        c_indices = c_indices + add.int().to(x.device)
        mask = torch.zeros(mask.size()).reshape(-1).to(x.device)
        mask[c_indices.reshape(-1)] = 1
        mask = mask.reshape(B, x.size()[2])
        mask = mask.unsqueeze(1).expand(-1, 32*32 // (self.num_layers ** 4), -1)

        x = x * mask
        return x, mask
    

    
    
    def Channel_Nakagami(self,feature,std,m=1.0,omega=1.0):
        device=feature.device
        noise_real = torch.normal(mean=0.0,std=std,size=feature.shape,device=device)
        noise_imag = torch.normal(mean=0.0,std=std,size=feature.shape,device=device)
        noise = noise_real + 1j * noise_imag
        gamma_dist = torch.distributions.Gamma(m, omega/m)
        h = torch.sqrt(gamma_dist.sample(feature.shape)).to(device)
        return feature*h + noise






    
    

    def Encoding(self, x):
        B = x.shape[0]
        x = self.semantic_encoder(x)
        # Tx = (torch.tanh(x)* self.action_scale + self.action_bias).view(x.shape[0], -1)
        return x
    
    def Decoding(self, Rx,snr):
        B = Rx.shape[0]
        Rx = Rx.view(B, 64, 256)
        res = self.semantic_decoder(Rx,snr)
        return res

    def forward(self,x,is_training,is_val,snr_val,mask_ratio,Channel_type="AWGN"):
        B = x.shape[0]
        device = x.device
        s = self.Encoding(x)
        ratio = torch.rand(1,1,device=device).item()*0.9+0.1
        # snr_list = [0,1,4,7,10,13]
        snr_list = [-13,-10,-7,-4,-1, 0]
        Channel_list=[0,1]
        snr = random.choice(snr_list)
        if Channel_type == "Random":
            Channel = random.choice(Channel_list)
            if Channel == 0:
                Channel_type = "AWGN"
            elif Channel == 1:
                Channel_type = "Rayleigh"

        if is_val:
            ratio = mask_ratio
            snr = snr_val
        
        train_ratio = ratio
        std = self.snr2std(snr)
        masked_feature,mask = self.AdaptiveMasking(s,snr,train_ratio)
        s = self.complex_normalize(masked_feature).view(B,64,256).to(device)
        noisy_feature = self.Pass_Channel(s,std,Channel_type)
        noisy_feature = noisy_feature*mask
        # noisy_feature = s + N
        #s.shape:[B,3072]
                
        res = self.Decoding(noisy_feature,snr)
        return res


