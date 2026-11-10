import torch
import torch.nn as nn
import numpy as np
import random
from . import encoder_swinvit, decoder_swinvit
from channel import ChannelMixin
from .check_encoder import Compression_Network_IB_Attn
from .lpips_estimator import Estimator_LPIPS
from .retransmission import Temporal_Entropy_model, Second_Compression_Network_IB_Attn, PreConcator
class JS3C_HARQ_LPIPS(ChannelMixin, nn.Module):
    def __init__(self):
        super().__init__()
        image_dims = (3, 32, 32)

        embed_dims = [128,256]
        self.embed_dims=embed_dims
        self.hidden_dim = int(self.embed_dims[len(embed_dims)-1] * 1.5)
        self.layer_num=layer_num=7
        self.num_layers = 2
        self.compression_module = Compression_Network_IB_Attn()
        self.estimation_module = Estimator_LPIPS()
        self.semantic_encoder = encoder_swinvit.SwinJSCC_Encoder(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=96,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True)
        
        
        self.real_semantic_decoder = decoder_swinvit.SwinJSCC_Decoder(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=97,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,)
        
        self.semantic_decoder = decoder_swinvit.SwinJSCC_Decoder(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=96,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,)
        
        self.second_decoder = decoder_swinvit.SwinJSCC_Decoder(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=96,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,)
        
        self.Preconcator = PreConcator()
        self.SecondEstimator = Estimator_LPIPS()
        self.compression_module2 = Second_Compression_Network_IB_Attn()
        self.decoder_linear = nn.Linear(257,256)

        self.entropy_encoder = Temporal_Entropy_model()

        self.sm_list = nn.ModuleList()
        self.bm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(self.embed_dims[len(embed_dims)-1],self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = self.embed_dims[len(embed_dims) - 1]
            else:
                outdim = self.hidden_dim
            self.sm_list.append(nn.Linear(self.hidden_dim,outdim))
            self.bm_list.append(encoder_swinvit.AdaptiveModulator(self.hidden_dim))
        self.sigmoid = nn.Sigmoid()
        self.sigmoid1 = nn.Sigmoid()
        self.bm_list1 = nn.ModuleList()
        self.sm_list1 = nn.ModuleList()
        self.sm_list1.append(nn.Linear(self.embed_dims[len(embed_dims) - 1], self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = self.embed_dims[len(embed_dims) - 1]
            else:
                outdim = self.hidden_dim
            self.bm_list1.append(encoder_swinvit.AdaptiveModulator(self.hidden_dim))
            self.sm_list1.append(nn.Linear(self.hidden_dim, outdim))

        #freeze parameters for encoding and masking
        
        for param in self.semantic_encoder.parameters():
                param.requires_grad = False
        
        for param in self.semantic_decoder.parameters():
                param.requires_grad = False
        
        for param in self.bm_list.parameters():
                param.requires_grad = False
        
        for param in self.bm_list1.parameters():
                param.requires_grad = False
        
        for param in self.sm_list.parameters():
                param.requires_grad = False
        
        for param in self.sm_list1.parameters():
                param.requires_grad = False
        
        for param in self.real_semantic_decoder.parameters():
                param.requires_grad = False
        
        for param in self.compression_module.parameters():
                param.requires_grad = False
        
        for param in self.estimation_module.parameters():
                param.requires_grad = False
        
        for param in self.decoder_linear.parameters():
                param.requires_grad = False

        
            
    
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

    
    

    
    




    
    

    def Encoding(self, x):
        B = x.shape[0]
        x = self.semantic_encoder(x)
        # Tx = (torch.tanh(x)* self.action_scale + self.action_bias).view(x.shape[0], -1)
        return x
    
    def Decoding(self, Rx, Rxcom, snr):
        B = Rx.shape[0]
        Rx = Rx.view(B, 64, 256)
        Rxcom = Rxcom.unsqueeze(2)
        feature = torch.cat([Rx,Rxcom],dim=2)
        feature = self.decoder_linear(feature)
        res = self.real_semantic_decoder(feature,snr)
        return res
    



    def forward(self,x,is_training,is_val,snr_val,mask_ratio,mask_ratio2,Channel_type='AWGN'):
        #encoding
        B = x.shape[0]
        device = x.device
        s = self.Encoding(x)
        #generate snr and ratio
        ratio = torch.rand(1,1,device=device).item()*0.9+0.1
        ratio2 = torch.rand(1,1,device=device).item()*0.9+0.1
        snr_list = [0,1,4,7,10,13]
        Channel_list = [0,1]
        if Channel_type == "Random":
            Channel = random.choice(Channel_list)
            if Channel == 0:
                Channel_type = "AWGN"
            elif Channel == 1:
                Channel_type = "Rayleigh"
        snr = random.choice(snr_list)
        if is_val:
            ratio = mask_ratio
            snr = snr_val
            ratio2 = mask_ratio2
        train_ratio = ratio
        train_ratio2 = ratio2
        std = self.snr2std(snr)
        #compress xcom
        s_for_com = s
        xcom,_ = self.Compress(s_for_com,snr,train_ratio)
        #Adaptive Masking
        masked_feature,mask = self.AdaptiveMasking(s,snr,train_ratio)
        #power normalization and Channel Pass
        s = self.complex_normalize(masked_feature).view(B,64,256).to(device)
        xcom = self.complex_normalize(xcom).view(B,64).to(device)
        noisy_feature = self.Pass_Channel(s,std,Channel_type)
        noisy_feature = noisy_feature*mask
        Noisy_xcom = self.Pass_Channel(xcom,std,Channel_type)
        #decode
        res1 = self.Decoding(noisy_feature,Noisy_xcom,snr)
        decoded_feature = self.Encoding(res1)
        ssim_predict = self.estimation_module(decoded_feature,snr,ratio,Noisy_xcom)
        #second encode
        x2 = self.Encoding(x)
        x2_for_com = x2
        x2_for_com,KL_loss1 = self.entropy_encoder(s_for_com,x2)
        x2_for_com = x2_for_com.view(B,64,256)
        xcom2,KL_loss2 = self.compression_module2(s_for_com,x2_for_com,ssim_predict,snr,train_ratio2)
        masked_feature2,mask2 = self.AdaptiveMasking(x2_for_com,snr,train_ratio2)
        s2 = self.complex_normalize(masked_feature2).view(B,64,256).to(device)
        xcom2 = self.complex_normalize(xcom2).view(B,64).to(device)
        noisy_feature2 = self.Pass_Channel(s2,std,Channel_type)
        noisy_feature2 = noisy_feature2*mask2
        Noisy_xcom2 = self.Pass_Channel(xcom2,std,Channel_type)
        #decode
        feature = self.Preconcator(noisy_feature,noisy_feature2,Noisy_xcom,Noisy_xcom2)
        res2 = self.second_decoder(feature,snr)
        decoded_feature2 = self.Encoding(res2)
        psnr_predict = self.SecondEstimator(decoded_feature2,snr,train_ratio2,Noisy_xcom2)
        return res1,res2,psnr_predict,ssim_predict,KL_loss1,KL_loss2,Noisy_xcom
    
    def Compress(self,x,snr,ratio):
        B = x.shape[0]
        out, KL_loss = self.compression_module(x,snr,ratio)
        return out, KL_loss


