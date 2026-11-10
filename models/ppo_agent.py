import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
class ActorCritic(nn.Module):
    """
    Actor-Critic网络结构
    - Actor：输出动作概率分布
    - Critic：输出状态价值V(s)
    """
    def __init__(self, state_dim, action_dim=2, hidden_dim=256):
        super().__init__()
        self.action_dim = action_dim
        self.snr_linear = nn.Linear(1,64)
        self.ratio_linear = nn.Linear(1,64)
        self.lpips_linear = nn.Linear(3,64)
        self.xcom_linear = nn.Linear(64,64)
        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Actor头：输出动作logits
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic头：输出状态价值V(s)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        # Actor最后一层用更小的初始化
        for m in self.actor.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
    
    def forward(self, x):
        snr_feat = x[:, 0:1]  # [batch, 1]
        ratio_feat = x[:, 1:2]  # [batch, 1]
        lpips_feat = x[:, 2:5]  # [batch, 4] - target, pred_lpips, lpips_gap, need_retrans_signal
        xcom_feat = x[:, 5:]  # [batch, 64]

        # 分别通过对应的线性层进行特征对齐
        snr_aligned = self.snr_linear(snr_feat)  # [batch, 64]
        ratio_aligned = self.ratio_linear(ratio_feat)  # [batch, 64]
        lpips_aligned = self.lpips_linear(lpips_feat)  # [batch, 64]
        xcom_aligned = self.xcom_linear(xcom_feat)  # [batch, 64]

        # 在隐空间concat所有对齐特征
        aligned_features = torch.cat([
            snr_aligned,    # [batch, 64]
            ratio_aligned,  # [batch, 64]
            lpips_aligned,  # [batch, 64]
            xcom_aligned    # [batch, 64]
        ], dim=-1)  # [batch, 256]
        feature = self.feature(aligned_features)
        logits = self.actor(feature)
        value = self.critic(feature)
        return logits, value
    
    def get_action_and_value(self, x, action=None):
        """获取动作、对数概率和价值"""
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)


