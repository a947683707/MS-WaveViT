import torch  # 添加缺失的导入
import torch.nn as nn
import torch.nn.functional as F
from wave_models.wavevit import wavevit_s


# 优化的 TargetFC 类 - 使用批量矩阵乘法替代分组卷积
class SimplifiedTargetFC(nn.Module):
    def __init__(self):
        super(SimplifiedTargetFC, self).__init__()
        
    def forward(self, x, weights, bias):
        # 使用批量矩阵乘法提高效率
        B, C = x.shape[:2]
        x_flat = x.view(B, C)
        weights_flat = weights.view(B, weights.size(1), weights.size(2))
        
        # 批量矩阵乘法: [B, 1, C] x [B, C, out_features] = [B, 1, out_features]
        output = torch.bmm(x_flat.unsqueeze(1), weights_flat).squeeze(1)
        output = output + bias
        return output


# 优化的 TargetNet 类 - 简化为3层结构
class SimplifiedTargetNet(nn.Module):
    def __init__(self):
        super(SimplifiedTargetNet, self).__init__()
        self.fc1 = SimplifiedTargetFC()
        self.fc2 = SimplifiedTargetFC()
        self.fc3 = SimplifiedTargetFC()
        
        # HDR 特定的自适应权重
        self.hdr_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, target_in_vec, paras):
        x = target_in_vec.view(target_in_vec.size(0), -1)
        
        # 第一层 - ReLU 激活
        x = self.fc1(x, paras['target_fc1w'], paras['target_fc1b'])
        x = F.relu(x)
        
        # 第二层 - ReLU 激活
        x = self.fc2(x, paras['target_fc2w'], paras['target_fc2b'])
        x = F.relu(x)
        
        # 第三层 - 输出层
        x = self.fc3(x, paras['target_fc3w'], paras['target_fc3b'])
        
        # HDR 特定的自适应权重调整
        x = x * self.hdr_weight
        
        # 压缩到 [0,1] 范围
        x = torch.sigmoid(x)
        
        return x


class WaveHyperNet(nn.Module):
    def __init__(self,
                 target_in_size=448,
                 hyper_in_channels=64,  # 减少通道数
                 pretrained_path=r"H:\ZhangR\hyperIQA-master\premodel\wavevit_s.pth"):
        super(WaveHyperNet, self).__init__()

        self.backbone = wavevit_s()
        state_dict = torch.load(pretrained_path, map_location='cuda:0', weights_only=True)
        self.backbone.load_state_dict(state_dict, strict=False)
        self.backbone.train(True)

        self.target_in_size = target_in_size
        self.hyperInChn = hyper_in_channels
        
        # 优化的特征投影 - 更轻量级
        self.conv1 = nn.Sequential(
            nn.Conv2d(448, 128, 1),  # 减少中间层通道数
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.hyperInChn, 1),
            nn.ReLU(inplace=True)
        )
        
        # HDR 特征增强模块
        self.hdr_enhance = nn.Sequential(
            nn.Conv2d(self.hyperInChn, self.hyperInChn, 1),
            nn.Sigmoid()
        )
        
        # 简化的3层结构
        self.f1, self.f2, self.f3 = 128, 64, 1
        
        # 使用 1x1 卷积减少计算量
        self.fc1w_conv = nn.Conv2d(self.hyperInChn, self.f1 * self.target_in_size, 1)
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)
        
        self.fc2w_conv = nn.Conv2d(self.hyperInChn, self.f2 * self.f1, 1)
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)
        
        self.fc3w_conv = nn.Conv2d(self.hyperInChn, self.f3 * self.f2, 1)
        self.fc3b_fc = nn.Linear(self.hyperInChn, self.f3)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        vec = self.backbone.forward_features(x)  # [B, 448]
        vec_reshaped = vec.view(vec.size(0), 448, 1, 1)

        hyper_in_feat = self.conv1(vec_reshaped)
        
        # HDR 特征增强
        hdr_weights = self.hdr_enhance(hyper_in_feat)
        hyper_in_feat = hyper_in_feat * hdr_weights
        
        pool_feat = self.pool(hyper_in_feat).squeeze()
        B = x.size(0)
        
        # 优化版本输出 - 简化的3层结构，修正维度格式
        out = {
            'target_in_vec': vec_reshaped,
            'target_fc1w': self.fc1w_conv(hyper_in_feat).view(B, self.f1, self.target_in_size, 1, 1),
            'target_fc1b': self.fc1b_fc(pool_feat).view(B, self.f1),
            
            'target_fc2w': self.fc2w_conv(hyper_in_feat).view(B, self.f2, self.f1, 1, 1),
            'target_fc2b': self.fc2b_fc(pool_feat).view(B, self.f2),
            
            'target_fc3w': self.fc3w_conv(hyper_in_feat).view(B, self.f3, self.f2, 1, 1),
            'target_fc3b': self.fc3b_fc(pool_feat).view(B, self.f3),
        }

        return out