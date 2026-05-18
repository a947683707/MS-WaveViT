import torch
import torch.nn as nn
import torch.nn.functional as F
from wave_models.wavevit import wavevit_s
from wave_models.torch_wavelets import DWT_2D, IDWT_2D
import os

# 多尺度小波注意力模块
class MultiScaleWaveletAttention(nn.Module):
    def __init__(self, in_channels, scales=[1, 2]):
        super(MultiScaleWaveletAttention, self).__init__()
        self.scales = scales
        self.in_channels = in_channels
        
        # 使用 db4 小波替代 haar
        self.dwt = DWT_2D(wave='db4')
        self.idwt = IDWT_2D(wave='db4')
        
        # 为每个尺度创建注意力模块
        self.scale_attentions = nn.ModuleList()
        for scale in scales:
            self.scale_attentions.append(
                nn.Sequential(
                    nn.Conv2d(in_channels * 4, in_channels, 1),  # 4个小波子带
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, in_channels, 3, padding=1),
                    nn.Sigmoid()
                )
            )
        
        # 多尺度融合权重
        self.fusion_weights = nn.Parameter(torch.ones(len(scales)))
        self.fusion_conv = nn.Conv2d(in_channels * len(scales), in_channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        scale_features = []
        
        # 检查输入尺寸，确保足够大以进行小波变换
        min_size = 8  # db4 小波需要至少 8x8 的输入
        
        for i, scale in enumerate(self.scales):
            # 多尺度下采样
            if scale > 1:
                new_h, new_w = H // scale, W // scale
                # 确保下采样后的尺寸不小于最小要求
                if new_h < min_size or new_w < min_size:
                    # 跳过这个尺度，使用原始输入
                    scaled_x = x
                else:
                    scaled_x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            else:
                scaled_x = x
            
            # 检查当前尺寸是否足够进行小波变换
            curr_h, curr_w = scaled_x.shape[2], scaled_x.shape[3]
            if curr_h < min_size or curr_w < min_size:
                # 如果尺寸太小，直接使用原始特征
                reconstructed = scaled_x
            else:
                # 小波变换
                wavelet_coeffs = self.dwt(scaled_x)
                LL, LH, HL, HH = wavelet_coeffs
                
                # 拼接四个子带
                wavelet_concat = torch.cat([LL, LH, HL, HH], dim=1)
                
                # 注意力计算
                attention = self.scale_attentions[i](wavelet_concat)
                
                # 应用注意力到LL子带
                enhanced_LL = LL * attention
                
                # 重构
                reconstructed = self.idwt((enhanced_LL, LH, HL, HH))
            
            # 恢复到原始尺寸
            if reconstructed.shape[2:] != (H, W):
                reconstructed = F.interpolate(reconstructed, size=(H, W), mode='bilinear', align_corners=False)
            
            scale_features.append(reconstructed)
        
        # 多尺度特征融合
        weighted_features = []
        for i, feat in enumerate(scale_features):
            weighted_features.append(feat * self.fusion_weights[i])
        
        fused_features = torch.cat(weighted_features, dim=1)
        output = self.fusion_conv(fused_features)
        
        return output + x  # 残差连接

# 优化的 TargetFC 类
class SimplifiedTargetFC(nn.Module):
    def __init__(self):
        super(SimplifiedTargetFC, self).__init__()
        
    def forward(self, x, weights, bias):
        B, C = x.shape[:2]
        x_flat = x.view(B, C)
        weights_flat = weights.view(B, weights.size(1), weights.size(2))
        
        output = torch.bmm(x_flat.unsqueeze(1), weights_flat).squeeze(1)
        output = output + bias
        return output

# 优化的 TargetNet 类
class SimplifiedTargetNet(nn.Module):
    def __init__(self):
        super(SimplifiedTargetNet, self).__init__()
        self.fc1 = SimplifiedTargetFC()
        self.fc2 = SimplifiedTargetFC()
        self.fc3 = SimplifiedTargetFC()
        
        self.hdr_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, target_in_vec, paras):
        x = target_in_vec.view(target_in_vec.size(0), -1)
        
        x = self.fc1(x, paras['target_fc1w'], paras['target_fc1b'])
        x = F.relu(x)
        
        x = self.fc2(x, paras['target_fc2w'], paras['target_fc2b'])
        x = F.relu(x)
        
        x = self.fc3(x, paras['target_fc3w'], paras['target_fc3b'])
        
        x = x * self.hdr_weight
        x = torch.sigmoid(x)
        
        return x

# 增强版 WaveHyperNet（第二阶段）
class EnhancedWaveHyperNet_Stage2(nn.Module):
    def __init__(self,
                 target_in_size=448,
                 hyper_in_channels=64,
                 pretrained_path='./premodel/wavevit_s.pth'):
        super(EnhancedWaveHyperNet_Stage2, self).__init__()

        self.backbone = wavevit_s()
        self.target_in_size = target_in_size
        self.hyperInChn = hyper_in_channels
        
        # 加载预训练权重（兼容性加载）
        self.load_pretrained_with_compatibility(pretrained_path)
        
        # 新增：多尺度小波注意力模块（减少尺度数量以避免尺寸问题）
        self.multiscale_wavelet_attention = MultiScaleWaveletAttention(448, scales=[1, 2])
        
        # 特征投影（保持原有结构）
        self.conv1 = nn.Sequential(
            nn.Conv2d(448, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.hyperInChn, 1),
            nn.ReLU(inplace=True)
        )
        
        # HDR 特征增强模块
        self.hdr_enhance = nn.Sequential(
            nn.Conv2d(self.hyperInChn, self.hyperInChn, 1),
            nn.Sigmoid()
        )
        
        # 3层结构
        self.f1, self.f2, self.f3 = 128, 64, 1
        
        # 超网络层
        self.fc1w_conv = nn.Conv2d(self.hyperInChn, self.f1 * self.target_in_size, 1)
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)
        
        self.fc2w_conv = nn.Conv2d(self.hyperInChn, self.f2 * self.f1, 1)
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)
        
        self.fc3w_conv = nn.Conv2d(self.hyperInChn, self.f3 * self.f2, 1)
        self.fc3b_fc = nn.Linear(self.hyperInChn, self.f3)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def load_pretrained_with_compatibility(self, pretrained_path):
        """兼容性加载预训练权重"""
        if os.path.exists(pretrained_path):
            try:
                state_dict = torch.load(pretrained_path, map_location='cuda:0', weights_only=True)
                
                # 只加载backbone的兼容权重
                backbone_state_dict = {}
                for key, value in state_dict.items():
                    # 跳过小波相关的权重（因为从haar改为db4）
                    if 'dwt' not in key and 'idwt' not in key and 'wavelet' not in key:
                        backbone_state_dict[key] = value
                
                # 加载兼容的权重
                missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=False)
                
                print(f"✅ 成功加载预训练权重，跳过了 {len(missing_keys)} 个不兼容的键")
                if missing_keys:
                    print(f"未加载的键: {missing_keys[:5]}...")  # 只显示前5个
                    
            except Exception as e:
                print(f"⚠️ 预训练权重加载失败: {e}，将使用随机初始化")
        else:
            print(f"⚠️ 预训练文件不存在: {pretrained_path}，将使用随机初始化")
    
    def get_new_modules(self):
        """返回新增的模块，用于分阶段训练"""
        return [self.multiscale_wavelet_attention]
    
    def freeze_pretrained_modules(self):
        """冻结预训练模块"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.hdr_enhance.parameters():
            param.requires_grad = False
        # 超网络层也冻结
        for param in self.fc1w_conv.parameters():
            param.requires_grad = False
        for param in self.fc1b_fc.parameters():
            param.requires_grad = False
        for param in self.fc2w_conv.parameters():
            param.requires_grad = False
        for param in self.fc2b_fc.parameters():
            param.requires_grad = False
        for param in self.fc3w_conv.parameters():
            param.requires_grad = False
        for param in self.fc3b_fc.parameters():
            param.requires_grad = False
    
    def unfreeze_all_modules(self):
        """解冻所有模块"""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        # 骨干网络特征提取
        vec = self.backbone.forward_features(x)  # [B, 448]
        vec_reshaped = vec.view(vec.size(0), 448, 1, 1)
        
        # 新增：多尺度小波注意力
        enhanced_vec = self.multiscale_wavelet_attention(vec_reshaped)
        
        # 特征投影
        hyper_in_feat = self.conv1(enhanced_vec)
        
        # HDR 特征增强
        hdr_weights = self.hdr_enhance(hyper_in_feat)
        hyper_in_feat = hyper_in_feat * hdr_weights
        
        pool_feat = self.pool(hyper_in_feat).squeeze()
        B = x.size(0)
        
        # 超网络输出
        out = {
            'target_in_vec': enhanced_vec,  # 使用增强后的特征
            'target_fc1w': self.fc1w_conv(hyper_in_feat).view(B, self.f1, self.target_in_size, 1, 1),
            'target_fc1b': self.fc1b_fc(pool_feat).view(B, self.f1),
            
            'target_fc2w': self.fc2w_conv(hyper_in_feat).view(B, self.f2, self.f1, 1, 1),
            'target_fc2b': self.fc2b_fc(pool_feat).view(B, self.f2),
            
            'target_fc3w': self.fc3w_conv(hyper_in_feat).view(B, self.f3, self.f2, 1, 1),
            'target_fc3b': self.fc3b_fc(pool_feat).view(B, self.f3),
        }

        return out

# 保持原有的 WaveHyperNet 类不变（向后兼容）
class WaveHyperNet(nn.Module):
    def __init__(self,
                 target_in_size=448,
                 hyper_in_channels=64,
                 pretrained_path='./premodel/wavevit_s.pth'):
        super(WaveHyperNet, self).__init__()

        self.backbone = wavevit_s()
        state_dict = torch.load(pretrained_path, map_location='cuda:0', weights_only=True)
        self.backbone.load_state_dict(state_dict, strict=False)
        self.backbone.train(True)

        self.target_in_size = target_in_size
        self.hyperInChn = hyper_in_channels
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(448, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.hyperInChn, 1),
            nn.ReLU(inplace=True)
        )
        
        self.hdr_enhance = nn.Sequential(
            nn.Conv2d(self.hyperInChn, self.hyperInChn, 1),
            nn.Sigmoid()
        )
        
        self.f1, self.f2, self.f3 = 128, 64, 1
        
        self.fc1w_conv = nn.Conv2d(self.hyperInChn, self.f1 * self.target_in_size, 1)
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)
        
        self.fc2w_conv = nn.Conv2d(self.hyperInChn, self.f2 * self.f1, 1)
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)
        
        self.fc3w_conv = nn.Conv2d(self.hyperInChn, self.f3 * self.f2, 1)
        self.fc3b_fc = nn.Linear(self.hyperInChn, self.f3)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        vec = self.backbone.forward_features(x)
        vec_reshaped = vec.view(vec.size(0), 448, 1, 1)

        hyper_in_feat = self.conv1(vec_reshaped)
        
        hdr_weights = self.hdr_enhance(hyper_in_feat)
        hyper_in_feat = hyper_in_feat * hdr_weights
        
        pool_feat = self.pool(hyper_in_feat).squeeze()
        B = x.size(0)
        
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
