import torch
import torch.nn as nn

from wave_models.wavevit import wavevit_s


class WaveHyperNet(nn.Module):
    def __init__(self,
                 target_in_size=448,
                 hyper_in_channels=112,
                 target_fc_sizes=(224, 112, 56, 28, 14, 7),
                 feature_size=1,
                 pretrained_path=r"E:\xiazai\hyperIQA-master\premodel\wavevit_s.pth"):
        super(WaveHyperNet, self).__init__()

        self.backbone = wavevit_s()
        state_dict = torch.load(pretrained_path, map_location='cuda:0', weights_only=True)
        self.backbone.load_state_dict(state_dict, strict=False)
        self.backbone.train(True)

        self.target_in_size = target_in_size
        self.hyperInChn = hyper_in_channels
        self.f1, self.f2, self.f3, self.f4, self.f5, self.f6 = target_fc_sizes
        self.feature_size = feature_size

        # Conv to project 448-dim feature to hyperInChn channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(448, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.hyperInChn, 1),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        def conv_layer(out_channels):
            return nn.Conv2d(self.hyperInChn, out_channels, 3, padding=1)

        def fc_layer(out_features):
            return nn.Linear(self.hyperInChn, out_features)

        self.fc1w_conv = conv_layer(self.f1 * self.target_in_size // feature_size**2)
        self.fc1b_fc = fc_layer(self.f1)

        self.fc2w_conv = conv_layer(self.f2 * self.f1 // feature_size**2)
        self.fc2b_fc = fc_layer(self.f2)

        self.fc3w_conv = conv_layer(self.f3 * self.f2 // feature_size**2)
        self.fc3b_fc = fc_layer(self.f3)

        self.fc4w_conv = conv_layer(self.f4 * self.f3 // feature_size**2)
        self.fc4b_fc = fc_layer(self.f4)

        self.fc5w_fc = fc_layer(self.f4)
        self.fc5b_fc = fc_layer(1)

    def forward(self, x):
        vec = self.backbone.forward_features(x)  # shape: [B, 448]
        vec_reshaped = vec.view(vec.size(0), 448, 1, 1)

        # 提取特征用于生成 TargetNet 参数
        hyper_in_feat = self.conv1(vec_reshaped)
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

            'target_fc4w': self.fc4w_conv(hyper_in_feat).view(B, self.f4, self.f3, 1, 1),
            'target_fc4b': self.fc4b_fc(pool_feat).view(B, self.f4),

            'target_fc5w': self.fc5w_fc(pool_feat).view(B, 1, self.f4, 1, 1),
            'target_fc5b': self.fc5b_fc(pool_feat).view(B, 1),
        }

        return out


# 计算模型的参数量
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

# 测试模型
model = WaveHyperNet()
total_params = count_parameters(model)

print(f"Total parameters: {total_params}")
