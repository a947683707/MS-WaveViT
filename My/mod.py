import torch
import torch.nn as nn
from functools import partial

from My.wavevit import wavevit_s


class HyperNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.fc(x)

class MultiBranchWaveViTHyperNet(nn.Module):
    def __init__(self, weight_paths, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 创建WaveViT多个分支
        self.branches = nn.ModuleList()
        for wp in weight_paths:
            model = wavevit_s(pretrained=False)
            state_dict = torch.load(wp, map_location='cpu')
            # 如果权重键带'module.'，可以用下面一行去除
            # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.branches.append(model)

        # 用dummy推断单分支输出特征维度
        self.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_out = self.branches[0](dummy_input)
            if isinstance(dummy_out, (tuple, list)):
                dummy_out = dummy_out[0]
            feat_dim = dummy_out.shape[1]

        # HyperNet输入维度是所有分支特征拼接维度
        self.hypernet = HyperNet(input_dim=feat_dim * len(self.branches)).to(self.device)

    def forward(self, imgs):
        """
        imgs: list，长度等于分支数，每个元素是Tensor或PIL.Image
        返回: Tensor，[batch]分数
        """
        feats = []
        for i, img in enumerate(imgs):
            if isinstance(img, torch.Tensor):
                x = img.to(self.device)
            else:
                raise RuntimeError("请确保输入已预处理为Tensor并在正确设备")
            with torch.no_grad():
                out = self.branches[i](x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                feats.append(out)
        feat_concat = torch.cat(feats, dim=1)
        score = self.hypernet(feat_concat)
        return score.squeeze(1)


# 使用示例
if __name__ == '__main__':
    from torchvision import transforms
    from PIL import Image
    import os

    weight_paths = [
        r"E:\xiazai\hyperIQA-master\premodel\wavevit_s.pth",
        r"E:\xiazai\hyperIQA-master\premodel\wavevit_s.pth"
    ]
    device = 'cuda'

    model = MultiBranchWaveViTHyperNet(weight_paths, device=device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 载入两张图片分别输入两个分支
    img1_path = "test1.jpg"
    img2_path = "test2.jpg"
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        t_img1 = transform(img1).unsqueeze(0).to(device)
        t_img2 = transform(img2).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            score = model([t_img1, t_img2])
        print("融合多分支 + HyperNet预测分数:", score.item())
    else:
        print("请准备两张测试图片 test1.jpg 和 test2.jpg")
