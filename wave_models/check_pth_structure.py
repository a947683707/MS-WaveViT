import torch

# 替换成你的 .pth 文件路径
path = r"E:\xiazai\hyperIQA-master\premodel\wavevit_s.pth"

state = torch.load(path, map_location='cpu')

print(f"Loaded object type: {type(state)}")

if isinstance(state, dict):
    print("Top-level keys:")
    for k in list(state.keys())[:10]:  # 最多显示前 10 个 key
        print(f"  {k}")
else:
    print("该 .pth 文件不是以字典格式保存的，可能无法使用 weights_only=True")
