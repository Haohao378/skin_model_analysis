import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models
import timm

WEIGHTS_DIR = './checkpoints'
FIGURE_DIR = './figures'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(FIGURE_DIR, exist_ok=True)

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super().__init__()
    
        self.features = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3,
            original_model.layer4
        )
        
    def forward(self, x):
        return self.features(x)

def get_resnet_erf(weights_path):
    print("🧠 Computing ERF for ResNet-50...")
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 7) 

    if os.path.exists(weights_path):
        try:
            ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
            if 'model' in ckpt: ckpt = ckpt['model']

            ckpt = {k: v for k, v in ckpt.items() if 'fc' not in k}
            
            model.load_state_dict(ckpt, strict=False)
            print("✅ ResNet 权重加载成功")
        except Exception as e:
            print(f"⚠️ ResNet 权重加载失败: {e}，将使用随机初始化")
    
    feature_model = ResNetFeatureExtractor(model).to(DEVICE).eval()
    
    return compute_erf(feature_model, (7, 7))

def get_vit_erf(weights_path):
    print("🧠 Computing ERF for ViT-Small...")
    model = timm.create_model('deit_small_patch16_224', pretrained=False, num_classes=7)
    
    if os.path.exists(weights_path):
        try:
            ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
            if 'model' in ckpt: ckpt = ckpt['model']

            ckpt = {k: v for k, v in ckpt.items() if 'head' not in k}
            
            model.load_state_dict(ckpt, strict=False)
            print("✅ ViT 权重加载成功")
        except Exception as e:
            print(f"⚠️ ViT 权重加载失败: {e}，将使用随机初始化")
        
    model.to(DEVICE).eval()

    class ViTWrapper(nn.Module):
        def __init__(self, vit):
            super().__init__()
            self.vit = vit
        def forward(self, x):
            x = self.vit.forward_features(x) 
            x = x[:, 1:, :]
            B, N, C = x.shape
            H = W = int(N**0.5) 
   
            x = x.permute(0, 2, 1).reshape(B, C, H, W) 
            return x

    wrapped_model = ViTWrapper(model)

    return compute_erf(wrapped_model, (14, 14))

def compute_erf(model, feature_map_size):

    input_img = torch.ones((1, 3, 224, 224), device=DEVICE, requires_grad=True)

    model.zero_grad()
    output = model(input_img) 

    H, W = feature_map_size
    center_h, center_w = H // 2, W // 2

    grad_mask = torch.zeros_like(output)

    grad_mask[:, :, center_h, center_w] = 1.0

    output.backward(gradient=grad_mask)

    grad = input_img.grad.abs().squeeze().cpu().numpy() 
    grad = np.max(grad, axis=0)
 
    grad = np.log10(grad + 1e-10) 
    grad = (grad - grad.min()) / (grad.max() - grad.min())
    
    return grad

def plot_erf_comparison():

    erf_res = get_resnet_erf(os.path.join(WEIGHTS_DIR, 'resnet50_best.pth'))
    erf_vit = get_vit_erf(os.path.join(WEIGHTS_DIR, 'vit_small_best.pth'))

    plt.style.use('default') 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(erf_res, cmap='inferno', ax=ax1, cbar=True, square=True, xticklabels=False, yticklabels=False)
    ax1.set_title('ResNet-50 ERF (Local / Gaussian)', fontsize=14, fontweight='bold')

    sns.heatmap(erf_vit, cmap='inferno', ax=ax2, cbar=True, square=True, xticklabels=False, yticklabels=False)
    ax2.set_title('ViT-Small ERF (Global / Contextual)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, 'erf_visualization.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ ERF 可视化图已保存: {save_path}")

if __name__ == "__main__":
    plot_erf_comparison()