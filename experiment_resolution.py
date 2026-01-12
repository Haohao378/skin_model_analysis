import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

DATA_DIR = './data/formatted/val'
WEIGHTS_DIR = './checkpoints'
FIGURE_DIR = './figures'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128

RESOLUTIONS = [32, 64, 96, 128, 160, 192, 224] 

class ResizeTest(object):
    def __init__(self, target_size):
        self.target_size = target_size
        self.final_size = 224
        
    def __call__(self, img):
        img_small = transforms.Resize((self.target_size, self.target_size))(img)
        img_back = transforms.Resize((self.final_size, self.final_size))(img_small)
        return img_back

def load_resnet():
    print("Loading ResNet-50 (Trained)...")
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 7)
    
    pth = os.path.join(WEIGHTS_DIR, 'resnet50_best.pth')
    if os.path.exists(pth):
        ckpt = torch.load(pth, map_location='cpu', weights_only=False)
        if 'model' in ckpt: ckpt = ckpt['model']
        
        try:
            model.load_state_dict(ckpt, strict=True)
            print("✅ ResNet-50 权重完美加载！")
        except RuntimeError as e:
            print(f"⚠️ 加载有瑕疵 (可能有些层不匹配): {e}")
            model.load_state_dict(ckpt, strict=False)
    else:
        print("❌ 警告：找不到 resnet50_best.pth，请先运行 main_train.py！")
        
    return model.eval().to(DEVICE)

def load_vit():
    print("Loading ViT-Small (Trained)...")
    model = timm.create_model('deit_small_patch16_224', pretrained=False, num_classes=7)
    
    pth = os.path.join(WEIGHTS_DIR, 'vit_small_best.pth')
    if os.path.exists(pth):
        ckpt = torch.load(pth, map_location='cpu', weights_only=False)
        if 'model' in ckpt: ckpt = ckpt['model']
        
        try:
            model.load_state_dict(ckpt, strict=True)
            print("✅ ViT-Small 权重完美加载！")
        except RuntimeError as e:
            print(f"⚠️ 加载有瑕疵: {e}")
            model.load_state_dict(ckpt, strict=False)
    else:
        print("❌ 警告：找不到 vit_small_best.pth，请先运行 main_train.py！")

    return model.eval().to(DEVICE)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def plot_sensitivity(res_list, res_accs, vit_accs):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    plt.plot(res_list, res_accs, 'o--', label='ResNet-50 (CNN)', color='#1f77b4', linewidth=2, markersize=8)
    plt.plot(res_list, vit_accs, 's-', label='ViT-Small (Transformer)', color='#d62728', linewidth=2.5, markersize=8)
 
    plt.fill_between(res_list, res_accs, vit_accs, alpha=0.1, color='#d62728')
    
    plt.title('Resolution Sensitivity Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Input Effective Resolution (pixels)', fontsize=13)
    plt.ylabel('Top-1 Accuracy (%)', fontsize=13)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(res_list)
    
    if max(max(res_accs), max(vit_accs)) > 20:
        plt.ylim(bottom=0) 

    save_path = os.path.join(FIGURE_DIR, 'resolution_sensitivity_fixed.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 修正版分析图已保存: {save_path}")

def main():
    print(f"🚀 开始分辨率敏感度测试 (Fixed): {RESOLUTIONS}")

    resnet = load_resnet()
    vit = load_vit()
    
    res_accs = []
    vit_accs = []

    for res in tqdm(RESOLUTIONS, desc="Testing Resolutions"):
        current_transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            ResizeTest(res),             
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        ds = datasets.ImageFolder(DATA_DIR, transform=current_transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
  
        acc_r = evaluate(resnet, loader)
        acc_v = evaluate(vit, loader)
        
        res_accs.append(acc_r)
        vit_accs.append(acc_v)
        
        print(f"Res {res}x{res} -> ResNet: {acc_r:.2f}%, ViT: {acc_v:.2f}%")

    plot_sensitivity(RESOLUTIONS, res_accs, vit_accs)

if __name__ == "__main__":
    main()