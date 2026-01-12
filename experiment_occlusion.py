import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

DATA_DIR = './data/formatted/val'
WEIGHTS_DIR = './checkpoints'
FIGURE_DIR = './figures'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64 
os.makedirs(FIGURE_DIR, exist_ok=True)

class OccludeCenter(object):
    def __init__(self, size=50):
        self.size = size

    def __call__(self, img_tensor):
        _, h, w = img_tensor.shape
        cy, cx = h // 2, w // 2
        half = self.size // 2
        img = img_tensor.clone()
        img[:, cy-half:cy+half, cx-half:cx+half] = 0
        return img

class OccludeRandom(object):

    def __init__(self, probability=0.2):
        self.p = probability

    def __call__(self, img_tensor):
        mask = torch.rand_like(img_tensor) > self.p
        return img_tensor * mask.float()

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
            print(f"⚠️ 加载有瑕疵: {e}")
            model.load_state_dict(ckpt, strict=False)
    else:
        print("❌ 警告: 找不到 resnet50_best.pth")
        
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
        print("❌ 警告: 找不到 vit_small_best.pth")

    return model.eval().to(DEVICE)

def evaluate(model, loader, desc="Evaluating"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def plot_robustness(results):
    labels = list(results.keys()) 
    res_scores = [results[k][0] for k in labels]
    vit_scores = [results[k][1] for k in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, res_scores, width, label='ResNet-50 (CNN)', color='#1f77b4', alpha=0.9)
    rects2 = ax.bar(x + width/2, vit_scores, width, label='ViT-Small (Transformer)', color='#d62728', alpha=0.9)
    
    def autolabel(rects, is_vit=False):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            if i > 0:
                clean_score = vit_scores[0] if is_vit else res_scores[0]
                drop = clean_score - height

                txt = f"{height:.1f}%\n(-{drop:.1f})"
            else:
                txt = f"{height:.1f}%"
                
            ax.annotate(txt,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2, is_vit=True)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Robustness Analysis: Occlusion & Missing Information', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100) 
    
    save_path = os.path.join(FIGURE_DIR, 'robustness_occlusion_fixed.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 鲁棒性分析图已保存: {save_path}")

# --- 主程序 ---
def main():
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("📦 正在准备测试数据...")

    ds_clean = datasets.ImageFolder(DATA_DIR, transform=base_transform)
    loader_clean = DataLoader(ds_clean, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    trans_center = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        OccludeCenter(size=50), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds_center = datasets.ImageFolder(DATA_DIR, transform=trans_center)
    loader_center = DataLoader(ds_center, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    trans_random = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        OccludeRandom(probability=0.2), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds_random = datasets.ImageFolder(DATA_DIR, transform=trans_random)
    loader_random = DataLoader(ds_random, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    resnet = load_resnet()
    vit = load_vit()
    
    results = {}
    
    print("\n--- 评测 1: Clean Data (基准) ---")
    acc_res_clean = evaluate(resnet, loader_clean, "ResNet Clean")
    acc_vit_clean = evaluate(vit, loader_clean, "ViT Clean")
    results['Clean'] = [acc_res_clean, acc_vit_clean]
    print(f"ResNet: {acc_res_clean:.2f}%, ViT: {acc_vit_clean:.2f}%")
    
    print("\n--- 评测 2: Center Occlusion (中心遮挡) ---")
    acc_res_center = evaluate(resnet, loader_center, "ResNet Center")
    acc_vit_center = evaluate(vit, loader_center, "ViT Center")
    results['Center Occ.'] = [acc_res_center, acc_vit_center]
    print(f"ResNet: {acc_res_center:.2f}%, ViT: {acc_vit_center:.2f}%")
    
    print("\n--- 评测 3: Random Occlusion (随机丢弃) ---")
    acc_res_rand = evaluate(resnet, loader_random, "ResNet Random")
    acc_vit_rand = evaluate(vit, loader_random, "ViT Random")
    results['Random Occ.'] = [acc_res_rand, acc_vit_rand]
    print(f"ResNet: {acc_res_rand:.2f}%, ViT: {acc_vit_rand:.2f}%")

    print("\n🎨 正在绘制分析图...")
    plot_robustness(results)

if __name__ == "__main__":
    main()