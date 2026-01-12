import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import models, transforms
import timm
import math

WEIGHTS_DIR = './checkpoints'
DATA_DIR = './data/formatted/val' 
FIGURE_DIR = './figures'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(FIGURE_DIR, exist_ok=True)

TARGET_CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'vasc']

class NativeGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        grads = self.gradients
        acts = self.activations
        
        if len(acts.shape) == 4: 
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * acts, dim=1).squeeze()
        elif len(acts.shape) == 3:
            weights = torch.mean(grads, dim=1, keepdim=True)
            cam = torch.sum(weights * acts, dim=2).squeeze()
            cam = cam[1:] 
            side = int(math.sqrt(cam.shape[0]))
            cam = cam.view(side, side)
            
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()

def overlay_heatmap(pil_img, cam_matrix, alpha=0.6): 
    cam_pil = Image.fromarray((cam_matrix * 255).astype(np.uint8))
    cam_pil = cam_pil.resize(pil_img.size, resample=Image.BICUBIC)
    
    cmap = plt.get_cmap("jet")
    cam_colored = cmap(np.array(cam_pil) / 255.0)
    cam_rgb = (cam_colored[:, :, :3] * 255).astype(np.uint8)
    cam_final = Image.fromarray(cam_rgb)
    
    return Image.blend(pil_img.convert("RGB"), cam_final, alpha=alpha)

def load_resnet():
    print("Loading ResNet-50...")
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 7)
    pth = os.path.join(WEIGHTS_DIR, 'resnet50_best.pth')
    if os.path.exists(pth):
        ckpt = torch.load(pth, map_location='cpu', weights_only=False)
        if 'model' in ckpt: ckpt = ckpt['model']
        ckpt = {k:v for k,v in ckpt.items() if 'fc' not in k}
        model.load_state_dict(ckpt, strict=False)
    return model.eval().to(DEVICE)

def load_vit():
    print("Loading ViT-Small...")
    model = timm.create_model('deit_small_patch16_224', pretrained=False, num_classes=7)
    pth = os.path.join(WEIGHTS_DIR, 'vit_small_best.pth')
    if os.path.exists(pth):
        ckpt = torch.load(pth, map_location='cpu', weights_only=False)
        if 'model' in ckpt: ckpt = ckpt['model']
        ckpt = {k:v for k,v in ckpt.items() if 'head' not in k}
        model.load_state_dict(ckpt, strict=False)
    return model.eval().to(DEVICE)

def run():
    img_paths = []
    for cls in TARGET_CLASSES:
        d = os.path.join(DATA_DIR, cls)
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.jpg'):
                    img_paths.append((cls, os.path.join(d, f)))
                    break
    
    if not img_paths: 
        print("未找到图片")
        return

    resnet = load_resnet()
    vit = load_vit()
    cam_res = NativeGradCAM(resnet, resnet.layer4[-1])
    cam_vit = NativeGradCAM(vit, vit.norm)

    print(f"🎨 Generating Heatmaps for {len(img_paths)} classes...")

    cols = len(img_paths)
    rows = 3
    
    fig_width = 3.0 * cols 
    fig_height = 3.2 * rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.99, top=0.92, bottom=0.02)

    for idx, (label, p) in enumerate(img_paths):
        print(f"[{idx+1}/{cols}] Processing {label}...")
        pil_img = Image.open(p).convert('RGB').resize((224, 224))
        
        img_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])(pil_img).unsqueeze(0).to(DEVICE).requires_grad_(True)

        res_cam = cam_res(img_t)
        vit_cam = cam_vit(img_t)
        
        res_vis = overlay_heatmap(pil_img, res_cam)
        vit_vis = overlay_heatmap(pil_img, vit_cam)

        ax_orig = axes[0, idx]
        ax_orig.imshow(pil_img)
        ax_orig.axis('off')
        ax_orig.set_title(label.upper(), fontsize=16, fontweight='bold', pad=10)

        ax_res = axes[1, idx]
        ax_res.imshow(res_vis)
        ax_res.axis('off')

        ax_vit = axes[2, idx]
        ax_vit.imshow(vit_vis)
        ax_vit.axis('off')

        if idx == 0:
            ax_orig.text(-0.2, 0.5, "Original", transform=ax_orig.transAxes, 
                         fontsize=16, fontweight='bold', va='center', rotation=90)
            
            ax_res.text(-0.2, 0.5, "ResNet-50\n(CNN)", transform=ax_res.transAxes, 
                         fontsize=16, fontweight='bold', va='center', rotation=90)
            
            ax_vit.text(-0.2, 0.5, "ViT-Small\n(Transformer)", transform=ax_vit.transAxes, 
                         fontsize=16, fontweight='bold', va='center', rotation=90, color='#d62728')

    out_path = os.path.join(FIGURE_DIR, 'gradcam_compact.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ 横向紧凑版已保存: {out_path}")

if __name__ == "__main__":
    run()