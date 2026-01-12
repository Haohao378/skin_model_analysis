import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import timm

DATA_DIR = './data/formatted'
WEIGHTS_DIR = './checkpoints'
FIGURE_DIR = './figures'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_CLASSES = 7

os.makedirs(FIGURE_DIR, exist_ok=True)

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def build_model(model_name, weights_path):
    print(f"🏗️ Loading {model_name} from {weights_path}...")
    
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif model_name == 'vit_small':
        model = timm.create_model('deit_small_patch16_224', pretrained=False, num_classes=NUM_CLASSES)

    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint: checkpoint = checkpoint['model']

        keys_to_remove = ['head.weight', 'head.bias', 'fc.weight', 'fc.bias']
        for k in keys_to_remove:
            if k in checkpoint and checkpoint[k].shape[0] != NUM_CLASSES:
                del checkpoint[k]
                
        model.load_state_dict(checkpoint, strict=False)
    else:
        print(f"❌ Warning: Weights not found at {weights_path}")
        
    return model.to(DEVICE).eval()

def get_all_preds(model, loader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Running Inference"):
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrices(y_true, y_pred_res, y_pred_vit):
    cm_res = confusion_matrix(y_true, y_pred_res, normalize='true')
    cm_vit = confusion_matrix(y_true, y_pred_vit, normalize='true')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    fmt = '.2f'
    cmap = 'Blues'
    
    sns.heatmap(cm_res, annot=True, fmt=fmt, cmap=cmap, ax=axes[0], 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
    axes[0].set_title('ResNet-50 (CNN)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    sns.heatmap(cm_vit, annot=True, fmt=fmt, cmap='Reds', ax=axes[1], 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
    axes[1].set_title('ViT-Small (Transformer)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_yticks([])

    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, 'confusion_matrix_compare.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ 混淆矩阵已保存: {save_path}")

def plot_error_topology(y_true, y_pred, model_name, highlight_pair=('mel', 'nv')):
    """
    画一个有向图，节点是类别，边是误判流向。
    边的粗细代表误判的比例。
    重点高亮 mel -> nv 的边。
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    G = nx.DiGraph()

    for cls in CLASS_NAMES:
        G.add_node(cls)

    edge_colors = []
    edge_widths = []
    labels = {}
    
    threshold = 0.01 
    
    for i, true_cls in enumerate(CLASS_NAMES):
        for j, pred_cls in enumerate(CLASS_NAMES):
            if i == j: continue
            
            weight = cm[i, j]
            if weight > threshold:
                G.add_edge(true_cls, pred_cls, weight=weight)
                labels[(true_cls, pred_cls)] = f"{weight:.2f}"

                is_dangerous = (true_cls == highlight_pair[0] and pred_cls == highlight_pair[1])
                
                if is_dangerous:
                    edge_colors.append('#d62728')
                    edge_widths.append(weight * 15 + 2)
                else:
                    edge_colors.append('#999999')
                    edge_widths.append(weight * 8)

    plt.figure(figsize=(10, 8))
    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='#f0f0f0', edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, 
                           arrowstyle='->', arrowsize=20, connectionstyle="arc3,rad=0.1")
 
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='black', label_pos=0.3)
    
    plt.title(f'{model_name} Misclassification Topology\n(Red Edge: High-Risk {highlight_pair[0].upper()} -> {highlight_pair[1].upper()} Error)', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    
    save_path = os.path.join(FIGURE_DIR, f'topology_{model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 拓扑图已保存: {save_path}")

def main():

    print("📦 Preparing Validation Data...")
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    resnet = build_model('resnet50', os.path.join(WEIGHTS_DIR, 'resnet50_best.pth'))
    y_true, y_pred_res = get_all_preds(resnet, val_loader)

    vit = build_model('vit_small', os.path.join(WEIGHTS_DIR, 'vit_small_best.pth'))
    _, y_pred_vit = get_all_preds(vit, val_loader)

    print("🎨 Plotting Confusion Matrices...")
    plot_confusion_matrices(y_true, y_pred_res, y_pred_vit)

    print("🕸️ Plotting Topology Graphs...")
    plot_error_topology(y_true, y_pred_res, 'ResNet-50', highlight_pair=('mel', 'nv'))
    plot_error_topology(y_true, y_pred_vit, 'ViT-Small', highlight_pair=('mel', 'nv'))

if __name__ == "__main__":
    main()