import json
import matplotlib.pyplot as plt
import numpy as np
import os

LOG_DIR = './checkpoints'
RESNET_LOG = os.path.join(LOG_DIR, 'resnet50_history.json')
VIM_LOG = os.path.join(LOG_DIR, 'vit_small_history.json')
SAVE_PATH = './figures/experiment_1_dynamics.png'


def load_log(path):
    if not os.path.exists(path):
        print(f"❌ 找不到文件: {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)

def plot_dynamics():
    res_data = load_log(RESNET_LOG)
    vit_data = load_log(VIM_LOG)
    
    if res_data is None or vit_data is None:
        return

    epochs = range(1, len(res_data['train_loss']) + 1)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    c_res = '#1f77b4' 
    c_vit = '#d62728' 

    ax1.plot(epochs, res_data['train_loss'], label='ResNet-50 (CNN)', color=c_res, linestyle='--', linewidth=2, alpha=0.7)
    ax1.plot(epochs, vit_data['train_loss'], label='ViT-Small (Transformer)', color=c_vit, linestyle='-', linewidth=2.5)

    ax1.fill_between(epochs, res_data['train_loss'], vit_data['train_loss'], where=np.array(res_data['train_loss']) > np.array(vit_data['train_loss']), 
                     facecolor=c_vit, alpha=0.1, interpolate=True)

    ax1.set_title('Training Optimization Dynamics', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=12)
    ax1.legend(fontsize=12)
 
    ax1.annotate('Rapid Global Feature Learning\n(Transformer Advantage)', 
                 xy=(3, vit_data['train_loss'][2]), 
                 xytext=(10, vit_data['train_loss'][2] + 0.3),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 fontsize=11, fontweight='bold', color='#333333')

    ax2.plot(epochs, res_data['val_acc'], label='ResNet-50 (CNN)', color=c_res, linestyle='--', linewidth=2, alpha=0.8)
    ax2.plot(epochs, vit_data['val_acc'], label='ViT-Small (Transformer)', color=c_vit, linestyle='-', linewidth=2.5)

    final_res = res_data['val_acc'][-1]
    final_vit = vit_data['val_acc'][-1]
  
    ax2.vlines(x=len(epochs), ymin=min(final_res, final_vit), ymax=max(final_res, final_vit), colors='gray', linestyles=':', linewidth=2)
    
    mid_point = (final_res + final_vit) / 2
    ax2.annotate(f'CNN Inductive Bias\nAdvantage (+{final_res - final_vit:.1f}%)', 
                 xy=(len(epochs), mid_point), 
                 xytext=(len(epochs)-15, mid_point-1),
                 arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"),
                 fontsize=11, fontstyle='italic', backgroundcolor='#f0f0f0')

    ax2.set_title('Validation Accuracy & Generalization', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(loc='lower right', fontsize=12)

    plt.tight_layout()
    plt.savefig(SAVE_PATH, bbox_inches='tight')
    print(f"✅ 新版分析图已生成: {SAVE_PATH}")

if __name__ == "__main__":
    plot_dynamics()