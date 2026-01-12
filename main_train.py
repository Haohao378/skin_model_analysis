import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
import timm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description="Skin Lesion Classification: ResNet vs Vim")
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'vit_small'], help='选择模型架构')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch Size (L40可以设大点)')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--data_dir', type=str, default='./data/formatted', help='数据路径')
    parser.add_argument('--weights_dir', type=str, default='./models/pretrained', help='预训练权重存放路径')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存路径')
    return parser.parse_args()

def build_model(model_name, num_classes, weights_dir):
    print(f"🏗️ 正在构建模型: {model_name}...")
    
    if model_name == 'resnet50':

        model = models.resnet50(pretrained=False)
        pth_path = os.path.join(weights_dir, 'resnet50.pth')
        if os.path.exists(pth_path):
            print(f"📥 加载 ResNet 离线权重: {pth_path}")
            state_dict = torch.load(pth_path, map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict, strict=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'vit_small': 
        model = timm.create_model(
            'deit_small_patch16_224', 
            pretrained=False, 
            num_classes=num_classes
        )
        
        pth_path = os.path.join(weights_dir, 'deit_small.pth')
        if os.path.exists(pth_path):
            print(f"📥 加载 ViT (DeiT) 离线权重: {pth_path}")
            checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
            
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
 
            keys_to_remove = ['head.weight', 'head.bias']
            for key in keys_to_remove:
                if key in checkpoint:
                    print(f"✂️  移除预训练权重中的不匹配层: {key}")
                    del checkpoint[key]

            msg = model.load_state_dict(checkpoint, strict=False)
            print(f"权重加载完毕，未匹配层（预期应包含head）: {msg.missing_keys}")
            
        else:
            print("⚠️ 未找到权重，使用随机初始化！")
            
    return model.to(DEVICE)

def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Training Epoch {epoch+1}", leave=False)
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
        
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), 100. * correct / total


def main():
    args = get_args()

    print("📦 正在准备数据...")
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=transform_train)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = build_model(args.model, num_classes=7, weights_dir=args.weights_dir)
 
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
   
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"🚀 开始训练 {args.model} | GPU: {torch.cuda.get_device_name(0)}")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion)
       
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, f"{args.model}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"🔥 新的最佳模型已保存! ({best_acc:.2f}%)")
            
    total_time = time.time() - start_time
    print(f"\n✅ 训练完成! 总耗时: {total_time/60:.2f} 分钟. 最佳精度: {best_acc:.2f}%")
    
    log_path = os.path.join(args.save_dir, f"{args.model}_history.json")
    with open(log_path, 'w') as f:
        json.dump(history, f)
    print(f"📊 训练日志已保存至: {log_path}")

if __name__ == '__main__':
    main()