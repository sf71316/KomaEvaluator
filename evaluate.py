import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import time
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import timm # 導入 timm 庫

# ==============================================================================
# Helper Functions
# ==============================================================================

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# ==============================================================================
# Model Evaluation Function
# ==============================================================================

def evaluate_model(model_path, data_dir, num_classes, model_name, batch_size=32, num_workers=0, device='cuda:0'):
    print(f"使用裝置: {device}")
    print(f"載入測試資料集來自: {data_dir}")

    # 準備資料轉換
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_dataset = datasets.ImageFolder(data_dir, data_transforms)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = image_dataset.classes

    # --- 模型建立 (使用 timm) ---
    print(f"建立模型: {model_name}")
    try:
        model = timm.create_model(
            model_name,
            pretrained=False, # 預訓練權重從 model_path 載入
            num_classes=num_classes
        )
        print(f"成功使用 timm 建立模型: {model_name}")
    except Exception as e:
        print(f"timm 建立模型失敗: {e}")
        print("請確認模型名稱是否正確 (參考 timm.list_models())")
        return

    # 載入訓練好的模型權重
    if not os.path.exists(model_path):
        print(f"錯誤: 模型權重檔案 {model_path} 不存在。")
        return

    print(f"載入模型權重來自: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    # 處理 DataParallel 的 'module.' 前綴
    if list(state_dict.keys())[0].startswith('module.') and not list(model.state_dict().keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False) # strict=False 允許 Head 層形狀不匹配
    model = model.to(device)
    model.eval() # 設定為評估模式

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 計算總體準確率 (Top-1)
    total_correct = np.sum(np.array(all_preds) == np.array(all_labels))
    total_samples = len(all_labels)
    accuracy = total_correct / total_samples
    print(f'\n總體準確率: {accuracy:.4f}')

    # 生成分類報告和混淆矩陣
    print('\n分類報告:')
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print('\n混淆矩陣:')
    print(confusion_matrix(all_labels, all_preds))

# ==============================================================================
# Main Function
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型評估 (基於 timm)')
    parser.add_argument('--model_path', type=str, required=True, help='訓練好的模型權重檔案路徑 (.pth)')
    parser.add_argument('--data_dir', type=str, required=True, help='測試資料集路徑 (例如 Manga_Dataset/test)')
    parser.add_argument('--num_classes', type=int, required=True, help='分類類別數量')
    parser.add_argument('--model', type=str, default='convnextv2_tiny.fcmae_ft_in22k_in1k', help='模型架構名稱 (timm 格式)')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='資料載入的執行緒數量 (Windows 建議設為 0)')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    evaluate_model(args.model_path, args.data_dir, args.num_classes, args.model, args.batch_size, args.num_workers, device)