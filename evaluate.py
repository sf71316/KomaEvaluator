import torch
import torch.nn as nn
import os
import sys

from torchvision import datasets, transforms, models

from torch.utils.data import DataLoader
import os
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import json


def evaluate_model(model_path, data_dir, num_classes, model_name, batch_size=32, num_workers=4):
    """
    評估預訓練的 ResNet-50 模型。

    Args:
        model_path (str): 預訓練模型權重的路徑。
        data_dir (str): 測試資料集的根目錄。
        num_classes (int): 資料集中的類別數量。
        batch_size (int): 批次大小。
        num_workers (int): 資料載入器的工作進程數量。
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"載入測試資料集來自: {data_dir}")
    test_dataset = datasets.ImageFolder(os.path.join(data_dir), data_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 載入模型並修改最後的分類層
    if model_name == 'resnet50':
        model = models.resnet50(weights=None) # 評估時不載入預訓練權重，因為我們載入的是自定義訓練的權重
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights=None)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'convnext_tiny': # 新增對 convnext_tiny 的支援
        model = models.convnext_tiny(weights=None)
        num_ftrs = model.classifier[len(model.classifier) - 1].in_features
        model.classifier[len(model.classifier) - 1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'convnext_tiny_v2': # 新增對 convnext_tiny_v2 的支援
        model = models.convnext_tiny(weights=None) # 注意這裡仍然是 convnext_tiny，但會載入 V2 權重
        num_ftrs = model.classifier[len(model.classifier) - 1].in_features
        model.classifier[len(model.classifier) - 1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'convnext_v2_tiny_local' or model_name == 'convnext_v2_tiny':
        import sys
        sys.path.append('ConvNeXt_V2_Official')
        # Mock MinkowskiEngine to avoid ImportError
        class MockSparseTensor:
            pass
        class MockMinkowskiEngine:
            SparseTensor = MockSparseTensor
        sys.modules['MinkowskiEngine'] = MockMinkowskiEngine()
        from models.convnextv2 import convnextv2_tiny
        model = convnextv2_tiny(num_classes=num_classes)

    else:
        raise ValueError(f"不支援的模型: {model_name}")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功載入模型權重來自: {model_path}")
    else:
        print(f"錯誤: 找不到模型權重於 {model_path}。請提供有效的模型路徑。")
        return

    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    all_preds = torch.tensor([], dtype=torch.long).to(device)
    all_labels = torch.tensor([], dtype=torch.long).to(device)

    print("開始評估...")
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

            all_preds = torch.cat((all_preds, preds))
            all_labels = torch.cat((all_labels, labels))

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    print(f"評估完成。")
    print(f"測試損失: {epoch_loss:.4f} 測試準確度: {epoch_acc:.4f}")

    # 計算並顯示混淆矩陣和分類報告
    class_names = test_dataset.classes
    all_labels_np = all_labels.cpu().numpy()
    all_preds_np = all_preds.cpu().numpy()

    print("\n混淆矩陣:")
    cm = confusion_matrix(all_labels_np, all_preds_np)
    print(cm)

    print("\n分類報告:")
    report = classification_report(all_labels_np, all_preds_np, target_names=class_names, digits=4)
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ResNet-50 Evaluation')
    parser.add_argument('--data_dir', type=str, default='Manga_Dataset/test',
                        help='測試資料集的根目錄 (預設: Manga_Dataset/test)')
    parser.add_argument('--model_path', type=str, default='saved_model.pth',
                        help='預訓練模型權重的路徑 (預設: saved_model.pth)')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='要評估的模型名稱 (支援 resnet50, vit_b_16, efficientnet_b0, convnext_tiny, convnext_tiny_v2)')
    parser.add_argument('--num_classes', type=int, help='資料集中的類別數量')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='每個批次的圖像數量 (預設: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='資料載入器的工作進程數量 (預設: 4)')
    parser.add_argument('--show_config', action='store_true',
                        help='是否顯示模型的訓練設定')
    args = parser.parse_args()

    if args.show_config:
        config_path = os.path.splitext(args.model_path)[0] + '_config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("模型訓練設定:")
            print(json.dumps(config, indent=4))
        else:
            print(f"錯誤: 找不到設定檔於 {config_path}")
    else:
        if args.num_classes is None:
            parser.error("--num_classes is required when not using --show_config")
        evaluate_model(args.model_path, args.data_dir, args.num_classes, args.model, args.batch_size, args.num_workers)
