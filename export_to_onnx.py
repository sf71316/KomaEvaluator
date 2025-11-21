import torch
import torch.nn as nn
from torchvision import models
import argparse
import sys
import os

# 將專案根目錄添加到 sys.path，以便找到 models 模組
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))

def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX format')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        help='模型名稱 (支援 resnet50, vit_b_16, efficientnet_b0)')
    parser.add_argument('--model_path', type=str, default='efficientnet_model.pth',
                        help='預訓練模型權重的路徑')
    parser.add_argument('--output_path', type=str, default='efficientnet_model.onnx',
                        help='匯出的 ONNX 模型路徑')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='資料集中的類別數量')
    args = parser.parse_args()

    # 1. 載入與訓練時相同的模型架構
    print(f"正在載入模型架構: {args.model}...")
    if args.model == 'resnet50':
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.num_classes)
    elif args.model == 'vit_b_16':
        model = models.vit_b_16(weights=None)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, args.num_classes)
    elif args.model == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, args.num_classes)
    elif args.model == 'convnext_v2_tiny_local':
        from ConvNeXt_V2_Official.models.convnextv2 import convnextv2_tiny
        model = convnextv2_tiny(num_classes=args.num_classes, drop_path_rate=0.0) # 匯出時 drop_path_rate 設為 0.0
    else:
        raise ValueError(f"不支援的模型: {args.model}")

    # 2. 載入訓練好的權重
    print(f"正在從 '{args.model_path}' 載入權重...")
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval() # 設定為評估模式

    # 3. 建立一個符合模型輸入尺寸的虛擬輸入張量
    # 我們的模型需要 (batch_size, channels, height, width)，這裡 batch_size 為 1
    dummy_input = torch.randn(1, 3, 224, 224)

    # 4. 匯出為 ONNX 格式
    print(f"正在將模型匯出至 '{args.output_path}'...")
    torch.onnx.export(model,               # 要匯出的模型
                      dummy_input,         # 模型的虛擬輸入
                      args.output_path,    # 儲存路徑
                      export_params=True,  # 儲存訓練好的權重
                      opset_version=11,    # ONNX 版本
                      do_constant_folding=True, # 是否執行常數折疊以進行優化
                      input_names = ['input'],   # 輸入張量的名稱
                      output_names = ['output'], # 輸出張量的名稱
                      dynamic_axes={'input' : {0 : 'batch_size'},    # 動態軸
                                    'output' : {0 : 'batch_size'}})

    print("模型成功匯出為 ONNX 格式！")

if __name__ == '__main__':
    main()
