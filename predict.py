import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
import os

def load_model(model_path, model_name, num_classes, device):
    if model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights=None) # 不載入預訓練權重，因為我們將載入自己的
        num_ftrs = model.classifier[len(model.classifier) - 1].in_features
        model.classifier[len(model.classifier) - 1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'convnext_v2_tiny_local':
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

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, class_names, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0) # 增加批次維度
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        
    return class_names[preds[0]]

def main():
    parser = argparse.ArgumentParser(description='漫畫家畫風分類模型預測')
    parser.add_argument('--image_path', type=str, required=True, help='要預測的圖片路徑')
    parser.add_argument('--model_path', type=str, default='DL_Output_Models/convnext_tiny/final_model_convnext_tiny_bs16_7artists.pth', help='模型權重路徑')
    parser.add_argument('--model_name', type=str, default='convnext_tiny', help='模型名稱 (convnext_tiny, efficientnet_b0, efficientnet_v2_s)')
    parser.add_argument('--num_classes', type=int, default=7, help='資料集中的類別數量')
    parser.add_argument('--class_names', type=str, default='Reco,ShiBi,さくま司[Sakuma Tsukasa],みくに瑞貴[Mikuni Mizuki],スミヤ,木谷椎,紺菓[konka]', help='類別名稱，以逗號分隔')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    class_names = [name.strip() for name in args.class_names.split(',')]

    print(f"載入模型: {args.model_name} 從 {args.model_path}")
    model = load_model(args.model_path, args.model_name, args.num_classes, device)

    print(f"預測圖片: {args.image_path}")
    predicted_artist = predict_image(model, args.image_path, class_names, device)
    print(f"預測的漫畫家畫風是: {predicted_artist}")

if __name__ == '__main__':
    main()