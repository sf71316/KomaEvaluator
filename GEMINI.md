# 漫畫家畫風分類專案

## 專案概覽 (Project Overview)

本專案旨在透過深度學習模型，對漫畫圖片進行分類，以識別其作者的畫風。專案比較了多種主流的卷積神經網路 (CNN) 與 Transformer 架構，包括 ResNet-50、Vision Transformer (ViT)、EfficientNet-B0 以及 ConvNeXt-Tiny。

經過一系列的優化，包括**增強資料增強**、**CosineAnnealingLR 學習率排程器**以及**標籤平滑 (Label Smoothing)**，最終的 **ConvNeXt-V2 Tiny (人臉特徵優化版)** 模型在一個包含七位藝術家的獨立測試集上，達到了 **94.76%** 的卓越準確率。此模型使用 `convnext_v2_tiny_faces_patience12.pth` 的設定。

主要使用的技術棧為 Python、PyTorch 和 scikit-learn。

## 環境設定 (Environment Setup)

### 1. 前置需求 (Prerequisites)

*   Python 3.10+
*   NVIDIA GPU
*   NVIDIA 顯示卡驅動程式
*   **CUDA Toolkit 12.1** (請務必安裝此版本以確保相容性)
*   **cuDNN for CUDA 12.x**

### 2. 安裝步驟 (Installation)

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ComicArtistTraning
    ```

2.  **建立並啟用虛擬環境:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    ```

3.  **安裝 PyTorch 及相關套件:**
    ```bash
    # 安裝支援 CUDA 12.1 的 PyTorch
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # 安裝 scikit-learn 用於模型評估
    pip install scikit-learn
    
    # 安裝 onnx 用於匯出模型
    pip install onnx
    ```

## 資料集結構 (Dataset Structure)

您需要將資料集整理成 `train`、`val` 和 `test` 三個子目錄，每個子目錄下再以藝術家姓名建立資料夾。結構如下：

```
Manga_Dataset/
├───train/
│   ├───Artist_A/
│   │   ├───image_a1.jpg
│   │   └───...
│   └───Artist_B/
│       └───...
├───val/
│   ├───Artist_A/
│   │   └───...
│   └───Artist_B/
│       └───...
└───test/
    ├───Artist_A/
    │   └───...
    └───Artist_B/
        └───...
```

## 模型訓練 (Training the Model)

使用 `train.py` 腳本進行模型訓練。腳本已整合多項優化技術。

**指令範例:**

```bash
# 訓練最終優化版的 ConvNeXt V2 Tiny (人臉特徵優化版，推薦)
python train.py --data_dir Manga_Dataset_Faces --model convnext_v2_tiny_local --epochs 50 --batch_size 32 --drop_path 0.1 --lr 8e-05 --weight_decay 0.05 --num_workers 0 --label_smoothing 0.1 --early_stopping_patience 12 --amp --bf16 --warmup_epochs 5 --save_path convnext_v2_tiny_faces_patience12.pth
```
*注意: 在 Windows 上建議將 `--num_workers` 設為 `0` 以避免多執行緒問題。*

## 模型評估 (Evaluating the Model)

使用 `evaluate.py` 腳本在**獨立的測試集**上評估訓練好的模型。

**指令範例:**

```bash
python evaluate.py --data_dir Manga_Dataset/test --model_path final_model.pth --num_classes 3 --model efficientnet_b0
```

## 最終模型性能 (Final Model Performance)

*   **整體準確率 (Accuracy)**: **94.76%** (基於平衡人臉資料集 `Manga_Dataset_Faces` 訓練，模型為 `convnext_v2_tiny_faces_patience12.pth`)

*(詳細的混淆矩陣和分類報告請參考評估日誌。)*

## 模型部署 (Model Deployment)

### 1. 匯出為 ONNX 格式 (Python)

若要將訓練好的 `.pth` 模型給 C# 或其他語言使用，需先將其轉換為 ONNX (Open Neural Network Exchange) 格式。

**指令範例:**
```bash
python export_to_onnx.py --model_path DL_Output_Models/convnext_v2_tiny_local/convnext_v2_tiny_faces_patience12.pth --output_path convnext_v2_tiny_faces_patience12.onnx --model convnext_v2_tiny_local --num_classes 7
```

### 2. 在 C# 中使用 ONNX 模型

專案中已包含一個 `CSharpPredict` 資料夾，作為在 .NET 環境中執行 ONNX 模型的範例。

**前置需求:**
*   .NET SDK
*   在 `CSharpPredict` 資料夾中安裝必要的 NuGet 套件:
    ```bash
    dotnet add package Microsoft.ML.OnnxRuntime
    dotnet add package SixLabors.ImageSharp
    ```

**執行範例:**
1.  將 `efficientnet_model.onnx` 檔案和一張測試圖片複製到 `CSharpPredict` 資料夾中。
2.  執行 C# 專案:
    ```bash
    cd CSharpPredict
    dotnet run
    ```
程式將會載入模型和圖片，並在主控台中輸出預測結果。
