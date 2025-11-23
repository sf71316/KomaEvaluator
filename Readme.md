# KomaEvaluator - 大規模漫畫畫風分類系統

**KomaEvaluator** 是一個專為大規模漫畫圖像分類設計的深度學習專案。它能夠自動化地從原始漫畫檔案中提取特徵（人臉與紋理），並訓練出高準確度的畫風識別模型。



## 🌟 核心特色

*   **🚀 大規模高效能**：
    *   **多進程預處理**：`crop_faces.py` 與 `prepare_patches.py` 支援多核心並行處理，速度提升 10 倍以上。
    *   **混合精度訓練 (AMP)**：支援 FP16/BF16，大幅降低顯存需求並加速訓練。
    *   **後台優先度運行**：支援 `--low_priority` 參數，降低進程優先度，不影響前台操作。
*   **🛡️ 穩健的訓練流程**：
    *   **中斷恢復 (Resume)**：訓練途中當機或中斷 (`Ctrl+C`)，可隨時從 Checkpoint 恢復，無需重頭再來。
    *   **自動資料清洗**：內建 `check_dataset_health.py` 掃描壞檔，`prepare_dataset.py` 支援互動式白名單過濾。
*   **🔄 增量訓練機制**：
    *   智慧記錄已訓練過的作者，再次準備資料時自動執行 **減量採樣 (20%)**，大幅節省硬碟空間與訓練時間。
*   **🧠 AI模型架構**：
    *   預設採用 **ConvNeXt V2**  模型，針對漫畫線條與網點特徵有極佳的表現。

## 🛠️ 快速開始

詳細操作請參閱 [📖 操作手冊 (USER_GUIDE.md)](USER_GUIDE.md)。

### 1. 環境安裝
```bash
pip install -r requirements.txt
```

### 2. 資料準備 (Data Preparation)
將原始作者資料夾放入 `MangaOriginalData/`，然後執行：
```bash
# 互動式腳本：自動掃描、建立白名單、評估硬碟空間 (可選 --low_priority 後台運行)
python prepare_dataset.py --num_samples_per_artist 400 --low_priority
```

### 3. 資料預處理 (Preprocessing)
使用一鍵腳本完成人臉裁切、紋理提取與資料集合併：
```bash
python process_features.py --src_dir Manga_Dataset_Clean --output_dir Manga_Dataset_Mixed --num_workers 4 --low_priority --suppress_libpng_warnings
```

### 4. 模型訓練 (Training)
```bash
# 使用合併後的資料集進行訓練 (可選 --low_priority 後台運行)
python train.py --data_dir Manga_Dataset_Mixed --model convnextv2_tiny.fcmae_ft_in22k_in1k --epochs 50 --record_history --low_priority
```

### 5. 部署 (Deployment)
```bash
# 匯出為 ONNX
python export_to_onnx.py --model_path DL_Output_Models/.../final_model.pth ...
```
本專案包含一個 `CSharpPredict` 目錄，提供在 .NET 環境下載入 ONNX 模型進行預測的範例。

## 📂 目錄結構

*   `MangaOriginalData/`: 原始資料存放區
*   `Manga_Dataset_Clean/`: 清洗後的圖片
*   `Manga_Dataset_Faces/`: 人臉特徵資料集
*   `DL_Output_Models/`: 訓練輸出 (模型與 Checkpoints)
*   `USER_GUIDE.md`: **完整操作說明書**

## 📝 授權
MIT License