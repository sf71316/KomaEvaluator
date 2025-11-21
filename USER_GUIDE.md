# 漫畫家畫風分類系統操作手冊 (User Guide)

本手冊旨在指導使用者如何從零開始，使用本專案提供的自動化工具鏈，完成從漫畫資料收集、清洗、預處理到模型訓練與部署的完整流程。本系統特別針對大規模資料集 (200+ 作者) 設計，具備多進程加速、中斷恢復 (Checkpoint) 與增量訓練功能。

## 1. 環境準備 (Environment Setup)

### 1.1 系統需求
*   **OS**: Windows 10/11 或 Linux
*   **GPU**: NVIDIA 顯示卡 (建議 VRAM 8GB 以上)
*   **Python**: 3.10+
*   **CUDA**: 11.8 或 12.x (需配合 PyTorch 版本)

### 1.2 安裝依賴
**強烈建議您建立並啟動 Python 虛擬環境，以避免套件衝突並保持系統環境整潔。**

1.  **建立虛擬環境 (Virtual Environment):**
    ```bash
    python -m venv venv
    ```

2.  **啟動虛擬環境:**
    *   **在 Windows 上:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **在 Linux / macOS 上:**
        ```bash
        source venv/bin/activate
        ```
    *   啟動後，您的命令提示符前會顯示 `(venv)`，表示您已在虛擬環境中。

3.  **安裝專案所需套件:**
    在虛擬環境中執行：
    ```bash
    pip install -r requirements.txt
    ```
*確保已安裝 `torch`, `torchvision`, `opencv-python`, `pillow`, `tqdm`, `psutil` 等核心套件。*

### 1.3 目錄結構說明
```
Project_Root/
├── MangaOriginalData/      # [輸入] 原始漫畫資料存放處 (每位作者一個子資料夾)
├── Manga_Dataset_Clean/    # [輸出] 清洗與採樣後的乾淨圖片
├── Manga_Dataset_Faces/    # [輸出] 裁切後的人臉資料集
├── Manga_Dataset_Patches/  # [輸出] 提取的紋理切塊資料集
├── Manga_Dataset_Mixed/    # [輸出] 最終用於訓練的混合資料集 (人臉 + 紋理)
├── DL_Output_Models/       # [輸出] 訓練好的模型與 Checkpoints
├── whitelist.txt           # [設定] 作者白名單 (自動生成)
├── trained_history.txt     # [紀錄] 已訓練過的作者紀錄 (自動生成)
├── prepare_dataset.py      # [工具] 資料清洗與採樣
├── crop_faces.py           # [工具] 多進程人臉裁切
├── prepare_patches.py      # [工具] 多進程紋理提取
├── merge_and_split.py      # [工具] 資料集合併
├── train.py                # [工具] 模型訓練
└── ...
```

---

## 2. 資料準備流程 (Data Preparation)

此步驟負責從原始壓縮檔中提取圖片、過濾壞檔、並根據歷史紀錄進行動態採樣。

### 2.1 準備原始資料
將整理好的漫畫作者資料夾放入 `MangaOriginalData` 中。支援 `.zip` 壓縮檔或直接圖片。

**預期目錄結構範例:**
```
MangaOriginalData/
├── Artist_A/             # 作者 A 的資料夾
│   ├── Work_A1.zip       # 作品 A1 的壓縮包
│   ├── Work_A2/          # 作品 A2 的資料夾
│   │   ├── image_01.jpg
│   │   └── image_02.png
│   └── loose_image.webp  # 散落的圖片
├── Artist_B/             # 作者 B 的資料夾
│   ├── Work_B1.zip
│   └── ...
└── ...
```

### 2.2 執行清洗腳本
此步驟是整個資料處理流程的第一步，負責從原始資料中提取、清洗、過濾並分割圖片。它將引導您完成白名單確認與硬碟空間評估。

**指令範例:**
```bash
python prepare_dataset.py \
    --original_data_dir ./MangaOriginalData \
    --target_dataset_dir ./Manga_Dataset_Clean \
    --num_samples_per_artist 400 \
    --whitelist whitelist.txt \
    --history trained_history.txt \
    --min_aspect_ratio 0.5 \
    --max_aspect_ratio 2.0
```

**參數說明:**

| 參數 | 必填 | 預設值 | 說明 |
| :--- | :---: | :--- | :--- |
| `--original_data_dir` | 是 | 無 | **原始資料來源目錄路徑**。您的作者資料夾應位於此目錄下 (可為絕對路徑或相對路徑)。 |
| `--target_dataset_dir` | 是 | `Manga_Dataset_Clean` | **目標資料集輸出目錄路徑**。清洗後的圖片將會輸出到此目錄，並自動分為 `train`, `val`, `test` 子目錄。 |
| `--num_samples_per_artist` | 否 | `None` | **每個藝術家目標採樣的圖片總數**。若該作者已被記錄在 `trained_history.txt` 中，則會自動調整為 `20%` 的採樣量。若未設定，則提取所有圖片。 |
| `--whitelist` | 否 | `whitelist.txt` | **白名單檔案名稱**。程式會建立或讀取此檔案，以決定哪些作者的資料應被處理。 |
| `--history` | 否 | `trained_history.txt` | **訓練歷史紀錄檔案名稱**。程式會讀取此檔案來判斷哪些作者是「已訓練」的，以應用減量採樣策略。 |
| `--min_aspect_ratio` | 否 | `0.5` | **最小圖片長寬比**。用於過濾掉過於狹長的圖片。 |
| `--max_aspect_ratio` | 否 | `2.0` | **最大圖片長寬比**。用於過濾掉過於寬扁的圖片。 |

---

## 3. 資料預處理 (Preprocessing)

此步驟將清洗後的圖片轉換為模型可用的特徵圖 (人臉)。

### 3.1 執行人臉裁切
使用多進程加速裁切：
```bash
python crop_faces.py --src_dir Manga_Dataset_Clean --dst_dir Manga_Dataset_Faces --num_workers 8
```
**參數說明:**

| 參數 | 必填 | 預設值 | 說明 |
| :--- | :---: | :--- | :--- |
| `--src_dir` | 是 | 無 | **來源圖片目錄**。通常是上一步產生的 `Manga_Dataset_Clean`。 |
| `--dst_dir` | 是 | 無 | **輸出目錄**。裁切後的人臉圖片將存放在此處。 |
| `--num_workers` | 否 | CPU 核心數 | **並行處理的進程數量**。建議設為 CPU 核心數的 80% 以獲得最佳效能。 |
| `--cascade` | 否 | `lbpcascade_animeface.xml` | **OpenCV Cascade 檔案路徑**。用於人臉偵測的模型檔。 |
| `--min_size` | 否 | `40` | **最小人臉尺寸**。小於此尺寸的人臉將被忽略。 |

### 3.2 執行紋理提取
提取高品質的紋理切塊，以捕捉畫風的筆觸與網點特徵。
```bash
python prepare_patches.py --src_dir Manga_Dataset_Clean --dst_dir Manga_Dataset_Patches --num_workers 8 --target_count 400
```

**參數說明:**

| 參數 | 必填 | 預設值 | 說明 |
| :--- | :---: | :--- | :--- |
| `--src_dir` | 是 | 無 | **來源圖片目錄**。通常是 `Manga_Dataset_Clean`。 |
| `--dst_dir` | 是 | 無 | **輸出目錄**。提取的紋理切塊將存放在此處。 |
| `--num_workers` | 否 | CPU 核心數 | **並行處理的進程數量**。 |
| `--patch_size` | 否 | `224` | **切塊大小** (像素)。建議與模型輸入大小一致。 |
| `--target_count` | 否 | `400` | **每位畫師的目標切塊數量**。 |

### 3.3 合併資料集 (混合訓練)
將「人臉特徵」與「紋理特徵」合併為單一訓練集，通常能獲得最佳的畫風識別效果。

```bash
python merge_and_split.py --faces_dir Manga_Dataset_Faces --patches_dir Manga_Dataset_Patches --output_dir Manga_Dataset_Mixed --split_ratio 0.5
```

**參數說明:**

| 參數 | 必填 | 預設值 | 說明 |
| :--- | :---: | :--- | :--- |
| `--faces_dir` | 是 | 無 | **人臉資料集目錄** (例如 `Manga_Dataset_Faces`)。 |
| `--patches_dir` | 是 | 無 | **紋理切塊資料集目錄** (例如 `Manga_Dataset_Patches`)。 |
| `--output_dir` | 是 | 無 | **合併後的輸出目錄**。這將作為訓練的輸入 (例如 `Manga_Dataset_Mixed`)。 |
| `--split_ratio` | 否 | `0.5` | **人臉與紋理切塊的混合比例**。例如，`0.5` 表示各佔 50%。 |

---

## 4. 模型訓練流程 (Model Training)

### 4.1 開始訓練

使用 `convnext_v2_tiny_local` 模型進行訓練，並啟用歷史紀錄功能。以下參數是經過優化的推薦設定：



```bash

python train.py --data_dir Manga_Dataset_Mixed --model convnext_v2_tiny_local --epochs 50 --batch_size 32 --lr 1.2e-4 --weight_decay 0.05 --drop_path 0.2 --label_smoothing 0.1 --warmup_epochs 5 --early_stopping_patience 10 --amp --save_path final_model.pth --record_history

```



**參數說明:**



| 參數 | 必填 | 預設值 | 說明 |

| :--- | :---: | :--- | :--- |

| `--data_dir` | 是 | `Manga_Dataset` | **訓練資料集目錄**。推薦使用合併後的 `Manga_Dataset_Mixed`。 |

| `--model` | 否 | `efficientnet_b0` | **使用的模型架構**。推薦使用 `convnext_v2_tiny_local`。 |
| `--epochs` | 否 | `20` | **訓練總輪數**。 |
| `--batch_size` | 否 | `32` | **批次大小**。視顯存大小調整，越大越快但顯存需求越高。 |
| `--lr` | 否 | `0.001` | **學習率**。ConvNeXt V2 推薦設為 `1.2e-4`。 |
| `--save_path` | 否 | `final_model.pth` | **模型儲存檔案名稱**。將儲存在 `DL_Output_Models/[model_name]/` 下。 |
| `--record_history` | 否 | `False` | **啟用歷史紀錄**。訓練成功後，將本次作者寫入 `trained_history.txt`。 |
| `--amp` | 否 | `False` | **啟用混合精度訓練**。強烈建議開啟，可節省顯存並加速。 |
| `--resume_path` | 否 | `None` | **恢復訓練的 Checkpoint 路徑**。用於中斷後繼續訓練。 |

### 4.2 中斷與恢復 (Checkpoint & Resume)
*   **手動中斷**：在終端機按 `Ctrl+C`。程式會自動儲存當前進度 (Checkpoint) 並顯示恢復指令。
*   **恢復訓練**：使用 `--resume_path` 參數指定 Checkpoint 檔案：
    ```bash
    python train.py ... --resume_path DL_Output_Models/convnext_v2_tiny_local/checkpoint_last.pth
    ```
    程式會自動載入模型權重、優化器狀態、Epoch 數，並從中斷點繼續訓練。

### 4.3 監控訓練進度(如果有需要)
使用 TensorBoard 查看 Loss 和 Accuracy (包含 Top-1 和 Top-5)：
```bash
tensorboard --logdir runs
```
瀏覽器打開 `http://localhost:6006`。

---

## 5. 模型部署與使用 (Deployment)

### 5.1 匯出為 ONNX
將 PyTorch 模型轉換為通用的 ONNX 格式，以便在 C# 或其他語言中使用。
```bash
python export_to_onnx.py --model_path DL_Output_Models/convnext_v2_tiny_local/final_model.pth --output_path mysmodel.onnx --model convnext_v2_tiny_local --num_classes <類別數量>
```

**參數說明:**

| 參數 | 必填 | 預設值 | 說明 |
| :--- | :---: | :--- | :--- |
| `--model_path` | 是 | 無 | **PyTorch 模型檔案路徑** (`.pth`)。 |
| `--output_path` | 是 | 無 | **輸出的 ONNX 模型檔案路徑** (`.onnx`)。 |
| `--model` | 是 | 無 | **模型架構名稱**。必須與訓練時使用的名稱一致 (如 `convnext_v2_tiny_local`)。 |
| `--num_classes` | 是 | 無 | **分類類別數量**。必須與訓練時的類別數一致。 |
1.  將 `mysmodel.onnx` 複製到 C# 專案目錄。
2.  確保 C# 專案安裝了 `Microsoft.ML.OnnxRuntime`。
3.  更新 C# 程式碼中的類別列表 (`classNames`)，需與 Python 訓練時的順序一致 (可參考 `train.py` 輸出或產生的 `classes.txt`)。

---

## 6. 常見問題與維護 (FAQ)

### Q: 如何新增一位新作者？
1.  將新作者資料夾放入 `MangaOriginalData`。
2.  重新執行 `python prepare_dataset.py ...`。
3.  程式會提示發現新作者，並將其加入 `whitelist.txt`。
4.  確認後，程式會對新作者進行全量採樣，對舊作者進行減量採樣。
5.  執行 `crop_faces.py` 和 `train.py` (建議使用 `--resume_path` 接續之前的模型微調，或重新訓練)。

### Q: 訓練到一半當機怎麼辦？
請檢查 `DL_Output_Models` 目錄下的 `checkpoint_last.pth`，使用 `python train.py ... --resume_path ...` 指令恢復訓練。

### Q: 硬碟空間不足怎麼辦？
1.  檢查 `trained_history.txt`，確保舊作者都有被記錄，這樣 `prepare_dataset.py` 才會執行減量採樣。
2.  手動清理 `temp_unzip` 或舊的 `Test_*` 測試資料夾。
3.  在 `prepare_dataset.py` 中使用更嚴格的 `--num_samples_per_artist` 限制。
