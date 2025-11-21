# 漫畫家畫風分類系統操作手冊 (User Guide)

本手冊旨在指導使用者如何從零開始，使用本專案提供的自動化工具鏈，完成從漫畫資料收集、清洗、預處理到模型訓練與部署的完整流程。本系統特別針對大規模資料集 (200+ 作者) 設計，具備多進程加速、中斷恢復 (Checkpoint) 與增量訓練功能。

## 1. 環境準備 (Environment Setup)

### 1.1 系統需求
*   **OS**: Windows 10/11 或 Linux
*   **GPU**: NVIDIA 顯示卡 (建議 VRAM 8GB 以上)
*   **Python**: 3.10+
*   **CUDA**: 11.8 或 12.x (需配合 PyTorch 版本)

### 1.2 安裝依賴
在專案根目錄下執行：
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
├── Manga_Dataset_Mixed/    # [輸出] 最終用於訓練的混合資料集
├── DL_Output_Models/       # [輸出] 訓練好的模型與 Checkpoints
├── whitelist.txt           # [設定] 作者白名單 (自動生成)
├── trained_history.txt     # [紀錄] 已訓練過的作者紀錄 (自動生成)
├── prepare_dataset.py      # [工具] 資料清洗與採樣
├── crop_faces.py           # [工具] 多進程人臉裁切
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
```bash
python prepare_dataset.py --num_samples_per_artist 400
```

### 2.3 互動式操作指引
1.  **白名單確認**：
    *   首次執行時，程式會自動掃描目錄並建立 `whitelist.txt`。
    *   **請依提示打開該檔案編輯**：保留要訓練的作者，若要略過某位作者，請在行首加上 `#`。
    *   編輯存檔後，回到終端機按 `Enter` 繼續。
    *   *注意：若未來新增作者資料夾，再次執行此腳本時會自動偵測並加入白名單，提示您確認。*

2.  **硬碟空間評估**：
    *   程式會根據白名單與歷史紀錄 (`trained_history.txt`) 估算所需空間。
    *   **新作者**：全量採樣 (預設 400 張)。
    *   **已訓練作者**：減量採樣 (預設 20%，即 80 張)，以節省空間並保留記憶。
    *   確認空間充足後，輸入 `y` 繼續。

---

## 3. 資料預處理 (Preprocessing)

此步驟將清洗後的圖片轉換為模型可用的特徵圖 (人臉)。

### 3.1 執行人臉裁切
使用多進程加速裁切：
```bash
python crop_faces.py --src_dir Manga_Dataset_Clean --dst_dir Manga_Dataset_Faces --num_workers 8
```
*   `--num_workers`: 設定使用的 CPU 核心數 (建議設為 CPU 核心數的 80%)。
*   此步驟會自動略過無法辨識人臉的圖片。

---

## 4. 模型訓練流程 (Model Training)

### 4.1 開始訓練
使用 `convnext_v2_tiny_local` 模型進行訓練，並啟用歷史紀錄功能。以下參數是經過優化的推薦設定：

```bash
python train.py --data_dir Manga_Dataset_Faces --model convnext_v2_tiny_local --epochs 50 --batch_size 32 --lr 1.2e-4 --weight_decay 0.05 --drop_path 0.2 --label_smoothing 0.1 --warmup_epochs 5 --early_stopping_patience 10 --amp --save_path final_model.pth --record_history
```
*   `--record_history`: **重要！** 訓練成功結束後，會將本次訓練的作者寫入 `trained_history.txt`，供下次 `prepare_dataset.py` 減量採樣使用。
*   `--amp`: 建議開啟混合精度訓練以節省顯存並加速。
*   `--lr`, `--weight_decay`, `--drop_path`: 針對 ConvNeXt V2 微調過的超參數。

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

### 5.2 C# 整合 (參考 CSharpPredict 專案)
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
