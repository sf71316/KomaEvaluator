# 冠軍模型詳細資訊 (ConvNeXt V2 Tiny)

**模型檔案:** `DL_Output_Models/convnext_v2_tiny_local/convnext_v2_tiny_tuned_repro.pth`
**設定檔案:** `DL_Output_Models/convnext_v2_tiny_local/convnext_v2_tiny_tuned_repro_config.json`
**日期:** 2025-11-19

## 性能指標
*   **測試準確率 (Test Accuracy):** **89.52%** (在獨立測試集上評估)
*   **驗證準確率 (Validation Accuracy):** **85.71%** (訓練過程中的最佳 epoch)
*   **資料集大小:** 490 個訓練樣本 (共 7 位藝術家)
*   **訓練時間:** 約 9 分 8 秒

## 關鍵架構與功能
*   **模型:** ConvNeXt V2 Tiny (`convnext_v2_tiny_local`)
*   **預訓練權重:** ImageNet-1K (`convnextv2_tiny_1k_224_ema.pt`)
*   **精度:** 混合精度 (AMP)，啟用 **BFloat16 (BF16)**
*   **優化器:** AdamW

## 訓練超參數
| 參數 | 數值 | 描述 |
| :--- | :--- | :--- |
| `epochs` (訓練週期) | 50 | 在第 33 個 epoch 觸發提前停止 (Early Stopping) |
| `batch_size` (批次大小) | 32 | |
| `lr` (學習率) | 8e-05 | 初始學習率 |
| `weight_decay` (權重衰減) | 0.05 | 正則化參數 |
| `layer_decay` (層級學習率衰減) | 0.8 | Layer-wise Learning Rate Decay (LLRD) |
| `label_smoothing` (標籤平滑) | 0.1 | 損失函數正則化 |
| `early_stopping_patience` | 10 | 等待改善的 epoch 數 |

## 正則化與增強策略
*   **Drop Path:** **0.1** (比之前最佳模型的 0.2 更低，適合較小的資料集)
*   **Mixup:** **禁用** (False)
*   **Cutmix:** **禁用** (False)
*   **AutoAugment:** 標準 ImageNet 策略 (透過 `timm` 預設隱式啟用)

## 為什麼這個模型獲勝？
1.  **較低的 Drop Path (0.1):** 對於這樣大小的資料集 (約 500 張圖片)，0.2 的 drop path 可能過於激進，抑制了特徵學習。將其降低到 0.1 讓模型能捕捉到更細微的藝術風格。
2.  **無 Mixup/Cutmix:** 雖然 Mixup/Cutmix 是強大的工具，但它們通常需要更長的訓練時間才能收斂。在較短的訓練週期或有限的數據下，純粹的圖像增強通常能帶來更快且更穩定的收斂。
3.  **效率:** 在 10 分鐘內達到最先進的結果，使其非常適合快速迭代。

## 重現指令
```bash
python train.py --data_dir Manga_Dataset_7_Artists --model convnext_v2_tiny_local --save_path convnext_v2_tiny_tuned_repro.pth --epochs 50 --batch_size 32 --lr 8e-05 --weight_decay 0.05 --drop_path 0.1 --label_smoothing 0.1 --layer_decay 0.8 --early_stopping_patience 10 --amp --bf16
```