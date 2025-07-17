# 手語辨識訓練流程 v1 文檔

## 📋 系統概述

手語辨識 v1 是基於 GRU（門控循環單元）的時序分類系統，能夠識別手語動作。系統採用滑動窗口技術和數據增強策略，支持變長序列處理。

## 🏗️ 系統架構

```
原始數據 (CSV) → 數據預處理 → 序列準備 → GRU模型 → 訓練/測試
```

## 📊 技術規格

| 項目 | 規格 |
|------|------|
| **模型架構** | 雙向 GRU + 全連接層 |
| **特徵維度** | 162 維 (姿態36 + 左手63 + 右手63) |
| **序列長度** | 20 幀 |
| **批次大小** | 16 |
| **學習率** | 0.001 |
| **訓練輪數** | 50 |

## 🔄 完整訓練流程

### 步驟 1: 環境準備

```bash
# 啟動 conda 環境
conda activate sign_language

# 確認 GPU 可用
python v1/verify_gpu_fix.py
```

**要求檢查:**
- ✅ PyTorch with CUDA 11.8
- ✅ GPU: NVIDIA RTX A2000 (6GB)
- ✅ Python 3.9+

### 步驟 2: 數據預處理

```bash
python v1/run_pipeline_v1.py --step preprocess
```

**預處理流程:**

1. **數據載入**
   - 搜尋 `dataset/` 目錄下所有 `sign*.csv` 文件
   - 合併所有 CSV 文件
   - 按 `sign_language` 和 `frame` 排序確保時間序列正確

2. **特徵定義**
   ```python
   # 總共 162 維特徵 (不包含 frame)
   - 姿態特徵: pose_tag11-22 (12點 × 3座標 = 36維)
   - 左手特徵: Left_hand_tag0-20 (21點 × 3座標 = 63維)  
   - 右手特徵: Right_hand_tag0-20 (21點 × 3座標 = 63維)
   ```

3. **數據增強**
   - 對少於 5 筆的類別進行數據增強
   - 使用高斯雜訊 (noise_factor=0.01)
   - 確保每個類別至少有 5 個樣本

4. **訓練測試分割**
   - 80% 訓練集, 20% 測試集
   - 按類別分層分割
   - 處理缺失值 (填充為 0)

5. **輸出文件**
   ```
   v1/processed_data/
   ├── train_dataset.csv      # 訓練集
   ├── test_dataset.csv       # 測試集
   └── label_encoder.pkl      # 標籤編碼器
   ```

### 步驟 3: 模型訓練

```bash
python v1/run_pipeline_v1.py --step train
```

**訓練流程:**

1. **序列準備**
   - 載入預處理後的訓練數據
   - 按 `sign_language` 分組
   - 使用滑動窗口切割序列
   
   **滑動窗口機制:**
   ```python
   # 例子: 231幀 → 212個序列 (sequence_length=20)
   序列1: 幀[0-19]   → 20幀
   序列2: 幀[1-20]   → 20幀  
   序列3: 幀[2-21]   → 20幀
   ...
   序列212: 幀[211-230] → 20幀
   ```

2. **模型架構**
   ```python
   SignLanguageGRU(
       input_size=162,      # 特徵維度
       hidden_size=128,     # 隱藏層大小
       num_layers=2,        # GRU 層數
       num_classes=N,       # 手語類別數
       dropout=0.3,         # Dropout 率
       bidirectional=True   # 雙向 GRU
   )
   ```

3. **訓練設定**
   ```python
   - 優化器: Adam (lr=0.001, weight_decay=1e-5)
   - 學習率調度器: ReduceLROnPlateau
   - 損失函數: CrossEntropyLoss
   - 批次大小: 16
   - 訓練輪數: 50
   ```

4. **實時監控**
   - 每個 epoch 顯示損失和準確率
   - 每 20 個 batch 顯示訓練進度
   - 自動保存最佳模型

5. **輸出文件**
   ```
   v1/models/
   ├── sign_language_gru_v1_YYYYMMDD_HHMMSS.pth  # 帶時間戳的模型
   ├── latest_model.pth                          # 最新模型連結
   └── training_curves_YYYYMMDD_HHMMSS.png       # 訓練曲線圖
   ```

### 步驟 4: 模型測試

```bash
python v1/run_pipeline_v1.py --step test
```

**測試流程:**

1. **模型載入**
   - 載入最新訓練的模型
   - 恢復標籤編碼器

2. **測試數據處理**
   - 載入測試集
   - 應用相同的序列準備流程

3. **評估指標**
   - 整體準確率
   - 各類別精確率、召回率、F1-score
   - 混淆矩陣
   - 分類報告

4. **結果可視化**
   - 混淆矩陣熱力圖
   - 各類別性能柱狀圖
   - 錯誤案例分析

### 步驟 5: 完整流程

```bash
python v1/run_pipeline_v1.py --step all
```

依序執行: 預處理 → 訓練 → 測試

## 📁 文件結構

```
v1/
├── data_preprocessing_v1.py    # 數據預處理模組
├── train_model_v1.py          # 模型訓練模組  
├── test_model_v1.py           # 模型測試模組
├── run_pipeline_v1.py         # 主執行腳本
├── config_v1.py               # 配置文件
├── processed_data/            # 預處理後的數據
│   ├── train_dataset.csv
│   ├── test_dataset.csv  
│   └── label_encoder.pkl
└── models/                    # 訓練好的模型
    ├── latest_model.pth
    └── training_curves_*.png
```

## 🔧 關鍵技術特色

### 1. 滑動窗口序列切割
- **目的**: 最大化數據利用，增加訓練樣本
- **實現**: 每次移動 1 幀，創建重疊序列
- **效果**: 231 幀 → 212 個訓練序列

### 2. 變長序列處理
- **長序列**: 使用滑動窗口切割
- **短序列**: 使用最後一幀重複填充
- **保證**: 所有序列統一為 20 幀

### 3. 數據增強策略
- **觸發條件**: 類別樣本數 < 5
- **方法**: 高斯雜訊增強
- **效果**: 提升小樣本類別的泛化能力

### 4. 特徵工程優化
- **排除無關特徵**: frame, source_video
- **專注座標特徵**: 162 維純座標數據
- **時間序列排序**: 確保 frame 順序正確

## 🎯 模型性能

### 預期指標
- **訓練準確率**: > 90%
- **測試準確率**: > 85%
- **收斂速度**: ~30 epochs
- **GPU 利用率**: ~60% (RTX A2000)

### 訓練監控
```bash
🚀 Epoch 1/50 開始...
  Batch [  0/XXX] | Loss: X.XXXX | Acc: XX.X%
  Batch [ 20/XXX] | Loss: X.XXXX | Acc: XX.X%
  ...
Epoch [  1/50] | Loss: X.XXXX | Accuracy: XX.XX% | LR: 0.001000
```

## 🚨 常見問題排除

### GPU 相關
```bash
# 檢查 CUDA 可用性
python v1/verify_gpu_fix.py

# 如果顯示不可用，重新安裝 PyTorch
conda uninstall pytorch torchvision torchaudio -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### 數據相關
```bash
# 檢查數據格式
python debug_training.py

# 確認 CSV 文件在正確位置
ls dataset/sign*.csv
```

### 記憶體相關
- 減少 batch_size: 16 → 8
- 減少 sequence_length: 20 → 15
- 檢查系統記憶體使用率

## 📚 相關文檔

- [GPU 故障排除指南](GPU_TROUBLESHOOTING.md)
- [序列準備詳細說明](SEQUENCE_PREPARATION_EXPLAINED.md)
- [變長序列處理](VARIABLE_LENGTH_HANDLING.md)
- [Conda 環境設置](CONDA_SETUP_GUIDE.md)
- [快速開始指南](QUICKSTART.md)

## 🔄 版本更新日誌

### v1.0 (2025-07-16)
- ✅ 實現基礎 GRU 模型
- ✅ 滑動窗口序列處理
- ✅ 數據增強策略
- ✅ GPU 加速支持
- ✅ 完整的訓練/測試流程
- ✅ 實時訓練監控
- ✅ 特徵維度優化 (排除 frame)

---

**作者**: GitHub Copilot  
**最後更新**: 2025-07-16  
**系統版本**: v1.0
