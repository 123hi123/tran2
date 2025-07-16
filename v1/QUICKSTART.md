# 手語辨識系統 v1 - 快速開始指南

## 🚀 快速開始

### 1. 環境檢查
```bash
cd d:\file\畢業專題\tran2
python v1/run_pipeline_v1.py --check-only
```

### 2. 一鍵執行完整流程
```bash
python v1/run_pipeline_v1.py --step all
```

### 3. 分步執行（推薦用於調試）
```bash
# 步驟1：資料預處理
python v1/run_pipeline_v1.py --step preprocess

# 步驟2：模型訓練  
python v1/run_pipeline_v1.py --step train

# 步驟3：模型測試
python v1/run_pipeline_v1.py --step test
```

## 📁 輸出結構
執行完成後將產生以下資料夾結構：
```
v1/
├── processed_data/          # 預處理後的資料
│   ├── train_dataset.csv
│   ├── test_dataset.csv
│   └── label_encoder.pkl
├── models/                  # 訓練好的模型
│   ├── sign_language_gru_v1_YYYYMMDD_HHMMSS.pth
│   └── latest_model.pth
└── results/                 # 測試結果
    ├── test_results_YYYYMMDD_HHMMSS.csv
    ├── classification_report_YYYYMMDD_HHMMSS.txt
    ├── confusion_matrix_YYYYMMDD_HHMMSS.png
    └── class_performance_YYYYMMDD_HHMMSS.png
```

## ⚙️ 參數調整

如需調整訓練參數，編輯 `config_v1.py` 文件：

```python
# 主要參數
TRAINING_CONFIG = {
    "epochs": 50,           # 訓練週期
    "batch_size": 16,       # 批次大小
    "learning_rate": 0.001, # 學習率
    "sequence_length": 20,  # 序列長度
}
```

## 🔧 故障排除

### GPU記憶體不足
```python
# 在config_v1.py中調整
TRAINING_CONFIG["batch_size"] = 8  # 或更小
```

### 找不到資料檔案
```bash
# 確認dataset資料夾存在且包含CSV檔案
ls dataset/sign*.csv
```

### 依賴套件問題
```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn joblib
```

## 📊 預期結果

- **資料預處理**: 自動處理所有sign*.csv檔案，確保每類至少5筆資料
- **模型訓練**: 使用雙向GRU，預期訓練準確率70-90%+
- **模型測試**: 生成詳細評估報告和視覺化圖表

---
💡 **提示**: 首次執行建議使用較小的epochs數（如30）進行測試，確認流程正常後再進行完整訓練。
