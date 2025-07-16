# 手語辨識系統 v1

基於圖片左邊計畫創作的手語辨識訓練代碼，使用GRU深度學習模型進行手語動作分類。

## 系統概述

本系統包含三個主要組件：
1. **資料預處理** (`data_preprocessing_v1.py`) - 處理CSV資料集，分割訓練/測試集，進行資料增強
2. **模型訓練** (`train_model_v1.py`) - 使用GRU模型訓練手語分類器
3. **模型測試** (`test_model_v1.py`) - 評估模型性能，生成詳細報告

## 資料格式

### 輸入資料
- CSV文件名稱必須以"sign"開頭
- 包含以下欄位：
  - `sign_language`: 手語類別標籤（目標變數）
  - `source_video`: 來源影片（訓練時會被排除）
  - `frame`: 影格編號
  - `pose_tag11_x` ~ `pose_tag22_z`: 姿態關鍵點（11-22號點的x,y,z座標）
  - `Left_hand_tag0_x` ~ `Left_hand_tag20_z`: 左手關鍵點（0-20號點的x,y,z座標）
  - `Right_hand_tag0_x` ~ `Right_hand_tag20_z`: 右手關鍵點（0-20號點的x,y,z座標）

### 特徵維度
- 姿態特徵：12個點 × 3軸 = 36維
- 左手特徵：21個點 × 3軸 = 63維  
- 右手特徵：21個點 × 3軸 = 63維
- 總計：163維特徵（加上frame欄位）

## 安裝需求

### 硬體需求
- **GPU**: NVIDIA RTX A2000 (6GB VRAM) 或更好
- **記憶體**: 至少8GB RAM（建議16GB+）
- **儲存**: 至少2GB可用空間

### 軟體環境
```bash
# 建立Python 3.9環境
conda create -n sign_language python=3.9 -y
conda activate sign_language

# 安裝PyTorch (CUDA 11.4)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

# 安裝其他套件
conda install numpy pandas matplotlib seaborn jupyter scikit-learn -y
pip install joblib
```

## 使用方式

### 方法1：使用流程執行器（推薦）

```bash
# 檢查環境
python v1/run_pipeline_v1.py --check-only

# 執行完整流程
python v1/run_pipeline_v1.py --step all

# 或分步執行
python v1/run_pipeline_v1.py --step preprocess  # 資料預處理
python v1/run_pipeline_v1.py --step train       # 模型訓練  
python v1/run_pipeline_v1.py --step test        # 模型測試
```

### 方法2：分別執行各個腳本

```bash
# 1. 資料預處理
cd v1
python data_preprocessing_v1.py

# 2. 模型訓練
python train_model_v1.py

# 3. 模型測試
python test_model_v1.py
```

## 輸出結果

### 資料預處理 (`v1/processed_data/`)
- `train_dataset.csv`: 訓練資料集
- `test_dataset.csv`: 測試資料集  
- `label_encoder.pkl`: 標籤編碼器

### 模型訓練 (`v1/models/`)
- `sign_language_gru_v1_YYYYMMDD_HHMMSS.pth`: 帶時間戳的模型檔案
- `latest_model.pth`: 最新模型連結
- `training_curves_YYYYMMDD_HHMMSS.png`: 訓練曲線圖

### 模型測試 (`v1/results/`)
- `test_results_YYYYMMDD_HHMMSS.csv`: 詳細測試結果
- `classification_report_YYYYMMDD_HHMMSS.txt`: 分類報告
- `confusion_matrix_YYYYMMDD_HHMMSS.png`: 混淆矩陣圖
- `class_performance_YYYYMMDD_HHMMSS.png`: 各類別性能圖

## 模型架構

### GRU模型特點
- **雙向GRU**: 可以同時利用過去和未來的時序資訊
- **多層結構**: 2層GRU增強特徵提取能力
- **正規化**: Dropout防止過擬合
- **序列處理**: 專門處理時序手語動作資料

### 模型參數
- 隱藏層大小：128
- GRU層數：2層
- Dropout率：0.3
- 序列長度：20幀
- 批次大小：16
- 學習率：0.001

## 資料增強策略

對於不足5筆資料的類別：
1. 複製原始資料
2. 添加高斯雜訊（標準差=0.01）
3. 確保每個類別至少有5筆訓練資料

## 效能優化

### 記憶體優化
- 適當的批次大小（16）避免GPU記憶體溢出
- 序列長度限制（20幀）平衡效能和準確度

### 訓練優化
- 學習率調度器：ReduceLROnPlateau
- 早停機制：防止過擬合
- 權重衰減：L2正規化

## 故障排除

### 常見問題

1. **CUDA記憶體不足**
   ```
   解決方案：減少batch_size到8或4
   ```

2. **找不到CSV文件**
   ```
   確認dataset資料夾存在且包含sign*.csv文件
   ```

3. **套件缺失**
   ```bash
   pip install torch pandas numpy scikit-learn matplotlib seaborn joblib
   ```

4. **測試資料集為空**
   ```
   檢查是否有足夠的資料進行訓練/測試分割
   某些類別可能資料量太少，全部用於訓練
   ```

### 調整建議

根據您的硬體環境調整參數：

```python
# 在train_model_v1.py中調整
trainer.run_training(
    epochs=30,          # GPU記憶體不足時減少
    batch_size=8,       # 記憶體不足時減少
    learning_rate=0.001,
    sequence_length=15  # 可以減少以節省記憶體
)
```

## 評估指標

系統提供以下評估指標：
- **準確率 (Accuracy)**: 整體分類正確率
- **精確度 (Precision)**: 各類別預測準確度
- **召回率 (Recall)**: 各類別識別完整度  
- **F1分數**: 精確度和召回率的調和平均
- **混淆矩陣**: 詳細的分類錯誤分析

## 系統限制

基於env.md的系統環境：
- GPU記憶體：6GB (RTX A2000)
- 系統記憶體：48GB
- 建議最大批次大小：32
- 建議最大序列長度：30

## 未來改進方向

1. **資料增強**: 添加更多樣化的資料增強技術
2. **模型架構**: 嘗試Transformer或LSTM模型
3. **特徵工程**: 添加手部角度、速度等衍生特徵
4. **集成學習**: 結合多個模型提升性能
5. **即時推理**: 優化模型以支援即時手語辨識

## 授權

本專案為畢業專題代碼，請遵循相關學術使用規範。
