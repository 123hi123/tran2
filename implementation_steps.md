# 手語辨識訓練實施步驟

## 🎯 基於數據分析的具體行動計劃

### 第一步：立即開始的數據預處理（第1-2天）

#### 1.1 創建數據預處理模組
```python
# 文件：src/data_preprocessing.py
# 目的：解決66%右手缺失值和10%左手缺失值問題

class SignLanguagePreprocessor:
    def __init__(self):
        self.sequence_length = 30  # 基於分析，30幀約1秒
        self.stride = 15  # 滑動窗口，增加數據量
        
    def handle_missing_values(self, df):
        """
        智能缺失值處理策略：
        1. 時序線性插值（前後幀）
        2. 基於身體姿態的約束插值
        3. 如果整個序列缺失，使用身體中心點
        """
        
    def normalize_coordinates(self, df):
        """
        座標標準化：
        1. 以身體中心為原點
        2. 按身體大小縮放
        3. 消除個人差異
        """
        
    def create_sequences(self, df):
        """
        序列生成：
        1. 滑動窗口切割
        2. 固定長度padding/truncation
        3. 標籤對應
        """
```

**為什麼這樣設計**：
- **30幀序列**：基於視頻分析，大部分手語動作在1秒內完成
- **滑動窗口**：從121萬幀可以生成約8萬個序列，增加訓練數據
- **智能插值**：比簡單刪除或均值填充更保留時序信息

#### 1.2 創建數據載入器
```python
# 文件：src/data_loader.py
# 目的：高效處理3GB數據，避免記憶體溢出

class SignLanguageDataset(Dataset):
    def __init__(self, csv_files, sequence_length=30, stride=15):
        # 延遲載入：只載入索引，需要時載入數據
        # 記憶體映射：處理大檔案
        
    def __getitem__(self, idx):
        # 動態載入和預處理
        # 數據增強：旋轉、縮放、噪聲
```

**為什麼這樣設計**：
- **延遲載入**：3GB數據無法全部載入記憶體
- **記憶體映射**：pandas的chunksize處理大檔案
- **動態增強**：訓練時即時生成變化，增加泛化能力

### 第二步：基礎模型快速驗證（第3天）

#### 2.1 簡單GRU模型
```python
# 文件：src/models/simple_gru.py
# 目的：快速驗證數據管道，建立基線

class SimpleGRU(nn.Module):
    def __init__(self, input_size=162, hidden_size=64, num_classes=34):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch, sequence, features)
        gru_out, _ = self.gru(x)
        # 使用最後一個時間步的輸出
        last_output = gru_out[:, -1, :]
        output = self.fc(self.dropout(last_output))
        return output
```

**為什麼從簡單開始**：
- **快速驗證**：1-2小時訓練，確認數據管道正確
- **基線建立**：目標70%準確率，為後續比較提供基準
- **問題發現**：及早發現數據或代碼問題

#### 2.2 訓練腳本
```python
# 文件：src/train.py
# 關鍵設計決策

def train_model():
    # 批次大小16：基於記憶體分析
    batch_size = 16
    
    # 加權損失：處理類別不平衡
    class_weights = compute_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 學習率調度
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # 早停機制
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
```

**邏輯解釋**：
- **批次大小16**：記憶體使用1.6GB + 模型參數 < 6GB GPU限制
- **加權損失**：某些手語類別樣本很少，需要平衡
- **學習率調度**：大數據集容易陷入局部最優，需要動態調整

### 第三步：中級模型優化（第4-5天）

#### 3.1 加入注意力機制
```python
# 文件：src/models/attention_gru.py
# 目的：提升模型對關鍵幀的關注

class AttentionGRU(nn.Module):
    def __init__(self, input_size=162, hidden_size=128, num_layers=2, num_classes=34):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        # 注意力機制：找出重要的時間步
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        # 全局平均池化
        pooled = torch.mean(attn_out, dim=1)
        return self.fc(pooled)
```

**為什麼加入注意力**：
- **手語特性**：不是所有幀都重要，關鍵動作幀決定意義
- **長序列處理**：30幀序列中，注意力幫助模型聚焦關鍵部分
- **性能提升**：預期從70%提升到85%準確率

#### 3.2 超參數優化
```python
# 文件：src/hyperparameter_tuning.py
# 系統化調優策略

hyperparameter_space = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'hidden_size': [64, 128, 256],
    'num_layers': [1, 2, 3],
    'dropout_rate': [0.2, 0.3, 0.4, 0.5],
    'batch_size': [8, 16, 32]  # 基於記憶體限制
}
```

**調優邏輯**：
- **學習率**：大數據集需要較小學習率避免震盪
- **隱藏層大小**：平衡表達能力和過擬合風險
- **層數**：深度增加表達能力，但增加過擬合風險

### 第四步：高級模型實驗（第6-7天）

#### 4.1 雙向GRU + 自注意力
```python
# 文件：src/models/advanced_gru.py
# 目的：達到90%以上準確率

class AdvancedGRU(nn.Module):
    def __init__(self, input_size=162, hidden_size=256, num_layers=3, num_classes=34):
        super().__init__()
        # 雙向GRU：同時看過去和未來
        self.bigru = nn.GRU(input_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=True, dropout=0.3)
        
        # 自注意力：學習序列內部關係
        self.self_attention = SelfAttention(hidden_size * 2)
        
        # 分類頭
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
```

**為什麼用雙向GRU**：
- **手語特性**：當前幀的意義依賴於前後文
- **序列完整性**：雙向處理捕獲完整的動作模式
- **性能提升**：預期達到90%以上準確率

### 第五步：模型評估與部署（第8-10天）

#### 5.1 綜合評估框架
```python
# 文件：src/evaluation.py
# 多維度性能評估

class ModelEvaluator:
    def evaluate_model(self, model, test_loader):
        metrics = {
            'accuracy': self.compute_accuracy(),
            'per_class_f1': self.compute_f1_scores(),
            'confusion_matrix': self.compute_confusion_matrix(),
            'inference_time': self.measure_inference_speed(),
            'memory_usage': self.measure_memory_usage()
        }
        return metrics
```

#### 5.2 實時系統集成
```python
# 文件：src/real_time_inference.py
# 攝像頭實時推理

class RealTimeSignDetector:
    def __init__(self, model_path):
        self.model = self.load_optimized_model(model_path)
        self.pose_detector = MediaPipeHands()
        self.sequence_buffer = deque(maxlen=30)
        
    def process_frame(self, frame):
        # 1. 提取手部座標
        # 2. 標準化處理
        # 3. 序列預測
        # 4. 平滑輸出
```

## 📊 預期訓練時間表

### 基於數據分析的時間估算

#### 簡單GRU模型（第3天）
- **數據準備**: 2小時
- **模型訓練**: 3.5小時（50 epochs, batch_size=16）
- **評估分析**: 1小時
- **總計**: 6.5小時

#### 中級注意力模型（第4-5天）
- **模型實現**: 3小時
- **超參數調優**: 8小時（多組實驗）
- **最佳模型訓練**: 5小時
- **總計**: 16小時

#### 高級雙向模型（第6-7天）
- **架構實現**: 4小時
- **完整訓練**: 12小時（150 epochs）
- **模型優化**: 4小時
- **總計**: 20小時

## 🎯 關鍵成功因素

### 1. 數據處理質量
- **缺失值策略**：直接影響模型性能
- **序列切割**：影響時序學習效果
- **數據增強**：影響泛化能力

### 2. 模型選擇邏輯
- **漸進式複雜度**：避免過早過擬合
- **針對性設計**：考慮手語動作特性
- **實時性平衡**：準確率與速度平衡

### 3. 訓練策略優化
- **批次大小**：平衡記憶體與梯度穩定性
- **學習率調度**：適應大數據集訓練
- **早停機制**：防止過擬合

## ⚠️ 潛在問題與解決方案

### 問題1：記憶體不足
**解決方案**：
- 數據生成器分批載入
- 梯度累積模擬大批次
- 混合精度訓練節省記憶體

### 問題2：訓練時間過長
**解決方案**：
- 多GPU並行訓練
- 模型並行實驗
- 早期停止無效實驗

### 問題3：類別不平衡
**解決方案**：
- 加權損失函數
- 過採樣少數類別
- 數據增強平衡策略

這個訓練計劃基於實際數據分析，邏輯清晰，步驟漸進，既考慮了技術實現的可行性，也兼顧了項目時間限制和硬件約束。
