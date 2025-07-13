# 手語辨識GRU模型訓練專案

## 📊 專案概述

基於完整數據分析結果，本專案實現了一個智能的手語辨識系統，使用GRU神經網絡處理時序手語動作數據。

### 🎯 數據規模
- **總樣本數**: 1,210,017 幀（約121萬個時間點）
- **手語類別**: 34 種不同手語動作
- **特徵維度**: 162 維座標特徵（身體36 + 左手63 + 右手63）
- **數據大小**: 3.05GB
- **關鍵挑戰**: 右手66%缺失，左手10%缺失，類別不平衡

## 🚀 快速開始

### 1. 環境準備
```bash
# 創建虛擬環境 - 推薦使用最新穩定版本
conda create -n sign_language python=3.11
# 或者使用最新版本 (更佳性能)
# conda create -n sign_language python=3.12

conda activate sign_language

# 安裝深度學習依賴
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib seaborn opencv-python

# 手部檢測套件
pip install mediapipe
```

**為什麼推薦Python 3.11/3.12**：
- **性能大幅提升**: 3.11比3.9快10-60%，對於大數據處理特別明顯
- **更好的記憶體管理**: 降低記憶體使用，適合處理3GB大型數據集
- **改進的錯誤提示**: 更容易調試模型訓練問題
- **更強的typing支援**: 更好的代碼可維護性

### 2. 快速驗證
```bash
# 運行快速開始腳本
python quick_start.py
```

### 3. 開始訓練
```bash
# 完整訓練流程
python src/train.py
```

## 📁 專案結構

```
tran2/
├── dataset/                    # 數據檔案目錄
│   ├── sign_language1.csv
│   ├── sign_language2.csv
│   └── ... (共19個檔案)
├── src/                        # 源代碼
│   ├── data_preprocessing.py   # 數據預處理模組
│   ├── models/
│   │   └── gru_models.py      # GRU模型定義
│   └── train.py               # 訓練腳本
├── test/                       # 測試和分析
│   └── sign_language_dataset_analysis.csv
├── models/                     # 保存的模型
├── logs/                       # 訓練日誌
├── quick_start.py             # 快速開始腳本
├── verify_preprocessing.py    # 預處理驗證
├── training_strategy.md       # 詳細訓練策略
├── implementation_steps.md    # 實施步驟
└── development_plan.md        # 開發計劃
```

## 🎯 訓練策略邏輯

### 為什麼選擇這種方案？

#### 1. 數據預處理策略
- **缺失值處理**: 66%右手缺失→時序插值+姿態約束
- **序列長度**: 30幀(1秒)→平衡信息完整性與計算效率
- **滑動窗口**: 15幀步長→從121萬幀生成8萬序列

#### 2. 模型架構設計
- **漸進式複雜度**: 簡單→中等→高級，避免過早過擬合
- **GRU vs LSTM**: 參數更少，適合實時推理
- **注意力機制**: 聚焦手語動作關鍵幀

#### 3. 訓練策略優化
- **批次大小16**: 記憶體1.6GB + 模型參數 < 6GB GPU
- **加權損失**: 解決34類別不平衡問題
- **學習率調度**: 大數據集需要精細調優

## 🤖 模型架構對比

### 🥉 SimpleGRU - 基礎驗證模型
```
輸入(30, 162) → GRU(64) → Dropout(0.3) → FC(34)
```
- **參數量**: ~50K
- **目標準確率**: >70%
- **訓練時間**: 1-2小時
- **用途**: 快速驗證數據管道

### 🥈 AttentionGRU - 實用性能模型
```
輸入(30, 162) → GRU(128, layers=2) → Attention → FC(34)
```
- **參數量**: ~200K
- **目標準確率**: >85%
- **訓練時間**: 3-5小時
- **用途**: 平衡性能與效率

### 🥇 BiGRUWithSelfAttention - 最佳性能模型
```
輸入(30, 162) → BiGRU(256, layers=3) → SelfAttention → FC(34)
```
- **參數量**: ~800K
- **目標準確率**: >90%
- **訓練時間**: 8-12小時
- **用途**: 最終部署模型

## 📈 訓練時間估算

基於數據分析的實際估算：

| 模型類型 | 批次大小 | 每epoch時間 | 總訓練時間 |
|---------|---------|------------|------------|
| SimpleGRU | 16 | 4.2分鐘 | 3.5小時(50epochs) |
| AttentionGRU | 16 | 8.4分鐘 | 14小時(100epochs) |
| BiGRU | 16 | 16.8分鐘 | 42小時(150epochs) |

## 🛠️ 關鍵技術實現

### 數據預處理亮點
```python
# 智能缺失值處理
def handle_missing_values(self, df):
    # 1. 時序線性插值
    # 2. 姿態約束插值
    # 3. 身體中心點填充
```

### 模型訓練亮點
```python
# 類別權重平衡
class_weights = total_samples / (num_classes * class_counts)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 梯度裁剪防爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 📊 預期成果與驗證

### 技術指標
- [x] 數據分析完成 - 121萬樣本，34類
- [ ] 整體準確率 > 85%
- [ ] 每類別F1-score > 0.8
- [ ] 推理延遲 < 100ms
- [ ] 模型大小 < 50MB

### 實用性指標
- [ ] 支持實時攝像頭輸入
- [ ] 穩定的長時間運行
- [ ] 易於部署和擴展

## ⏰ 10天訓練時間表

### 第1-2天: 環境與數據準備
- [x] 環境配置完成
- [x] 數據分析完成 ✅
- [ ] 數據預處理管道
- [ ] 基線模型驗證

### 第3-4天: 基礎模型訓練
- [ ] SimpleGRU模型 (目標>70%)
- [ ] 數據管道驗證
- [ ] 初步性能分析

### 第5-6天: 中級模型優化
- [ ] AttentionGRU模型 (目標>85%)
- [ ] 超參數調優
- [ ] 性能對比分析

### 第7-8天: 高級模型實驗
- [ ] BiGRU模型 (目標>90%)
- [ ] 完整數據集訓練
- [ ] 模型壓縮優化

### 第9-10天: 部署與測試
- [ ] 實時系統集成
- [ ] 攝像頭測試
- [ ] 最終性能驗證

## 🔧 使用方法

### 基本訓練
```bash
# 1. 快速環境檢查
python quick_start.py

# 2. 預處理驗證
python verify_preprocessing.py

# 3. 開始訓練
python src/train.py
```

### 自定義訓練
```python
from src.train import TrainingConfig, SignLanguageTrainer

# 自定義配置
config = TrainingConfig()
config.batch_size = 32  # 調整批次大小
config.learning_rate = 0.0005  # 調整學習率

# 初始化訓練器
trainer = SignLanguageTrainer(config)

# 準備數據
csv_files = ["dataset/sign_language1.csv"]
train_loader, val_loader, test_loader, num_classes = trainer.prepare_data(csv_files)

# 創建和訓練模型
model = trainer.create_model('attention', num_classes)
trainer.train_model(model, train_loader, val_loader)
```

## 🚨 常見問題解決

### 記憶體不足
```python
# 減少批次大小
config.batch_size = 8

# 使用混合精度訓練
scaler = torch.cuda.amp.GradScaler()
```

### 訓練時間過長
```python
# 使用較少數據檔案開始
csv_files = csv_files[:3]  # 只用前3個檔案

# 減少epoch數量
config.num_epochs = 50
```

### 類別不平衡
```python
# 自動計算類別權重（已實現）
class_weights = trainer.calculate_class_weights(train_loader)
```

## 📈 監控與調試

### 訓練監控
- 訓練/驗證損失曲線
- 準確率變化趨勢
- 學習率調度狀態
- GPU記憶體使用率

### 結果分析
- 混淆矩陣分析
- 各類別性能詳情
- 注意力權重可視化
- 錯誤案例分析

## 🎯 專案成功的關鍵因素

### 數據質量
1. **智能缺失值處理** - 直接影響模型性能
2. **序列切割策略** - 影響時序學習效果  
3. **數據增強技術** - 提升泛化能力

### 模型設計
1. **漸進式複雜度** - 避免過早過擬合
2. **針對性架構** - 考慮手語動作特性
3. **實時性平衡** - 準確率與速度並重

### 訓練優化
1. **批次大小選擇** - 平衡記憶體與穩定性
2. **學習率調度** - 適應大數據集特點
3. **早停機制** - 防止過擬合

## 📚 參考文獻與資源

- GRU論文: "Learning Phrase Representations using RNN Encoder-Decoder"
- 注意力機制: "Attention Is All You Need"
- 手語識別綜述: "A Survey on Sign Language Recognition"
- MediaPipe手部檢測: Google MediaPipe文檔

## 🤝 貢獻與支持

如果你在使用過程中遇到問題或有改進建議：

1. 檢查 `logs/` 目錄中的錯誤日誌
2. 運行 `verify_preprocessing.py` 進行診斷
3. 查看 `training_strategy.md` 獲取詳細邏輯說明
4. 參考 `implementation_steps.md` 了解實施細節

---

**🎯 記住：這個訓練策略是基於實際數據分析制定的，每個步驟都有明確的邏輯原因。按部就班執行，成功率會更高！**
