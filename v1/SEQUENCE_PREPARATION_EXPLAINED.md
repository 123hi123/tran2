# 序列資料準備詳解 - prepare_sequences 函數

## 🎯 函數目的
`prepare_sequences` 函數的主要目的是將CSV格式的手語資料轉換成適合GRU模型訓練的時序序列格式。

## 📊 資料轉換過程

### 原始資料格式
```
CSV資料（每行是一幀）:
frame  pose_tag11_x  pose_tag11_y  ...  sign_language  sign_language_encoded
1      0.234         0.567         ...  hello          0
2      0.235         0.568         ...  hello          0
3      0.236         0.569         ...  hello          0
...
50     0.240         0.570         ...  hello          0
```

### 目標序列格式
```
3D張量 [樣本數, 序列長度, 特徵維度]:
sequences[0] = [
    [frame1的所有特徵],  # 第1幀
    [frame2的所有特徵],  # 第2幀
    ...
    [frame20的所有特徵]  # 第20幀
]
labels[0] = 0  # 對應的類別標籤
```

## 🔄 詳細處理步驟

### 步驟1: 提取特徵欄位
```python
feature_cols = [col for col in data.columns 
               if col not in ['sign_language', 'sign_language_encoded']]
```
**作用**: 移除標籤欄位，只保留實際的特徵數據（座標點等）

### 步驟2: 按類別分組處理
```python
for sign_language in data['sign_language'].unique():
    class_data = data[data['sign_language'] == sign_language]
```
**作用**: 將相同手語類別的所有幀數據集中處理

### 步驟3: 創建時序序列
有兩種情況：

#### 情況A: 資料充足（≥ sequence_length幀）
```python
if len(class_data) >= sequence_length:
    for i in range(len(class_data) - sequence_length + 1):
        seq = class_data.iloc[i:i+sequence_length][feature_cols].values
        sequences.append(seq)
```

**滑動窗口示例** (sequence_length=5):
```
原始50幀 → 創建46個序列:
序列1: [幀1, 幀2, 幀3, 幀4, 幀5]
序列2: [幀2, 幀3, 幀4, 幀5, 幀6]
序列3: [幀3, 幀4, 幀5, 幀6, 幀7]
...
序列46: [幀46, 幀47, 幀48, 幀49, 幀50]
```

#### 情況B: 資料不足（< sequence_length幀）
```python
else:
    # 重複最後一幀來填充
    padding_needed = sequence_length - len(seq_data)
    last_frame = seq_data[-1:]
    padding = np.repeat(last_frame, padding_needed, axis=0)
    seq_data = np.vstack([seq_data, padding])
```

**填充示例** (只有3幀，需要5幀):
```
原始: [幀1, 幀2, 幀3]
填充後: [幀1, 幀2, 幀3, 幀3, 幀3]  # 重複最後一幀
```

## 🧮 輸出結果

### 最終張量形狀
- **sequences**: `(樣本數, sequence_length, 特徵維度)`
- **labels**: `(樣本數,)`

### 實際例子
假設有：
- 3個手語類別 (hello, thanks, goodbye)
- 每個類別50幀資料
- sequence_length=20
- 特徵維度=163

結果：
```python
sequences.shape = (138, 20, 163)
# 138 = 每類46個序列 × 3類 = 138個訓練樣本
# 20 = 序列長度（時間步）
# 163 = 特徵維度

labels.shape = (138,)
# 138個對應的標籤
```

## ⚡ 為什麼需要這樣處理？

### 1. **時序建模需求**
- GRU/LSTM需要固定長度的序列輸入
- 手語是時序動作，需要分析連續幀之間的關係

### 2. **資料增強效果**
- 滑動窗口增加了訓練樣本數量
- 從50幀生成46個訓練樣本，增強模型泛化能力

### 3. **處理變長序列**
- 實際手語動作時間長短不一
- 填充機制確保所有序列長度一致

## 🔍 關鍵參數說明

### sequence_length (預設20)
```python
# 較短序列 (10-15): 計算快，但可能丟失長期依賴
# 中等序列 (20-30): 平衡效能和效果
# 較長序列 (40+): 捕捉更多時序信息，但計算昂貴
```

### 特徵維度計算
```python
# 姿態點: 12點 × 3軸 = 36維 (pose_tag11-22)
# 左手點: 21點 × 3軸 = 63維 (Left_hand_tag0-20)  
# 右手點: 21點 × 3軸 = 63維 (Right_hand_tag0-20)
# Frame: 1維
# 總計: 36 + 63 + 63 + 1 = 163維
```

## 💡 優化建議

### 根據硬體調整sequence_length
```python
# RTX A2000 (6GB): sequence_length=20, batch_size=16
# 低記憶體設備: sequence_length=15, batch_size=8
# 高效能設備: sequence_length=30, batch_size=32
```

### 資料品質考慮
- 確保幀之間時序連續性
- 檢查是否有缺失幀或跳幀
- 考慮正規化特徵數據

這個步驟是整個深度學習流程中的關鍵環節，它將原始的表格數據轉換成神經網路可以理解的時序序列格式！
