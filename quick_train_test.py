#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速訓練測試 - 跳過複雜預處理
直接測試訓練是否能正常進行
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from datetime import datetime

# 檢查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用設備: {device}")

# 簡單的 GRU 模型
class SimpleSignLanguageGRU(nn.Module):
    def __init__(self, input_size=163, hidden_size=64, num_classes=20, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])  # 取最後一個時間步
        out = self.fc(out)
        return out

def create_fake_data(num_samples=1000, seq_length=20, input_size=163, num_classes=20):
    """創建假數據用於測試"""
    print(f"創建測試數據: {num_samples} 樣本, 序列長度 {seq_length}")
    
    # 生成隨機序列數據
    X = torch.randn(num_samples, seq_length, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    
    return X, y

def train_quick_test():
    """快速訓練測試"""
    print("\n🎯 開始快速訓練測試")
    print("=" * 50)
    
    # 創建測試數據
    X, y = create_fake_data(num_samples=500, seq_length=20)
    
    # 分割訓練測試集
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"訓練集: {X_train.shape}, 測試集: {X_test.shape}")
    
    # 移到GPU
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # 創建模型
    model = SimpleSignLanguageGRU(
        input_size=163,
        hidden_size=64,
        num_classes=20,
        num_layers=1
    ).to(device)
    
    print(f"模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 創建數據載入器
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 訓練循環
    model.train()
    print("\n開始訓練...")
    
    for epoch in range(5):  # 只訓練5個epoch測試
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            # 前向傳播
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # 每10個batch顯示進度
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/5, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/5 完成, 平均損失: {avg_loss:.4f}")
    
    # 測試
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        
        print(f"\n✅ 測試完成")
        print(f"測試損失: {test_loss.item():.4f}")
        print(f"測試準確率: {accuracy:.4f}")
    
    print(f"\n🎉 快速訓練測試成功完成！")
    return True

def test_real_data_loading():
    """測試載入真實數據的第一個文件"""
    print("\n📁 測試載入真實數據")
    print("=" * 50)
    
    try:
        # 找第一個CSV文件
        csv_files = [f for f in os.listdir('dataset') if f.startswith('sign_language') and f.endswith('.csv')]
        if not csv_files:
            print("❌ 沒有找到CSV文件")
            return False
        
        first_file = os.path.join('dataset', csv_files[0])
        print(f"載入文件: {first_file}")
        
        # 載入第一個文件
        df = pd.read_csv(first_file)
        print(f"文件形狀: {df.shape}")
        print(f"類別: {df['sign_language'].unique()[:5]}...")  # 顯示前5個類別
        
        # 檢查特徵欄位
        feature_cols = [col for col in df.columns if col not in ['sign_language', 'source_video', 'frame']]
        print(f"特徵欄位數: {len(feature_cols)}")
        
        # 檢查缺失值
        missing_counts = df[feature_cols].isnull().sum().sum()
        print(f"總缺失值: {missing_counts}")
        
        print("✅ 真實數據載入測試成功")
        return True
        
    except Exception as e:
        print(f"❌ 真實數據載入失敗: {e}")
        return False

def main():
    print("🚀 快速訓練診斷工具")
    print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 測試GPU
    if torch.cuda.is_available():
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ GPU不可用，將使用CPU")
    
    # 測試真實數據載入
    if not test_real_data_loading():
        print("跳過真實數據測試，使用假數據")
    
    # 執行快速訓練測試
    try:
        train_quick_test()
        print("\n🎉 所有測試完成！訓練環境正常工作")
    except Exception as e:
        print(f"\n❌ 訓練測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
