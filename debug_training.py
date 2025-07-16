#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
訓練問題診斷工具
檢查訓練卡住的原因
"""

import torch
import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime

def check_environment():
    """檢查環境狀態"""
    print("🔍 環境診斷")
    print("=" * 50)
    
    # PyTorch 版本
    print(f"PyTorch版本: {torch.__version__}")
    
    # CUDA 狀態
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.version.cuda}")
        print(f"GPU設備: {torch.cuda.get_device_name(0)}")
        print(f"GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA不可用 - 將使用CPU (非常慢)")
    
    print()

def check_data():
    """檢查數據狀態"""
    print("📁 數據檢查")
    print("=" * 50)
    
    # 檢查數據資料夾
    if not os.path.exists('dataset'):
        print("❌ dataset 資料夾不存在")
        return False
    
    # 檢查CSV文件
    csv_files = [f for f in os.listdir('dataset') if f.startswith('sign_language') and f.endswith('.csv')]
    print(f"找到 {len(csv_files)} 個CSV文件")
    
    if len(csv_files) == 0:
        print("❌ 沒有找到數據文件")
        return False
    
    # 檢查第一個文件
    try:
        first_file = os.path.join('dataset', csv_files[0])
        df = pd.read_csv(first_file)
        print(f"✅ 樣本文件: {csv_files[0]}")
        print(f"   形狀: {df.shape}")
        print(f"   欄位: {list(df.columns)[:5]}...")
        
        # 檢查必要欄位
        required_columns = ['sign_language']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ 缺少必要欄位: {missing_columns}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ 讀取數據文件失敗: {e}")
        return False

def test_simple_training():
    """測試簡單的訓練步驟"""
    print("🚀 簡單訓練測試")
    print("=" * 50)
    
    try:
        # 創建簡單的假數據
        print("創建測試數據...")
        batch_size = 4  # 減小批次大小
        seq_length = 10
        input_size = 163
        num_classes = 3
        
        # 假數據
        X = torch.randn(batch_size, seq_length, input_size)
        y = torch.randint(0, num_classes, (batch_size,))
        
        # 檢查設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        
        X = X.to(device)
        y = y.to(device)
        
        # 簡單模型
        from torch import nn
        
        class SimpleGRU(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, num_classes)
                
            def forward(self, x):
                _, h = self.gru(x)
                return self.fc(h[-1])
        
        print("創建模型...")
        model = SimpleGRU(input_size, 32, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 測試一個訓練步驟
        print("執行測試訓練步驟...")
        start_time = time.time()
        
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        end_time = time.time()
        
        print(f"✅ 訓練步驟成功")
        print(f"   損失: {loss.item():.4f}")
        print(f"   時間: {end_time - start_time:.2f} 秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 訓練測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """測試數據載入"""
    print("📊 數據載入測試")
    print("=" * 50)
    
    try:
        # 嘗試載入實際數據
        from src.data_preprocessing import DataPreprocessor
        
        print("初始化數據預處理器...")
        preprocessor = DataPreprocessor()
        
        print("載入數據...")
        start_time = time.time()
        X, y = preprocessor.load_and_preprocess()
        end_time = time.time()
        
        print(f"✅ 數據載入成功")
        print(f"   X形狀: {X.shape}")
        print(f"   y形狀: {y.shape}")
        print(f"   載入時間: {end_time - start_time:.2f} 秒")
        print(f"   類別數: {len(np.unique(y))}")
        
        return True
        
    except Exception as e:
        print(f"❌ 數據載入失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_memory():
    """檢查記憶體使用"""
    print("💾 記憶體檢查")
    print("=" * 50)
    
    try:
        import psutil
        
        # 系統記憶體
        memory = psutil.virtual_memory()
        print(f"系統記憶體: {memory.total / 1024**3:.1f} GB")
        print(f"可用記憶體: {memory.available / 1024**3:.1f} GB")
        print(f"使用率: {memory.percent:.1f}%")
        
        # GPU 記憶體
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated(0)
            gpu_cached = torch.cuda.memory_reserved(0)
            
            print(f"GPU總記憶體: {gpu_memory / 1024**3:.1f} GB")
            print(f"GPU已分配: {gpu_allocated / 1024**3:.3f} GB")
            print(f"GPU快取: {gpu_cached / 1024**3:.3f} GB")
        
    except ImportError:
        print("psutil 未安裝，無法檢查系統記憶體")
    except Exception as e:
        print(f"記憶體檢查失敗: {e}")

def main():
    print(f"🔧 訓練問題診斷工具")
    print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    # 執行所有檢查
    check_environment()
    
    if not check_data():
        print("❌ 數據檢查失敗，無法繼續")
        return
    
    check_memory()
    
    if not test_simple_training():
        print("❌ 簡單訓練測試失敗")
        return
    
    if not test_data_loading():
        print("❌ 數據載入測試失敗")
        return
    
    print()
    print("🎉 所有診斷完成")
    print("=" * 60)
    print("💡 建議:")
    print("1. 如果CUDA不可用，訓練會很慢，建議修復GPU問題")
    print("2. 如果記憶體不足，嘗試減小batch_size")
    print("3. 如果數據載入慢，檢查硬碟空間和檔案完整性")

if __name__ == "__main__":
    main()
