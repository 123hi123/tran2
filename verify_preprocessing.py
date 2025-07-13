"""
快速驗證腳本：測試數據預處理管道
在開始大規模訓練前，確保所有組件正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import SignLanguagePreprocessor
import time

def quick_data_verification():
    """快速數據驗證"""
    print("🔍 快速數據驗證開始")
    print("=" * 50)
    
    # 載入一個小樣本進行測試
    sample_file = "dataset/sign_language1.csv"
    
    if not os.path.exists(sample_file):
        print(f"❌ 找不到檔案: {sample_file}")
        return False
    
    print(f"📄 載入測試檔案: {sample_file}")
    
    # 讀取前1000行進行快速測試
    df_sample = pd.read_csv(sample_file, nrows=1000)
    print(f"📊 樣本形狀: {df_sample.shape}")
    print(f"🏷️ 手語類別: {df_sample['sign_language'].unique()}")
    
    # 檢查基本結構
    expected_columns = ['sign_language', 'source_video', 'frame']
    for col in expected_columns:
        if col not in df_sample.columns:
            print(f"❌ 缺少必要欄位: {col}")
            return False
    
    print("✅ 基本結構檢查通過")
    
    # 缺失值分析
    missing_rates = df_sample.isnull().sum() / len(df_sample) * 100
    high_missing = missing_rates[missing_rates > 50]
    
    print(f"📈 高缺失率欄位 (>50%): {len(high_missing)}")
    if len(high_missing) > 0:
        print("   主要缺失欄位:", high_missing.head().to_dict())
    
    return True

def test_preprocessing_pipeline():
    """測試預處理管道"""
    print("\n🧪 測試預處理管道")
    print("=" * 50)
    
    try:
        # 初始化預處理器
        preprocessor = SignLanguagePreprocessor(sequence_length=30, stride=15)
        print("✅ 預處理器初始化成功")
        
        # 載入測試數據
        df_sample = pd.read_csv("dataset/sign_language1.csv", nrows=1000)
        print(f"📄 載入測試數據: {df_sample.shape}")
        
        # 測試缺失值處理
        print("🔧 測試缺失值處理...")
        start_time = time.time()
        df_clean = preprocessor.handle_missing_values(df_sample)
        process_time = time.time() - start_time
        print(f"   處理時間: {process_time:.2f}秒")
        print(f"   缺失值減少: {df_sample.isnull().sum().sum()} → {df_clean.isnull().sum().sum()}")
        
        # 測試座標標準化
        print("📐 測試座標標準化...")
        start_time = time.time()
        df_normalized = preprocessor.normalize_coordinates(df_clean)
        process_time = time.time() - start_time
        print(f"   處理時間: {process_time:.2f}秒")
        
        # 測試序列生成
        print("🔄 測試序列生成...")
        start_time = time.time()
        sequences, labels = preprocessor.create_sequences(df_normalized)
        process_time = time.time() - start_time
        print(f"   處理時間: {process_time:.2f}秒")
        print(f"   生成序列數量: {len(sequences)}")
        print(f"   序列形狀: {sequences.shape if len(sequences) > 0 else 'None'}")
        print(f"   標籤種類: {len(np.unique(labels)) if len(labels) > 0 else 0}")
        
        if len(sequences) > 0:
            # 測試數據增強
            print("🎭 測試數據增強...")
            start_time = time.time()
            aug_sequences, aug_labels = preprocessor.augment_sequences(
                sequences[:10], labels[:10], augment_factor=2.0  # 只測試前10個序列
            )
            process_time = time.time() - start_time
            print(f"   處理時間: {process_time:.2f}秒")
            print(f"   增強後數量: {len(aug_sequences)}")
            
            return True
        else:
            print("❌ 未能生成有效序列")
            return False
            
    except Exception as e:
        print(f"❌ 預處理管道測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def analyze_sequence_characteristics():
    """分析序列特徵"""
    print("\n📊 序列特徵分析")
    print("=" * 50)
    
    try:
        # 載入並處理數據
        preprocessor = SignLanguagePreprocessor(sequence_length=30, stride=15)
        df_sample = pd.read_csv("dataset/sign_language1.csv", nrows=5000)  # 更大樣本
        
        print("處理數據中...")
        df_clean = preprocessor.handle_missing_values(df_sample)
        df_normalized = preprocessor.normalize_coordinates(df_clean)
        sequences, labels = preprocessor.create_sequences(df_normalized)
        
        if len(sequences) == 0:
            print("❌ 無法生成序列進行分析")
            return
        
        print(f"📈 分析 {len(sequences)} 個序列")
        
        # 基本統計
        print(f"   序列形狀: {sequences.shape}")
        print(f"   特徵維度: {sequences.shape[2]}")
        print(f"   數值範圍: [{np.nanmin(sequences):.3f}, {np.nanmax(sequences):.3f}]")
        print(f"   平均值: {np.nanmean(sequences):.3f}")
        print(f"   標準差: {np.nanstd(sequences):.3f}")
        
        # 標籤分布
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"   標籤分布:")
        for label, count in zip(unique_labels, counts):
            print(f"     {label}: {count} 個序列")
        
        # 缺失值分析
        missing_ratio = np.isnan(sequences).sum() / sequences.size
        print(f"   缺失值比例: {missing_ratio:.1%}")
        
        return sequences, labels
        
    except Exception as e:
        print(f"❌ 序列分析失敗: {str(e)}")
        return None, None

def performance_benchmark():
    """性能基準測試"""
    print("\n⚡ 性能基準測試")
    print("=" * 50)
    
    # 測試不同數據量的處理時間
    test_sizes = [100, 500, 1000, 5000]
    
    for size in test_sizes:
        try:
            print(f"📊 測試 {size} 樣本...")
            
            start_time = time.time()
            
            # 載入數據
            df = pd.read_csv("dataset/sign_language1.csv", nrows=size)
            load_time = time.time() - start_time
            
            # 預處理
            preprocessor = SignLanguagePreprocessor()
            
            process_start = time.time()
            df_clean = preprocessor.handle_missing_values(df)
            df_normalized = preprocessor.normalize_coordinates(df_clean)
            sequences, labels = preprocessor.create_sequences(df_normalized)
            process_time = time.time() - process_start
            
            total_time = time.time() - start_time
            
            print(f"   載入時間: {load_time:.2f}s")
            print(f"   處理時間: {process_time:.2f}s")  
            print(f"   總時間: {total_time:.2f}s")
            print(f"   生成序列: {len(sequences)}")
            print(f"   處理速度: {size/total_time:.1f} 樣本/秒")
            print()
            
        except Exception as e:
            print(f"   ❌ {size} 樣本測試失敗: {str(e)}")

def estimate_full_processing_time():
    """估算完整數據處理時間"""
    print("\n⏱️ 完整數據處理時間估算")
    print("=" * 50)
    
    # 基於小樣本估算
    sample_size = 1000
    total_samples = 1_210_017  # 基於數據分析結果
    
    try:
        start_time = time.time()
        
        df = pd.read_csv("dataset/sign_language1.csv", nrows=sample_size)
        preprocessor = SignLanguagePreprocessor()
        
        df_clean = preprocessor.handle_missing_values(df)
        df_normalized = preprocessor.normalize_coordinates(df_clean)
        sequences, labels = preprocessor.create_sequences(df_normalized)
        
        sample_time = time.time() - start_time
        
        # 估算總時間
        estimated_total_hours = (sample_time / sample_size) * total_samples / 3600
        
        print(f"📊 樣本大小: {sample_size}")
        print(f"   處理時間: {sample_time:.2f}秒")
        print(f"   處理速度: {sample_size/sample_time:.1f} 樣本/秒")
        print()
        print(f"🔮 完整數據估算:")
        print(f"   總樣本數: {total_samples:,}")
        print(f"   預估總時間: {estimated_total_hours:.1f} 小時")
        print(f"   建議: 分批處理，使用多進程")
        
        if estimated_total_hours > 12:
            print("⚠️  警告: 處理時間過長，建議:")
            print("   1. 增加處理器核心數")
            print("   2. 優化算法效率") 
            print("   3. 分批並行處理")
            
    except Exception as e:
        print(f"❌ 估算失敗: {str(e)}")

def main():
    """主驗證流程"""
    print("🚀 手語數據預處理驗證流程")
    print("🕐 開始時間:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    # 1. 基本數據驗證
    if not quick_data_verification():
        print("❌ 基本驗證失敗，請檢查數據檔案")
        return
    
    # 2. 預處理管道測試
    if not test_preprocessing_pipeline():
        print("❌ 預處理管道測試失敗")
        return
    
    # 3. 序列特徵分析
    sequences, labels = analyze_sequence_characteristics()
    
    # 4. 性能基準測試
    performance_benchmark()
    
    # 5. 完整處理時間估算
    estimate_full_processing_time()
    
    print("\n" + "=" * 60)
    print("✅ 驗證流程完成!")
    print("🕐 結束時間:", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    # 總結建議
    print("\n📋 下一步建議:")
    print("1. 如果驗證通過，開始完整數據預處理")
    print("2. 實現基礎GRU模型")
    print("3. 開始訓練實驗")

if __name__ == "__main__":
    main()
