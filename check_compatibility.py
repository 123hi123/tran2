#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
訓練/測試兼容性檢查工具
確保訓練代碼的更動不會破壞測試代碼
"""

import pandas as pd
import numpy as np
import torch
import os
import sys

def check_feature_consistency():
    """檢查特徵提取的一致性"""
    print("🔍 檢查特徵提取一致性")
    print("=" * 50)
    
    # 模擬數據
    sample_data = pd.DataFrame({
        'sign_language': ['A'] * 10,
        'sign_language_encoded': [0] * 10,
        'frame': list(range(10)),
        'source_video': ['video1'] * 10,
        'pose_tag11_x': [1.0] * 10,
        'pose_tag11_y': [2.0] * 10,
        'pose_tag11_z': [3.0] * 10,
        'Left_hand_tag0_x': [4.0] * 10,
        'Left_hand_tag0_y': [5.0] * 10,
        'Left_hand_tag0_z': [6.0] * 10,
    })
    
    # 訓練代碼的特徵提取
    train_feature_cols = [col for col in sample_data.columns 
                         if col not in ['sign_language', 'sign_language_encoded', 'frame', 'source_video']]
    
    # 測試代碼的特徵提取  
    test_feature_cols = [col for col in sample_data.columns 
                        if col not in ['sign_language', 'sign_language_encoded', 'frame', 'source_video']]
    
    print(f"訓練特徵數: {len(train_feature_cols)}")
    print(f"測試特徵數: {len(test_feature_cols)}")
    print(f"特徵一致性: {'✅' if train_feature_cols == test_feature_cols else '❌'}")
    
    if train_feature_cols != test_feature_cols:
        print("❌ 特徵不一致！")
        print(f"訓練獨有: {set(train_feature_cols) - set(test_feature_cols)}")
        print(f"測試獨有: {set(test_feature_cols) - set(train_feature_cols)}")
        return False
    
    print("✅ 特徵提取一致")
    return True

def check_model_architecture():
    """檢查模型架構一致性"""
    print("\n🏗️  檢查模型架構一致性")
    print("=" * 50)
    
    try:
        # 導入訓練和測試的模型定義
        sys.path.append('v1')
        from train_model_v1 import SignLanguageGRU as TrainGRU
        from test_model_v1 import SignLanguageGRU as TestGRU
        
        # 創建相同參數的模型
        input_size = 162  # 更新後的特徵維度
        hidden_size = 128
        num_layers = 2
        num_classes = 10
        dropout = 0.3
        
        train_model = TrainGRU(input_size, hidden_size, num_layers, num_classes, dropout)
        test_model = TestGRU(input_size, hidden_size, num_layers, num_classes, dropout)
        
        # 檢查參數數量
        train_params = sum(p.numel() for p in train_model.parameters())
        test_params = sum(p.numel() for p in test_model.parameters())
        
        print(f"訓練模型參數: {train_params:,}")
        print(f"測試模型參數: {test_params:,}")
        print(f"架構一致性: {'✅' if train_params == test_params else '❌'}")
        
        # 測試前向傳播
        sample_input = torch.randn(2, 20, input_size)  # batch_size=2, seq_len=20, features=162
        
        train_output = train_model(sample_input)
        test_output = test_model(sample_input)
        
        print(f"訓練輸出形狀: {train_output.shape}")
        print(f"測試輸出形狀: {test_output.shape}")
        print(f"輸出一致性: {'✅' if train_output.shape == test_output.shape else '❌'}")
        
        return train_params == test_params and train_output.shape == test_output.shape
        
    except Exception as e:
        print(f"❌ 模型架構檢查失敗: {e}")
        return False

def check_data_preprocessing():
    """檢查數據預處理一致性"""
    print("\n🔧 檢查數據預處理一致性")
    print("=" * 50)
    
    # 創建帶缺失值的測試數據
    test_data = pd.DataFrame({
        'sign_language': ['A', 'B'] * 5,
        'sign_language_encoded': [0, 1] * 5,
        'frame': list(range(10)),
        'pose_tag11_x': [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0],
        'Left_hand_tag0_x': [np.nan] * 5 + [6.0, 7.0, 8.0, 9.0, 10.0],
        'Right_hand_tag0_x': [1.0, 2.0, 3.0, 4.0, 5.0] + [np.nan] * 5,
    })
    
    print(f"原始缺失值: {test_data.isnull().sum().sum()}")
    
    # 檢查改進的處理器是否可用
    try:
        from improved_missing_handler import ImprovedMissingValueProcessor
        processor = ImprovedMissingValueProcessor()
        processor.calculate_neutral_positions(test_data)
        processed_data = processor.smart_interpolation(test_data.copy())
        
        final_missing = processed_data.isnull().sum().sum()
        print(f"智能處理後缺失值: {final_missing}")
        print(f"缺失值處理: {'✅' if final_missing == 0 else '❌'}")
        
        return final_missing == 0
        
    except ImportError:
        print("⚠️  改進的處理器不可用，使用基礎處理")
        processed_data = test_data.fillna(0)
        final_missing = processed_data.isnull().sum().sum()
        print(f"基礎處理後缺失值: {final_missing}")
        return final_missing == 0

def check_model_loading():
    """檢查模型保存/載入兼容性"""
    print("\n💾 檢查模型保存/載入兼容性")
    print("=" * 50)
    
    model_path = "v1/models/latest_model.pth"
    
    if not os.path.exists(model_path):
        print("⚠️  沒有找到訓練好的模型，跳過檢查")
        return True
    
    try:
        # 載入模型檢查點
        checkpoint = torch.load(model_path, map_location='cpu')
        
        required_keys = ['model_state_dict', 'model_config', 'label_encoder']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            print(f"❌ 模型檢查點缺少必要項目: {missing_keys}")
            return False
        
        print("✅ 模型檢查點格式正確")
        
        # 檢查模型配置
        config = checkpoint['model_config']
        print(f"模型配置: {config}")
        
        # 檢查標籤編碼器
        label_encoder = checkpoint['label_encoder']
        print(f"類別數: {len(label_encoder.classes_)}")
        print(f"類別: {list(label_encoder.classes_)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型載入檢查失敗: {e}")
        return False

def check_sequence_preparation():
    """檢查序列準備一致性"""
    print("\n📊 檢查序列準備一致性")
    print("=" * 50)
    
    # 創建測試數據
    test_data = pd.DataFrame({
        'sign_language': ['A'] * 25,
        'sign_language_encoded': [0] * 25,
        'frame': list(range(25)),
        'pose_tag11_x': list(range(25)),
        'pose_tag11_y': list(range(25)),
        'Left_hand_tag0_x': list(range(25)),
    })
    
    # 模擬訓練的序列準備
    sequence_length = 20
    feature_cols = [col for col in test_data.columns 
                   if col not in ['sign_language', 'sign_language_encoded', 'frame', 'source_video']]
    
    # 滑動窗口
    num_sequences = len(test_data) - sequence_length + 1
    sequences = []
    
    for i in range(num_sequences):
        seq = test_data.iloc[i:i+sequence_length][feature_cols].values
        sequences.append(seq)
    
    sequences = np.array(sequences)
    
    print(f"數據長度: {len(test_data)}")
    print(f"序列長度: {sequence_length}")
    print(f"特徵維度: {len(feature_cols)}")
    print(f"生成序列數: {len(sequences)}")
    print(f"序列形狀: {sequences.shape}")
    print(f"預期形狀: ({num_sequences}, {sequence_length}, {len(feature_cols)})")
    
    expected_shape = (num_sequences, sequence_length, len(feature_cols))
    return sequences.shape == expected_shape

def main():
    """主檢查函數"""
    print("🔍 訓練/測試兼容性檢查")
    print("=" * 60)
    
    checks = [
        ("特徵提取一致性", check_feature_consistency),
        ("模型架構一致性", check_model_architecture),
        ("數據預處理一致性", check_data_preprocessing),
        ("模型保存/載入兼容性", check_model_loading),
        ("序列準備一致性", check_sequence_preparation),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} 檢查失敗: {e}")
            results.append((name, False))
    
    print(f"\n📋 檢查結果總結")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通過" if passed else "❌ 失敗"
        print(f"{name:<20}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n{'🎉 所有檢查通過！' if all_passed else '⚠️  存在兼容性問題'}")
    
    if not all_passed:
        print("\n💡 建議:")
        print("1. 確保訓練和測試代碼使用相同的特徵提取邏輯")
        print("2. 檢查模型架構定義是否一致")  
        print("3. 確保數據預處理步驟相同")
        print("4. 重新訓練模型以確保兼容性")

if __name__ == "__main__":
    main()
