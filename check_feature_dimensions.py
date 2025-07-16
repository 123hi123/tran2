#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
計算特徵維度工具
檢查我們實際使用的特徵數量
"""

def calculate_feature_dimensions():
    """計算特徵維度"""
    print("🧮 計算手語辨識模型的特徵維度")
    print("=" * 50)
    
    # 1. Frame 欄位
    frame_features = 1
    print(f"Frame 欄位: {frame_features}")
    
    # 2. 姿態特徵點 (11-22)
    pose_count = 12  # pose_tag11 到 pose_tag22
    pose_coords = 3  # x, y, z
    pose_features = pose_count * pose_coords
    print(f"姿態特徵: {pose_count} 個點 × {pose_coords} 座標 = {pose_features}")
    
    # 3. 左手特徵點 (0-20)
    left_hand_count = 21  # Left_hand_tag0 到 Left_hand_tag20
    left_hand_coords = 3  # x, y, z
    left_hand_features = left_hand_count * left_hand_coords
    print(f"左手特徵: {left_hand_count} 個點 × {left_hand_coords} 座標 = {left_hand_features}")
    
    # 4. 右手特徵點 (0-20)
    right_hand_count = 21  # Right_hand_tag0 到 Right_hand_tag20
    right_hand_coords = 3  # x, y, z
    right_hand_features = right_hand_count * right_hand_coords
    print(f"右手特徵: {right_hand_count} 個點 × {right_hand_coords} 座標 = {right_hand_features}")
    
    # 5. 總特徵維度
    total_features = frame_features + pose_features + left_hand_features + right_hand_features
    
    print("\n📊 特徵維度總結:")
    print("=" * 50)
    print(f"Frame:        {frame_features:3d} 維")
    print(f"姿態 (11-22): {pose_features:3d} 維")
    print(f"左手 (0-20):  {left_hand_features:3d} 維")
    print(f"右手 (0-20):  {right_hand_features:3d} 維")
    print("-" * 30)
    print(f"總計:         {total_features:3d} 維")
    
    # 6. 檢查實際使用的特徵（排除 frame）
    actual_features = total_features - frame_features  # 訓練時不使用 frame
    print(f"\n實際輸入GRU: {actual_features:3d} 維 (排除frame)")
    
    return total_features, actual_features

def check_feature_columns_match():
    """檢查程式碼中的設定是否一致"""
    print("\n🔍 檢查程式碼一致性")
    print("=" * 50)
    
    # 模擬 data_preprocessing_v1.py 中的計算
    pose_features = []
    for i in range(11, 23):
        pose_features.extend([f'pose_tag{i}_x', f'pose_tag{i}_y', f'pose_tag{i}_z'])
    
    left_hand_features = []
    for i in range(21):
        left_hand_features.extend([f'Left_hand_tag{i}_x', f'Left_hand_tag{i}_y', f'Left_hand_tag{i}_z'])
    
    right_hand_features = []
    for i in range(21):
        right_hand_features.extend([f'Right_hand_tag{i}_x', f'Right_hand_tag{i}_y', f'Right_hand_tag{i}_z'])
    
    # 加上 frame
    feature_columns = ['frame'] + pose_features + left_hand_features + right_hand_features
    
    print(f"實際程式碼計算的特徵欄位數: {len(feature_columns)}")
    print(f"其中包含:")
    print(f"  - frame: 1")
    print(f"  - pose_features: {len(pose_features)}")
    print(f"  - left_hand_features: {len(left_hand_features)}")
    print(f"  - right_hand_features: {len(right_hand_features)}")
    
    # 訓練時會排除某些欄位
    excluded_columns = ['sign_language', 'sign_language_encoded']
    training_features = len(feature_columns)  # 假設其他欄位都用於訓練
    
    print(f"\n訓練時特徵維度:")
    print(f"  - 全部特徵: {len(feature_columns)}")
    print(f"  - 排除標籤: -{len(excluded_columns)}")
    print(f"  - 實際輸入GRU: {training_features} 維")
    
    return len(feature_columns)

def main():
    # 理論計算
    total, actual = calculate_feature_dimensions()
    
    # 程式碼驗證
    code_features = check_feature_columns_match()
    
    print(f"\n✅ 驗證結果")
    print("=" * 50)
    print(f"理論計算: {total} 維特徵")
    print(f"程式碼實作: {code_features} 維特徵")
    print(f"實際輸入GRU: {actual} 維特徵 (排除frame)")
    
    if total == code_features:
        print("✅ 計算一致!")
    else:
        print("❌ 計算不一致，需要檢查!")

if __name__ == "__main__":
    main()
