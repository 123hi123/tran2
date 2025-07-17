#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手語數據缺失值分析工具
分析左右手座標的缺失情況並提供改進的處理方案
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_missing_values():
    """分析手語數據中的缺失值情況"""
    print("🔍 手語數據缺失值分析")
    print("=" * 60)
    
    # 載入第一個CSV文件進行分析
    csv_files = glob.glob("dataset/sign*.csv")
    if not csv_files:
        print("❌ 找不到數據文件")
        return
    
    print(f"分析文件: {csv_files[0]}")
    df = pd.read_csv(csv_files[0])
    
    print(f"數據形狀: {df.shape}")
    print(f"手語類別: {df['sign_language'].unique()}")
    
    # 分析各類特徵的缺失情況
    print("\n📊 缺失值統計:")
    print("-" * 50)
    
    # 姿態特徵
    pose_cols = [col for col in df.columns if col.startswith('pose_tag')]
    pose_missing = df[pose_cols].isnull().sum().sum()
    pose_total = len(pose_cols) * len(df)
    
    # 左手特徵  
    left_hand_cols = [col for col in df.columns if col.startswith('Left_hand_tag')]
    left_missing = df[left_hand_cols].isnull().sum().sum()
    left_total = len(left_hand_cols) * len(df)
    
    # 右手特徵
    right_hand_cols = [col for col in df.columns if col.startswith('Right_hand_tag')]
    right_missing = df[right_hand_cols].isnull().sum().sum()
    right_total = len(right_hand_cols) * len(df)
    
    print(f"姿態特徵缺失: {pose_missing:,}/{pose_total:,} ({pose_missing/pose_total*100:.1f}%)")
    print(f"左手特徵缺失: {left_missing:,}/{left_total:,} ({left_missing/left_total*100:.1f}%)")
    print(f"右手特徵缺失: {right_missing:,}/{right_total:,} ({right_missing/right_total*100:.1f}%)")
    
    # 分析每一幀的缺失情況
    print("\n🎯 每幀缺失分析:")
    print("-" * 50)
    
    frame_analysis = []
    for idx, row in df.head(1000).iterrows():  # 分析前1000幀
        pose_na = row[pose_cols].isnull().sum()
        left_na = row[left_hand_cols].isnull().sum()
        right_na = row[right_hand_cols].isnull().sum()
        
        frame_analysis.append({
            'frame': row.get('frame', idx),
            'pose_missing': pose_na,
            'left_missing': left_na, 
            'right_missing': right_na,
            'left_completely_missing': left_na == len(left_hand_cols),
            'right_completely_missing': right_na == len(right_hand_cols)
        })
    
    analysis_df = pd.DataFrame(frame_analysis)
    
    # 統計完全缺失的幀數
    left_complete_missing = analysis_df['left_completely_missing'].sum()
    right_complete_missing = analysis_df['right_completely_missing'].sum()
    both_hands_missing = ((analysis_df['left_completely_missing']) & 
                         (analysis_df['right_completely_missing'])).sum()
    
    print(f"左手完全缺失的幀: {left_complete_missing}/{len(analysis_df)} ({left_complete_missing/len(analysis_df)*100:.1f}%)")
    print(f"右手完全缺失的幀: {right_complete_missing}/{len(analysis_df)} ({right_complete_missing/len(analysis_df)*100:.1f}%)")
    print(f"雙手都缺失的幀: {both_hands_missing}/{len(analysis_df)} ({both_hands_missing/len(analysis_df)*100:.1f}%)")
    
    return analysis_df, df

def improved_missing_value_handler():
    """改進的缺失值處理方案"""
    print("\n🛠️  改進的缺失值處理策略")
    print("=" * 60)
    
    print("""
    當前問題:
    - 簡單用 0 填充可能不合理
    - 左右手消失時應該有更智能的處理
    
    改進策略:
    
    1️⃣ 分類處理:
       • 姿態特徵: 使用前後幀插值
       • 手部特徵: 根據缺失模式處理
    
    2️⃣ 手部缺失處理:
       • 部分缺失: 同一手的其他關鍵點平均值
       • 完全缺失: 前後幀的該手座標插值
       • 連續缺失: 使用中性手勢位置
    
    3️⃣ 時序考慮:
       • 利用手語動作的連續性
       • 考慮前後幀的變化趨勢
    """)

class ImprovedMissingValueProcessor:
    """改進的缺失值處理器"""
    
    def __init__(self):
        self.neutral_left_hand = None
        self.neutral_right_hand = None
        
    def calculate_neutral_positions(self, df):
        """計算中性手勢位置（所有有效手部座標的中位數）"""
        left_hand_cols = [col for col in df.columns if col.startswith('Left_hand_tag')]
        right_hand_cols = [col for col in df.columns if col.startswith('Right_hand_tag')]
        
        # 計算中性位置（去除缺失值後的中位數）
        self.neutral_left_hand = df[left_hand_cols].median()
        self.neutral_right_hand = df[right_hand_cols].median()
        
        print("✅ 計算了中性手勢位置")
        
    def interpolate_missing_values(self, df, method='smart'):
        """智能插值缺失值"""
        df_processed = df.copy()
        
        if method == 'smart':
            df_processed = self._smart_interpolation(df_processed)
        else:
            # 簡單填充 0（原始方法）
            df_processed = df_processed.fillna(0)
            
        return df_processed
    
    def _smart_interpolation(self, df):
        """智能插值方法"""
        print("執行智能插值...")
        
        # 1. 姿態特徵：線性插值
        pose_cols = [col for col in df.columns if col.startswith('pose_tag')]
        for col in pose_cols:
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
        
        # 2. 手部特徵：分組處理
        df = self._process_hand_features(df, 'Left_hand_tag')
        df = self._process_hand_features(df, 'Right_hand_tag')
        
        # 3. 剩餘缺失值用中性位置填充
        left_hand_cols = [col for col in df.columns if col.startswith('Left_hand_tag')]
        right_hand_cols = [col for col in df.columns if col.startswith('Right_hand_tag')]
        
        for col in left_hand_cols:
            df[col] = df[col].fillna(self.neutral_left_hand[col])
            
        for col in right_hand_cols:
            df[col] = df[col].fillna(self.neutral_right_hand[col])
            
        return df
    
    def _process_hand_features(self, df, hand_prefix):
        """處理手部特徵"""
        hand_cols = [col for col in df.columns if col.startswith(hand_prefix)]
        
        # 按幀處理
        for idx in df.index:
            hand_data = df.loc[idx, hand_cols]
            missing_count = hand_data.isnull().sum()
            
            if missing_count > 0:
                if missing_count < len(hand_cols) * 0.5:  # 部分缺失
                    # 用該手其他有效座標的平均值填充
                    valid_mean = hand_data.dropna().mean()
                    df.loc[idx, hand_data.isnull()] = valid_mean
                else:  # 大部分或完全缺失
                    # 使用前後幀插值
                    for col in hand_cols:
                        if pd.isnull(df.loc[idx, col]):
                            df.loc[idx, col] = self._get_interpolated_value(df, idx, col)
        
        return df
    
    def _get_interpolated_value(self, df, idx, col):
        """獲取插值"""
        # 簡單前後幀平均
        prev_val = None
        next_val = None
        
        # 向前找
        for i in range(max(0, idx-5), idx):
            if not pd.isnull(df.loc[i, col]):
                prev_val = df.loc[i, col]
                break
                
        # 向後找  
        for i in range(idx+1, min(len(df), idx+6)):
            if not pd.isnull(df.loc[i, col]):
                next_val = df.loc[i, col]
                break
        
        if prev_val is not None and next_val is not None:
            return (prev_val + next_val) / 2
        elif prev_val is not None:
            return prev_val
        elif next_val is not None:
            return next_val
        else:
            return 0  # 最後手段

def create_improved_preprocessing():
    """創建改進的預處理代碼"""
    print("\n💡 創建改進的預處理代碼...")
    
    improved_code = """
def improved_preprocess_features(self, data):
    \"\"\"改進的特徵預處理，智能處理缺失值\"\"\"
    from improved_missing_handler import ImprovedMissingValueProcessor
    
    processed_data = data.copy()
    
    # 使用改進的缺失值處理器
    processor = ImprovedMissingValueProcessor()
    processor.calculate_neutral_positions(processed_data)
    processed_data = processor.interpolate_missing_values(processed_data, method='smart')
    
    print(f"✅ 智能缺失值處理完成")
    print(f"剩餘缺失值: {processed_data.isnull().sum().sum()}")
    
    return processed_data
"""
    
    print("建議的改進代碼:")
    print(improved_code)

def main():
    """主函數"""
    try:
        # 分析缺失值情況
        analysis_df, original_df = analyze_missing_values()
        
        # 提供改進方案
        improved_missing_value_handler()
        
        # 創建改進代碼
        create_improved_preprocessing()
        
        print(f"\n🎯 總結:")
        print("=" * 60)
        print("✅ 分析了數據中的缺失值模式")
        print("✅ 識別了左右手消失的情況") 
        print("✅ 提供了智能插值解決方案")
        print("💡 建議實施改進的缺失值處理策略")
        
    except Exception as e:
        print(f"❌ 分析失敗: {e}")
        print("請確認 dataset/ 目錄下有 CSV 文件")

if __name__ == "__main__":
    main()
