#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改進的缺失值處理器
專門處理手語數據中左右手座標的缺失問題
"""

import pandas as pd
import numpy as np
from typing import Dict, List

class ImprovedMissingValueProcessor:
    """改進的缺失值處理器"""
    
    def __init__(self):
        self.neutral_positions = {}
        self.interpolation_stats = {
            'pose_interpolated': 0,
            'left_hand_interpolated': 0,
            'right_hand_interpolated': 0,
            'neutral_filled': 0
        }
        
    def analyze_missing_patterns(self, df: pd.DataFrame) -> Dict:
        """分析缺失值模式"""
        print("🔍 分析缺失值模式...")
        
        # 分類特徵欄位
        pose_cols = [col for col in df.columns if col.startswith('pose_tag')]
        left_hand_cols = [col for col in df.columns if col.startswith('Left_hand_tag')]
        right_hand_cols = [col for col in df.columns if col.startswith('Right_hand_tag')]
        
        analysis = {
            'total_frames': len(df),
            'pose_missing_rate': df[pose_cols].isnull().sum().sum() / (len(pose_cols) * len(df)),
            'left_hand_missing_rate': df[left_hand_cols].isnull().sum().sum() / (len(left_hand_cols) * len(df)),
            'right_hand_missing_rate': df[right_hand_cols].isnull().sum().sum() / (len(right_hand_cols) * len(df))
        }
        
        # 分析完全缺失的幀
        left_complete_missing = 0
        right_complete_missing = 0
        both_missing = 0
        
        for idx, row in df.iterrows():
            left_missing = row[left_hand_cols].isnull().all()
            right_missing = row[right_hand_cols].isnull().all()
            
            if left_missing:
                left_complete_missing += 1
            if right_missing:
                right_complete_missing += 1
            if left_missing and right_missing:
                both_missing += 1
        
        analysis.update({
            'left_complete_missing_frames': left_complete_missing,
            'right_complete_missing_frames': right_complete_missing,
            'both_hands_missing_frames': both_missing
        })
        
        print(f"  姿態缺失率: {analysis['pose_missing_rate']*100:.1f}%")
        print(f"  左手缺失率: {analysis['left_hand_missing_rate']*100:.1f}%")
        print(f"  右手缺失率: {analysis['right_hand_missing_rate']*100:.1f}%")
        print(f"  左手完全缺失幀數: {left_complete_missing}")
        print(f"  右手完全缺失幀數: {right_complete_missing}")
        print(f"  雙手都缺失幀數: {both_missing}")
        
        return analysis
    
    def calculate_neutral_positions(self, df: pd.DataFrame):
        """計算中性手勢位置"""
        print("📍 計算中性手勢位置...")
        
        # 姿態中性位置
        pose_cols = [col for col in df.columns if col.startswith('pose_tag')]
        self.neutral_positions['pose'] = df[pose_cols].median()
        
        # 左手中性位置
        left_hand_cols = [col for col in df.columns if col.startswith('Left_hand_tag')]
        self.neutral_positions['left_hand'] = df[left_hand_cols].median()
        
        # 右手中性位置  
        right_hand_cols = [col for col in df.columns if col.startswith('Right_hand_tag')]
        self.neutral_positions['right_hand'] = df[right_hand_cols].median()
        
        print("✅ 中性位置計算完成")
    
    def smart_interpolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """智能插值處理缺失值"""
        print("🔧 開始智能插值處理...")
        
        df_processed = df.copy()
        self.interpolation_stats = {k: 0 for k in self.interpolation_stats.keys()}
        
        # 步驟1: 處理姿態特徵（線性插值）
        df_processed = self._interpolate_pose_features(df_processed)
        
        # 步驟2: 處理左手特徵
        df_processed = self._interpolate_hand_features(df_processed, 'Left_hand_tag', 'left_hand')
        
        # 步驟3: 處理右手特徵
        df_processed = self._interpolate_hand_features(df_processed, 'Right_hand_tag', 'right_hand')
        
        # 步驟4: 最終填充剩餘缺失值
        df_processed = self._final_fillna(df_processed)
        
        self._print_interpolation_summary()
        
        return df_processed
    
    def _interpolate_pose_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """插值姿態特徵"""
        pose_cols = [col for col in df.columns if col.startswith('pose_tag')]
        
        for col in pose_cols:
            before_na = df[col].isnull().sum()
            
            # 線性插值
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            
            # 如果還有缺失，用中性位置填充
            df[col] = df[col].fillna(self.neutral_positions['pose'][col])
            
            after_na = df[col].isnull().sum() 
            self.interpolation_stats['pose_interpolated'] += (before_na - after_na)
        
        return df
    
    def _interpolate_hand_features(self, df: pd.DataFrame, hand_prefix: str, neutral_key: str) -> pd.DataFrame:
        """插值手部特徵"""
        hand_cols = [col for col in df.columns if col.startswith(hand_prefix)]
        
        for idx in df.index:
            hand_data = df.loc[idx, hand_cols]
            missing_count = hand_data.isnull().sum()
            
            if missing_count == 0:
                continue  # 沒有缺失值
            
            missing_rate = missing_count / len(hand_cols)
            
            if missing_rate < 0.3:  # 輕微缺失 (<30%)
                self._handle_partial_missing(df, idx, hand_cols, hand_data)
            elif missing_rate < 0.8:  # 中等缺失 (30-80%)
                self._handle_moderate_missing(df, idx, hand_cols, neutral_key)
            else:  # 嚴重缺失 (>80%)
                self._handle_severe_missing(df, idx, hand_cols, neutral_key)
            
            # 統計
            if hand_prefix == 'Left_hand_tag':
                self.interpolation_stats['left_hand_interpolated'] += missing_count
            else:
                self.interpolation_stats['right_hand_interpolated'] += missing_count
        
        return df
    
    def _handle_partial_missing(self, df: pd.DataFrame, idx: int, hand_cols: List[str], hand_data: pd.Series):
        """處理輕微缺失 - 用同一手的其他有效值平均"""
        valid_values = hand_data.dropna()
        if len(valid_values) > 0:
            # 分別處理 x, y, z 座標
            for axis in ['_x', '_y', '_z']:
                axis_cols = [col for col in hand_cols if col.endswith(axis)]
                axis_data = hand_data[axis_cols]
                
                if axis_data.isnull().any():
                    axis_mean = axis_data.dropna().mean()
                    df.loc[idx, axis_data.isnull()] = axis_mean
    
    def _handle_moderate_missing(self, df: pd.DataFrame, idx: int, hand_cols: List[str], neutral_key: str):
        """處理中等缺失 - 時序插值 + 中性位置"""
        for col in hand_cols:
            if pd.isnull(df.loc[idx, col]):
                # 嘗試時序插值
                interpolated_value = self._temporal_interpolation(df, idx, col, window=3)
                
                if interpolated_value is not None:
                    df.loc[idx, col] = interpolated_value
                else:
                    # 使用中性位置
                    df.loc[idx, col] = self.neutral_positions[neutral_key][col]
    
    def _handle_severe_missing(self, df: pd.DataFrame, idx: int, hand_cols: List[str], neutral_key: str):
        """處理嚴重缺失 - 直接用中性位置"""
        for col in hand_cols:
            if pd.isnull(df.loc[idx, col]):
                df.loc[idx, col] = self.neutral_positions[neutral_key][col]
    
    def _temporal_interpolation(self, df: pd.DataFrame, idx: int, col: str, window: int = 3) -> float:
        """時序插值"""
        values = []
        
        # 向前查找
        for i in range(max(0, idx - window), idx):
            if not pd.isnull(df.loc[i, col]):
                values.append(df.loc[i, col])
        
        # 向後查找
        for i in range(idx + 1, min(len(df), idx + window + 1)):
            if not pd.isnull(df.loc[i, col]):
                values.append(df.loc[i, col])
        
        return np.mean(values) if values else None
    
    def _final_fillna(self, df: pd.DataFrame) -> pd.DataFrame:
        """最終填充剩餘的缺失值"""
        remaining_na = df.isnull().sum().sum()
        
        if remaining_na > 0:
            print(f"⚠️  剩餘 {remaining_na} 個缺失值，用 0 填充")
            df = df.fillna(0)
            self.interpolation_stats['neutral_filled'] += remaining_na
        
        return df
    
    def _print_interpolation_summary(self):
        """打印插值摘要"""
        print("\n📊 插值處理摘要:")
        print("-" * 40)
        print(f"姿態特徵插值: {self.interpolation_stats['pose_interpolated']}")
        print(f"左手特徵插值: {self.interpolation_stats['left_hand_interpolated']}")
        print(f"右手特徵插值: {self.interpolation_stats['right_hand_interpolated']}")
        print(f"中性位置填充: {self.interpolation_stats['neutral_filled']}")
        print("-" * 40)

def demonstrate_improvement():
    """演示改進效果"""
    print("\n🎯 改進的缺失值處理策略")
    print("=" * 60)
    
    print("""
    🔄 處理流程:
    
    1️⃣ 姿態特徵 (pose_tag11-22):
       • 線性插值 → 中性位置填充
       
    2️⃣ 手部特徵 (Left/Right_hand_tag0-20):
       • 輕微缺失 (<30%): 同手其他座標平均值
       • 中等缺失 (30-80%): 時序插值 + 中性位置
       • 嚴重缺失 (>80%): 直接用中性位置
       
    3️⃣ 最終處理:
       • 剩餘缺失值用 0 填充
    
    ✅ 優點:
    • 保持手語動作的連續性
    • 考慮左右手的獨立性  
    • 避免不合理的座標值
    • 提供詳細的處理統計
    """)

if __name__ == "__main__":
    demonstrate_improvement()
