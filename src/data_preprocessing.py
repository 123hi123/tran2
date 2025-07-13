"""
手語辨識數據預處理模組
基於數據分析結果：121萬樣本，34類，右手66%缺失，左手10%缺失

主要功能：
1. 缺失值智能插值
2. 座標標準化
3. 序列切割與生成
4. 數據增強
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class SignLanguagePreprocessor:
    def __init__(self, sequence_length: int = 30, stride: int = 15):
        """
        參數設定基於數據分析：
        - sequence_length=30: 約1秒手語動作，平衡信息完整性與計算效率
        - stride=15: 50%重疊，從121萬幀生成約8萬序列
        """
        self.sequence_length = sequence_length
        self.stride = stride
        
        # 特徵欄位定義（基於CSV分析）
        self.pose_columns = [f'pose_tag{i}_{axis}' for i in range(11, 23) for axis in ['x', 'y', 'z']]
        self.left_hand_columns = [f'Left_hand_tag{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
        self.right_hand_columns = [f'Right_hand_tag{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
        
        # 統計信息（用於標準化）
        self.pose_stats = None
        self.hand_stats = None
        
    def load_and_analyze_data(self, csv_files: List[str]) -> Dict:
        """
        載入和分析數據
        返回統計信息用於後續處理
        """
        print("🔍 載入數據並分析統計特性...")
        
        all_stats = {
            'total_samples': 0,
            'sign_language_counts': {},
            'missing_rates': {},
            'coordinate_stats': {}
        }
        
        for file_path in csv_files:
            print(f"📄 處理檔案: {file_path}")
            
            # 分塊讀取大檔案
            chunk_stats = []
            for chunk in pd.read_csv(file_path, chunksize=10000):
                chunk_stats.append(self._analyze_chunk(chunk))
                
            # 合併統計信息
            file_stats = self._merge_chunk_stats(chunk_stats)
            all_stats = self._merge_file_stats(all_stats, file_stats)
            
        return all_stats
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        智能缺失值處理策略
        
        基於分析結果：
        - 右手缺失66%：可能是左手為主的手語
        - 左手缺失10%：偶爾的檢測失敗
        
        策略：
        1. 時序線性插值（前後幀）
        2. 基於身體姿態的約束插值  
        3. 極端情況使用身體中心點
        """
        df_processed = df.copy()
        
        print("🔧 處理缺失值...")
        
        # 1. 時序插值（組內插值，按視頻分組）
        for video in df['source_video'].unique():
            video_mask = df['source_video'] == video
            video_data = df_processed[video_mask].copy()
            
            # 按幀順序排序
            video_data = video_data.sort_values('frame')
            
            # 對座標欄位進行插值
            coordinate_columns = self.pose_columns + self.left_hand_columns + self.right_hand_columns
            for col in coordinate_columns:
                if col in video_data.columns:
                    # 線性插值
                    video_data[col] = video_data[col].interpolate(method='linear', limit_direction='both')
                    
                    # 如果開頭或結尾仍有缺失，使用前向/後向填充
                    video_data[col] = video_data[col].fillna(method='bfill').fillna(method='ffill')
            
            df_processed.loc[video_mask] = video_data
        
        # 2. 基於身體姿態的約束插值
        df_processed = self._pose_constrained_interpolation(df_processed)
        
        # 3. 最後的缺失值處理（使用身體中心點）
        df_processed = self._fill_remaining_missing(df_processed)
        
        return df_processed
    
    def normalize_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        座標標準化處理
        
        目標：
        1. 消除個人差異（身高、體型）
        2. 消除位置偏差（站立位置）
        3. 統一座標尺度
        """
        df_normalized = df.copy()
        
        print("📐 標準化座標...")
        
        # 計算身體中心點（肩膀中點）
        if 'pose_tag11_x' in df.columns and 'pose_tag12_x' in df.columns:
            body_center_x = (df['pose_tag11_x'] + df['pose_tag12_x']) / 2
            body_center_y = (df['pose_tag11_y'] + df['pose_tag12_y']) / 2
            
            # 計算身體大小（肩寬作為參考）
            body_scale = np.sqrt((df['pose_tag11_x'] - df['pose_tag12_x'])**2 + 
                               (df['pose_tag11_y'] - df['pose_tag12_y'])**2)
            body_scale = body_scale.replace(0, 1)  # 避免除零
            
            # 標準化所有座標
            coordinate_columns = self.pose_columns + self.left_hand_columns + self.right_hand_columns
            
            for col in coordinate_columns:
                if col in df.columns:
                    if col.endswith('_x'):
                        # X座標以身體中心為原點，按身體大小縮放
                        df_normalized[col] = (df[col] - body_center_x) / body_scale
                    elif col.endswith('_y'):
                        # Y座標以身體中心為原點，按身體大小縮放
                        df_normalized[col] = (df[col] - body_center_y) / body_scale
                    elif col.endswith('_z'):
                        # Z座標按身體大小縮放
                        df_normalized[col] = df[col] / body_scale
        
        return df_normalized
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        創建訓練序列
        
        策略：
        1. 滑動窗口生成序列
        2. 固定長度（30幀）
        3. 標籤對應
        
        返回：
        - sequences: (N, sequence_length, feature_dim)
        - labels: (N,)
        """
        print("🔄 生成訓練序列...")
        
        sequences = []
        labels = []
        
        # 按視頻分組處理
        for video in df['source_video'].unique():
            video_data = df[df['source_video'] == video].copy()
            video_data = video_data.sort_values('frame')
            
            # 提取特徵和標籤
            feature_columns = self.pose_columns + self.left_hand_columns + self.right_hand_columns
            available_features = [col for col in feature_columns if col in video_data.columns]
            
            features = video_data[available_features].values
            sign_language = video_data['sign_language'].iloc[0]  # 假設同一視頻同一標籤
            
            # 滑動窗口生成序列
            for start_idx in range(0, len(features) - self.sequence_length + 1, self.stride):
                end_idx = start_idx + self.sequence_length
                sequence = features[start_idx:end_idx]
                
                # 檢查序列完整性
                if not np.isnan(sequence).all():  # 不是全NaN
                    sequences.append(sequence)
                    labels.append(sign_language)
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        print(f"✅ 生成 {len(sequences)} 個序列")
        print(f"📊 序列形狀: {sequences.shape}")
        print(f"🏷️ 標籤種類: {len(np.unique(labels))}")
        
        return sequences, labels
    
    def augment_sequences(self, sequences: np.ndarray, labels: np.ndarray, 
                         augment_factor: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        數據增強
        
        策略：
        1. 時間扭曲（時間軸拉伸/壓縮）
        2. 空間變換（旋轉、縮放）
        3. 噪聲添加
        """
        print("🎭 數據增強...")
        
        augmented_sequences = [sequences]
        augmented_labels = [labels]
        
        n_augmentations = int(augment_factor - 1)
        
        for _ in range(n_augmentations):
            # 時間扭曲
            time_warped = self._time_warp(sequences)
            augmented_sequences.append(time_warped)
            augmented_labels.append(labels)
            
            # 空間變換
            spatially_transformed = self._spatial_transform(sequences)
            augmented_sequences.append(spatially_transformed)
            augmented_labels.append(labels)
            
            # 噪聲添加
            noisy = self._add_noise(sequences)
            augmented_sequences.append(noisy)
            augmented_labels.append(labels)
        
        final_sequences = np.concatenate(augmented_sequences, axis=0)
        final_labels = np.concatenate(augmented_labels, axis=0)
        
        print(f"✅ 增強後序列數量: {len(final_sequences)}")
        
        return final_sequences, final_labels
    
    def _analyze_chunk(self, chunk: pd.DataFrame) -> Dict:
        """分析單個數據塊"""
        stats = {
            'samples': len(chunk),
            'sign_languages': chunk['sign_language'].value_counts().to_dict(),
            'missing_rates': chunk.isnull().sum() / len(chunk)
        }
        return stats
    
    def _merge_chunk_stats(self, chunk_stats: List[Dict]) -> Dict:
        """合併塊統計信息"""
        # 實現統計信息合併邏輯
        merged = {
            'samples': sum(s['samples'] for s in chunk_stats),
            'sign_languages': {},
            'missing_rates': {}
        }
        # 詳細實現...
        return merged
    
    def _merge_file_stats(self, all_stats: Dict, file_stats: Dict) -> Dict:
        """合併檔案統計信息"""
        # 實現檔案間統計合併
        return all_stats
    
    def _pose_constrained_interpolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """基於身體姿態約束的插值"""
        # 實現基於身體結構的約束插值
        return df
    
    def _fill_remaining_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """填充剩餘缺失值"""
        # 使用身體中心點或統計值填充
        return df
    
    def _time_warp(self, sequences: np.ndarray) -> np.ndarray:
        """時間扭曲增強"""
        # 實現時間軸變換
        return sequences
    
    def _spatial_transform(self, sequences: np.ndarray) -> np.ndarray:
        """空間變換增強"""
        # 實現旋轉、縮放變換
        return sequences
    
    def _add_noise(self, sequences: np.ndarray) -> np.ndarray:
        """添加噪聲增強"""
        # 實現高斯噪聲添加
        noise_scale = 0.01
        noise = np.random.normal(0, noise_scale, sequences.shape)
        return sequences + noise

def main():
    """主函數 - 演示使用方法"""
    print("🚀 手語辨識數據預處理流程")
    print("=" * 50)
    
    # 初始化預處理器
    preprocessor = SignLanguagePreprocessor(sequence_length=30, stride=15)
    
    # 數據檔案列表
    csv_files = [
        'dataset/sign_language1.csv',
        'dataset/sign_language2.csv',
        # ... 其他檔案
    ]
    
    # 數據分析
    stats = preprocessor.load_and_analyze_data(csv_files)
    print(f"📊 總樣本數: {stats['total_samples']}")
    
    # 處理單個檔案示例
    df = pd.read_csv('dataset/sign_language1.csv')
    print(f"📄 載入檔案，形狀: {df.shape}")
    
    # 缺失值處理
    df_clean = preprocessor.handle_missing_values(df)
    print(f"🔧 缺失值處理完成")
    
    # 座標標準化
    df_normalized = preprocessor.normalize_coordinates(df_clean)
    print(f"📐 座標標準化完成")
    
    # 生成序列
    sequences, labels = preprocessor.create_sequences(df_normalized)
    print(f"🔄 序列生成完成: {sequences.shape}")
    
    # 數據增強
    aug_sequences, aug_labels = preprocessor.augment_sequences(sequences, labels)
    print(f"🎭 數據增強完成: {aug_sequences.shape}")
    
    print("✅ 預處理流程完成!")

if __name__ == "__main__":
    main()
