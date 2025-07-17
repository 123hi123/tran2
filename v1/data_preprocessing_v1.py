"""
手語辨識資料預處理腳本 v1
功能：
1. 讀取所有以sign開頭的CSV文件
2. 按sign_language類別分組處理
3. 分割為訓練集和測試集
4. 對不足5筆資料的類別進行資料增強
5. 儲存處理後的資料集
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class SignLanguageDataProcessor:
    def __init__(self, dataset_folder="dataset", output_folder="v1/processed_data"):
        self.dataset_folder = dataset_folder
        self.output_folder = output_folder
        self.feature_columns = None
        self.label_encoder = LabelEncoder()
        
    def setup_directories(self):
        """建立輸出資料夾"""
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"輸出資料夾已建立: {self.output_folder}")
    
    def define_feature_columns(self):
        """定義特徵欄位（排除source_video）"""
        # 姿態特徵點 (11-22)
        pose_features = []
        for i in range(11, 23):
            pose_features.extend([f'pose_tag{i}_x', f'pose_tag{i}_y', f'pose_tag{i}_z'])
        
        # 左手特徵點 (0-20)
        left_hand_features = []
        for i in range(21):
            left_hand_features.extend([f'Left_hand_tag{i}_x', f'Left_hand_tag{i}_y', f'Left_hand_tag{i}_z'])
        
        # 右手特徵點 (0-20)
        right_hand_features = []
        for i in range(21):
            right_hand_features.extend([f'Right_hand_tag{i}_x', f'Right_hand_tag{i}_y', f'Right_hand_tag{i}_z'])
        
        # 組合所有欄位（frame用於排序，pose/hand座標用於特徵）
        self.feature_columns = ['frame'] + pose_features + left_hand_features + right_hand_features
        print(f"總欄位數量: {len(self.feature_columns)} (包含frame)")
        print(f"實際特徵維度: {len(pose_features + left_hand_features + right_hand_features)} (不包含frame)")
    
    def load_all_csv_files(self):
        """載入所有以sign開頭的CSV文件"""
        csv_pattern = os.path.join(self.dataset_folder, "sign*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            raise ValueError(f"在 {self.dataset_folder} 中找不到以sign開頭的CSV文件")
        
        print(f"找到 {len(csv_files)} 個CSV文件")
        
        all_data = []
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                print(f"載入文件: {os.path.basename(file_path)} - 形狀: {df.shape}")
                all_data.append(df)
            except Exception as e:
                print(f"載入文件 {file_path} 時發生錯誤: {e}")
        
        if not all_data:
            raise ValueError("沒有成功載入任何CSV文件")
        
        # 合併所有資料
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"合併後總資料形狀: {combined_data.shape}")
        
        # 🔧 重要修正：按 sign_language 和 frame 排序，確保時間序列正確
        if 'frame' in combined_data.columns:
            print("按 sign_language 和 frame 排序資料...")
            combined_data = combined_data.sort_values(['sign_language', 'frame']).reset_index(drop=True)
            print("✅ 時間序列排序完成")
        else:
            print("⚠️  警告：沒有 frame 欄位，無法保證時間序列順序")
        
        return combined_data
    
    def data_augmentation(self, data, noise_factor=0.01):
        """資料增強：添加輕微的隨機雜訊"""
        augmented_data = data.copy()
        
        # 對特徵欄位添加輕微雜訊（不包含frame和非數值欄位）
        numeric_features = [col for col in self.feature_columns if col != 'frame']
        
        for col in numeric_features:
            if col in augmented_data.columns:
                # 添加高斯雜訊
                noise = np.random.normal(0, noise_factor, size=len(augmented_data))
                augmented_data[col] = augmented_data[col] + noise
        
        return augmented_data
    
    def ensure_minimum_samples(self, grouped_data, min_samples=5):
        """確保每個類別至少有min_samples筆資料"""
        processed_groups = {}
        
        for sign_language, group_data in grouped_data.items():
            current_count = len(group_data)
            print(f"類別 '{sign_language}': {current_count} 筆資料", end="")
            
            if current_count < min_samples:
                # 需要增強資料
                needed_samples = min_samples - current_count
                print(f" -> 需要增強 {needed_samples} 筆")
                
                # 重複原始資料並添加雜訊
                augmented_samples = []
                for i in range(needed_samples):
                    # 隨機選擇一筆原始資料進行增強
                    sample_idx = np.random.randint(0, current_count)
                    original_sample = group_data.iloc[[sample_idx]]
                    augmented_sample = self.data_augmentation(original_sample)
                    augmented_samples.append(augmented_sample)
                
                # 合併原始資料和增強資料
                enhanced_data = pd.concat([group_data] + augmented_samples, ignore_index=True)
                processed_groups[sign_language] = enhanced_data
            else:
                print(" -> 資料充足")
                processed_groups[sign_language] = group_data
        
        return processed_groups
    
    def split_train_test(self, processed_groups, test_size=0.2, random_state=42):
        """分割訓練集和測試集"""
        train_data = []
        test_data = []
        
        for sign_language, group_data in processed_groups.items():
            if len(group_data) < 2:
                # 資料太少，全部放入訓練集
                train_data.append(group_data)
                print(f"類別 '{sign_language}': 資料太少，全部作為訓練集")
            else:
                # 分割資料
                train_group, test_group = train_test_split(
                    group_data, 
                    test_size=test_size, 
                    random_state=random_state,
                    stratify=None  # 由於是按類別分組，不需要分層
                )
                train_data.append(train_group)
                test_data.append(test_group)
                print(f"類別 '{sign_language}': 訓練集 {len(train_group)} 筆, 測試集 {len(test_group)} 筆")
        
        # 合併所有類別的資料
        final_train = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
        final_test = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
        
        return final_train, final_test
    
    def preprocess_features(self, data):
        """改進的特徵預處理，智能處理缺失值"""
        print("\n🔧 開始特徵預處理...")
        processed_data = data.copy()
        
        # 檢查是否有缺失值
        total_missing = processed_data.isnull().sum().sum()
        if total_missing > 0:
            print(f"發現 {total_missing} 個缺失值，啟動智能處理...")
            
            # 使用改進的缺失值處理器
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 添加父目錄到路徑
                from improved_missing_handler import ImprovedMissingValueProcessor
                
                processor = ImprovedMissingValueProcessor()
                
                # 分析缺失模式
                analysis = processor.analyze_missing_patterns(processed_data)
                
                # 計算中性位置
                processor.calculate_neutral_positions(processed_data)
                
                # 智能插值
                processed_data = processor.smart_interpolation(processed_data)
                
                print("✅ 智能缺失值處理完成")
                
            except ImportError:
                print("⚠️  無法載入改進的處理器，使用基礎方法...")
                # 回退到基礎方法
                for col in self.feature_columns:
                    if col in processed_data.columns:
                        processed_data[col] = processed_data[col].fillna(0)
        else:
            print("✅ 沒有發現缺失值")
        
        # 最終檢查
        final_missing = processed_data.isnull().sum().sum()
        print(f"最終缺失值: {final_missing}")
        
        return processed_data
    
    def save_datasets(self, train_data, test_data):
        """儲存處理後的資料集"""
        train_path = os.path.join(self.output_folder, "train_dataset.csv")
        test_path = os.path.join(self.output_folder, "test_dataset.csv")
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        print(f"\n資料集已儲存:")
        print(f"訓練集: {train_path} - 形狀: {train_data.shape}")
        print(f"測試集: {test_path} - 形狀: {test_data.shape}")
        
        # 儲存標籤編碼器
        import joblib
        encoder_path = os.path.join(self.output_folder, "label_encoder.pkl")
        joblib.dump(self.label_encoder, encoder_path)
        print(f"標籤編碼器已儲存: {encoder_path}")
    
    def run_preprocessing(self):
        """執行完整的資料預處理流程"""
        print("=" * 60)
        print("手語辨識資料預處理 v1")
        print("=" * 60)
        
        # 1. 建立輸出資料夾
        self.setup_directories()
        
        # 2. 定義特徵欄位
        self.define_feature_columns()
        
        # 3. 載入所有CSV文件
        print("\n步驟 1: 載入資料...")
        combined_data = self.load_all_csv_files()
        
        # 4. 檢查必要欄位
        if 'sign_language' not in combined_data.columns:
            raise ValueError("找不到 'sign_language' 欄位")
        
        # 5. 移除source_video欄位（如果存在）
        if 'source_video' in combined_data.columns:
            combined_data = combined_data.drop('source_video', axis=1)
            print("已移除 'source_video' 欄位")
        
        # 6. 按sign_language分組
        print("\n步驟 2: 按類別分組...")
        grouped_data = dict(list(combined_data.groupby('sign_language')))
        print(f"找到 {len(grouped_data)} 個不同的手語類別")
        
        # 7. 確保每個類別至少有5筆資料
        print("\n步驟 3: 資料增強...")
        processed_groups = self.ensure_minimum_samples(grouped_data)
        
        # 8. 分割訓練集和測試集
        print("\n步驟 4: 分割資料集...")
        train_data, test_data = self.split_train_test(processed_groups)
        
        # 9. 編碼標籤
        print("\n步驟 5: 編碼標籤...")
        all_labels = pd.concat([train_data['sign_language'], test_data['sign_language']])
        self.label_encoder.fit(all_labels)
        
        train_data['sign_language_encoded'] = self.label_encoder.transform(train_data['sign_language'])
        if len(test_data) > 0:
            test_data['sign_language_encoded'] = self.label_encoder.transform(test_data['sign_language'])
        
        # 10. 預處理特徵
        print("\n步驟 6: 預處理特徵...")
        train_data = self.preprocess_features(train_data)
        if len(test_data) > 0:
            test_data = self.preprocess_features(test_data)
        
        # 11. 儲存資料集
        print("\n步驟 7: 儲存資料集...")
        self.save_datasets(train_data, test_data)
        
        print("\n" + "=" * 60)
        print("資料預處理完成!")
        print(f"類別數量: {len(self.label_encoder.classes_)}")
        print(f"類別列表: {list(self.label_encoder.classes_)}")
        print("=" * 60)

def main():
    """主函數"""
    try:
        processor = SignLanguageDataProcessor()
        processor.run_preprocessing()
    except Exception as e:
        print(f"處理過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
