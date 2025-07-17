"""
隨機動作測試器
隨機選擇一個手語動作，測試模型是否能正確預測
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SignLanguageGRU(nn.Module):
    """手語辨識GRU模型（與訓練時相同的架構）"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=10, dropout=0.3):
        super(SignLanguageGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU層
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 全連接層
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # *2 因為雙向GRU
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        gru_out, _ = self.gru(x)
        
        # 取最後一個時間步的輸出
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # 通過全連接層
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class RandomActionTester:
    def __init__(self, data_folder="v1/processed_data", model_folder="v1/models"):
        self.data_folder = data_folder
        self.model_folder = model_folder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_encoder = None
        
        print(f"🎯 隨機動作測試器啟動")
        print(f"使用設備: {self.device}")
    
    def load_model(self, model_path=None):
        """載入訓練好的模型"""
        if model_path is None:
            # 找最新的模型
            model_files = [f for f in os.listdir(self.model_folder) if f.endswith('.pth')]
            if not model_files:
                raise FileNotFoundError("找不到任何模型檔案")
            model_path = os.path.join(self.model_folder, sorted(model_files)[-1])
        
        print(f"載入模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 載入標籤編碼器
        encoder_path = os.path.join(self.data_folder, "label_encoder.pkl")
        self.label_encoder = joblib.load(encoder_path)
        
        # 建立模型
        model_config = checkpoint['model_config']
        self.model = SignLanguageGRU(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_classes=model_config['num_classes']
        ).to(self.device)
        
        # 載入模型權重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ 模型載入成功")
        print(f"📊 類別數量: {len(self.label_encoder.classes_)}")
        print(f"🏷️ 類別: {list(self.label_encoder.classes_)}")
        
        return checkpoint
    
    def load_test_data(self):
        """載入測試資料"""
        test_path = os.path.join(self.data_folder, "test_dataset.csv")
        
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"找不到測試資料集: {test_path}")
        
        test_data = pd.read_csv(test_path)
        print(f"📁 載入測試資料: {test_data.shape}")
        
        # 處理缺失值（與訓練時保持一致）
        self._preprocess_test_data(test_data)
        
        return test_data
    
    def _preprocess_test_data(self, data):
        """預處理測試數據，確保與訓練時一致"""
        total_missing = data.isnull().sum().sum()
        if total_missing > 0:
            print(f"⚠️  測試數據發現 {total_missing} 個缺失值，進行處理...")
            
            # 嘗試使用改進的處理器
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                from improved_missing_handler import ImprovedMissingValueProcessor
                
                processor = ImprovedMissingValueProcessor()
                processor.calculate_neutral_positions(data)
                data_processed = processor.smart_interpolation(data)
                
                # 將處理結果更新回原數據
                data.update(data_processed)
                print("✅ 智能缺失值處理完成")
                
            except ImportError:
                print("⚠️  使用基本缺失值處理...")
                # 基本的線性插值
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    if data[col].isnull().any():
                        data[col] = data[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    def get_random_action_sequence(self, test_data, sequence_length=20):
        """隨機選擇一個動作的一個序列"""
        # 獲取所有可用的類別
        available_classes = test_data['sign_language'].unique()
        
        # 隨機選擇一個類別
        random_class = random.choice(available_classes)
        print(f"🎲 隨機選擇的類別: {random_class}")
        
        # 獲取該類別的所有數據
        class_data = test_data[test_data['sign_language'] == random_class].copy()
        
        # 按source_video分組
        videos = class_data['source_video'].unique()
        random_video = random.choice(videos)
        print(f"📹 隨機選擇的視頻: {random_video}")
        
        # 獲取該視頻的數據
        video_data = class_data[class_data['source_video'] == random_video].copy()
        video_data = video_data.sort_values('frame')
        
        # 特徵欄位（162維，排除frame）
        pose_columns = [f'pose_{i}' for i in range(36)]
        left_hand_columns = [f'left_hand_{i}' for i in range(63)]
        right_hand_columns = [f'right_hand_{i}' for i in range(63)]
        feature_columns = pose_columns + left_hand_columns + right_hand_columns
        
        # 檢查可用特徵
        available_features = [col for col in feature_columns if col in video_data.columns]
        
        if len(video_data) < sequence_length:
            print(f"⚠️  視頻太短 ({len(video_data)} 幀)，需要至少 {sequence_length} 幀")
            return None, None, None, None
        
        # 隨機選擇序列起始位置
        max_start = len(video_data) - sequence_length
        random_start = random.randint(0, max_start)
        random_end = random_start + sequence_length
        
        print(f"🎬 選擇序列: 第 {random_start+1} - {random_end} 幀")
        
        # 提取序列
        sequence_data = video_data.iloc[random_start:random_end]
        features = sequence_data[available_features].values
        
        # 檢查序列完整性
        if np.isnan(features).all():
            print("⚠️  序列全為缺失值，重新選擇...")
            return self.get_random_action_sequence(test_data, sequence_length)
        
        # 標籤編碼
        true_label = self.label_encoder.transform([random_class])[0]
        
        return features, true_label, random_class, f"{random_video}[{random_start+1}:{random_end}]"
    
    def predict_single_sequence(self, sequence):
        """對單個序列進行預測"""
        # 轉換為tensor並添加batch維度
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # (1, seq_len, features)
        
        # 預測
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()
    
    def run_random_test(self, num_tests=1, sequence_length=20):
        """執行隨機測試"""
        print("=" * 70)
        print("🎯 隨機動作預測測試")
        print("=" * 70)
        
        # 載入模型和數據
        self.load_model()
        test_data = self.load_test_data()
        
        correct_predictions = 0
        
        for test_num in range(num_tests):
            print(f"\n🔄 測試 {test_num + 1}/{num_tests}")
            print("-" * 50)
            
            # 獲取隨機序列
            sequence, true_label, true_class, sequence_info = self.get_random_action_sequence(
                test_data, sequence_length
            )
            
            if sequence is None:
                print("跳過此次測試...")
                continue
            
            # 預測
            predicted_label, confidence, all_probs = self.predict_single_sequence(sequence)
            predicted_class = self.label_encoder.classes_[predicted_label]
            
            # 結果分析
            is_correct = predicted_label == true_label
            if is_correct:
                correct_predictions += 1
            
            # 顯示結果
            print(f"📍 序列來源: {sequence_info}")
            print(f"🎯 真實標籤: {true_class}")
            print(f"🤖 預測標籤: {predicted_class}")
            print(f"📊 預測信心: {confidence:.4f} ({confidence*100:.2f}%)")
            print(f"✅ 預測結果: {'正確' if is_correct else '錯誤'}")
            
            # 顯示所有類別的機率（前5名）
            prob_pairs = list(zip(self.label_encoder.classes_, all_probs))
            prob_pairs.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n📈 預測機率排名 (前5名):")
            for i, (class_name, prob) in enumerate(prob_pairs[:5]):
                marker = "👑" if i == 0 else f"{i+1}."
                print(f"  {marker} {class_name}: {prob:.4f} ({prob*100:.2f}%)")
            
            print("-" * 50)
        
        # 總結
        if num_tests > 0:
            accuracy = correct_predictions / num_tests
            print(f"\n🏆 測試總結:")
            print(f"總測試次數: {num_tests}")
            print(f"正確預測: {correct_predictions}")
            print(f"準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("=" * 70)

def main():
    """主函數"""
    print("🎯 歡迎使用隨機動作測試器!")
    
    try:
        tester = RandomActionTester()
        
        # 可以調整這些參數
        num_tests = int(input("請輸入測試次數 (預設: 5): ") or "5")
        sequence_length = int(input("請輸入序列長度 (預設: 20): ") or "20")
        
        print(f"\n開始進行 {num_tests} 次隨機測試...")
        tester.run_random_test(num_tests=num_tests, sequence_length=sequence_length)
        
    except KeyboardInterrupt:
        print("\n用戶中斷測試")
    except Exception as e:
        print(f"測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
