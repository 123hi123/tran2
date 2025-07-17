"""
隨機動作測試器 - 基於真實測試腳本邏輯
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
    def __init__(self, data_folder="processed_data", model_folder="models"):
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
        """載入測試資料 - 與測試腳本完全相同"""
        test_path = os.path.join(self.data_folder, "test_dataset.csv")
        
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"找不到測試資料集: {test_path}")
        
        test_data = pd.read_csv(test_path)
        print(f"📁 載入測試資料: {test_data.shape}")
        
        if len(test_data) == 0:
            raise ValueError("測試資料集為空")
        
        # 處理缺失值（與測試腳本保持一致）
        self._preprocess_test_data(test_data)
        
        return test_data
    
    def _preprocess_test_data(self, data):
        """預處理測試數據，與測試腳本完全相同"""
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
                print("⚠️  使用基礎缺失值處理...")
                # 基礎處理：填充 0
                data.fillna(0, inplace=True)
        else:
            print("✅ 測試數據沒有缺失值")
    
    def prepare_all_test_sequences(self, data, sequence_length=20):
        """準備所有測試序列 - 與測試腳本邏輯完全相同"""
        # 特徵欄位（排除標籤相關欄位和frame，與訓練時保持一致）
        feature_cols = [col for col in data.columns 
                       if col not in ['sign_language', 'sign_language_encoded', 'frame', 'source_video']]
        
        print(f"測試特徵維度: {len(feature_cols)} (排除: sign_language, sign_language_encoded, frame, source_video)")
        
        # 按類別分組創建序列
        sequences = []
        labels = []
        class_names = []
        sequence_info = []  # 添加序列信息追蹤
        
        # 按sign_language分組
        for sign_language in data['sign_language'].unique():
            class_data = data[data['sign_language'] == sign_language]
            
            # 如果資料長度超過sequence_length，創建滑動窗口序列
            if len(class_data) >= sequence_length:
                for i in range(len(class_data) - sequence_length + 1):
                    seq = class_data.iloc[i:i+sequence_length][feature_cols].values
                    sequences.append(seq)
                    labels.append(class_data.iloc[i]['sign_language_encoded'])
                    class_names.append(sign_language)
                    # 添加序列來源信息
                    start_frame = i + 1
                    end_frame = i + sequence_length
                    sequence_info.append(f"{sign_language}_seq_{start_frame}-{end_frame}")
            else:
                # 如果資料不足，進行填充
                seq_data = class_data[feature_cols].values
                if len(seq_data) < sequence_length:
                    # 重複最後一幀來填充
                    padding_needed = sequence_length - len(seq_data)
                    last_frame = seq_data[-1:] if len(seq_data) > 0 else np.zeros((1, len(feature_cols)))
                    padding = np.repeat(last_frame, padding_needed, axis=0)
                    seq_data = np.vstack([seq_data, padding])
                
                sequences.append(seq_data)
                labels.append(class_data.iloc[0]['sign_language_encoded'])
                class_names.append(sign_language)
                sequence_info.append(f"{sign_language}_padded")
        
        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        print(f"測試序列形狀: {sequences.shape}")
        print(f"測試標籤形狀: {labels.shape}")
        
        return sequences, labels, class_names, sequence_info
    
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
    
    def run_random_test(self, num_tests=5, sequence_length=20):
        """執行隨機測試"""
        print("=" * 70)
        print("🎯 隨機動作預測測試")
        print("=" * 70)
        
        # 載入模型和數據
        self.load_model()
        test_data = self.load_test_data()
        
        # 準備所有測試序列（使用與測試腳本相同的邏輯）
        X_test, y_test, class_names, sequence_info = self.prepare_all_test_sequences(test_data, sequence_length)
        
        if len(X_test) == 0:
            print("❌ 沒有可用的測試序列")
            return
        
        correct_predictions = 0
        
        for test_num in range(num_tests):
            print(f"\n🔄 測試 {test_num + 1}/{num_tests}")
            print("-" * 50)
            
            # 隨機選擇一個序列
            random_idx = random.randint(0, len(X_test) - 1)
            sequence = X_test[random_idx]
            true_label = y_test[random_idx]
            true_class = self.label_encoder.classes_[true_label]
            seq_info = sequence_info[random_idx]
            
            print(f"📍 序列編號: {random_idx}")
            print(f"📍 序列來源: {seq_info}")
            
            # 預測
            predicted_label, confidence, all_probs = self.predict_single_sequence(sequence)
            predicted_class = self.label_encoder.classes_[predicted_label]
            
            # 結果分析
            is_correct = predicted_label == true_label
            if is_correct:
                correct_predictions += 1
            
            # 顯示結果
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
            
            if accuracy >= 0.8:
                print("🌟 這批隨機樣本表現優秀!")
            elif accuracy >= 0.6:
                print("👍 這批隨機樣本表現良好!")
            else:
                print("📈 這批隨機樣本還有改進空間")
        
        print("=" * 70)

def main():
    """主函數"""
    print("🎯 歡迎使用隨機動作測試器!")
    
    try:
        tester = RandomActionTester()
        
        # 可以調整這些參數
        print("請輸入測試參數（直接按Enter使用預設值）:")
        num_tests_input = input("測試次數 (預設: 5): ").strip()
        num_tests = int(num_tests_input) if num_tests_input else 5
        
        sequence_length_input = input("序列長度 (預設: 20): ").strip()
        sequence_length = int(sequence_length_input) if sequence_length_input else 20
        
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
