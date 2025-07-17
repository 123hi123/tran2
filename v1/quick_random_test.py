"""
快速隨機測試 - 基於真實測試腳本邏輯
直接運行，隨機選擇5個動作測試模型預測能力
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

# 設定隨機種子，讓結果可重現（如果想要完全隨機，可以註解掉這行）
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class SignLanguageGRU(nn.Module):
    """手語辨識GRU模型"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=10, dropout=0.3):
        super(SignLanguageGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def preprocess_test_data(data):
    """預處理測試數據，與測試腳本保持一致"""
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

def prepare_all_test_sequences(data, sequence_length=20):
    """準備所有測試序列 - 與測試腳本邏輯完全相同"""
    # 特徵欄位（排除標籤相關欄位和frame，與訓練時保持一致）
    feature_cols = [col for col in data.columns 
                   if col not in ['sign_language', 'sign_language_encoded', 'frame', 'source_video']]
    
    print(f"測試特徵維度: {len(feature_cols)} (排除: sign_language, sign_language_encoded, frame, source_video)")
    
    # 按類別分組創建序列
    sequences = []
    labels = []
    class_names = []
    sequence_info = []
    
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

def load_model_and_data():
    """載入模型和數據"""
    data_folder = "processed_data"
    model_folder = "models"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🔧 使用設備: {device}")
    
    # 找最新模型
    model_files = [f for f in os.listdir(model_folder) if f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError("找不到任何模型檔案")
    
    model_path = os.path.join(model_folder, sorted(model_files)[-1])
    print(f"📁 載入模型: {model_path}")
    
    # 載入模型
    checkpoint = torch.load(model_path, map_location=device)
    
    # 載入標籤編碼器
    encoder_path = os.path.join(data_folder, "label_encoder.pkl")
    label_encoder = joblib.load(encoder_path)
    
    # 建立模型
    model_config = checkpoint['model_config']
    model = SignLanguageGRU(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_classes=model_config['num_classes']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 載入測試數據
    test_path = os.path.join(data_folder, "test_dataset.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到測試資料集: {test_path}")
    
    test_data = pd.read_csv(test_path)
    
    # 預處理測試數據
    preprocess_test_data(test_data)
    
    print(f"✅ 模型載入成功，類別: {list(label_encoder.classes_)}")
    print(f"📊 測試數據: {test_data.shape}")
    
    return model, label_encoder, test_data, device

def predict_sequence(model, sequence, device):
    """預測序列"""
    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(sequence_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

def main():
    """主函數"""
    print("🎯 快速隨機動作預測測試")
    print("=" * 60)
    
    try:
        # 載入模型和數據
        model, label_encoder, test_data, device = load_model_and_data()
        
        # 準備所有測試序列（使用與測試腳本相同的邏輯）
        X_test, y_test, class_names, sequence_info = prepare_all_test_sequences(test_data)
        
        if len(X_test) == 0:
            print("❌ 沒有可用的測試序列")
            return
        
        # 進行5次隨機測試
        num_tests = 5
        correct_predictions = 0
        
        for i in range(num_tests):
            print(f"\n🎲 第 {i+1} 次隨機測試")
            print("-" * 40)
            
            # 隨機選擇一個序列
            random_idx = random.randint(0, len(X_test) - 1)
            sequence = X_test[random_idx]
            true_label = y_test[random_idx]
            true_class = label_encoder.classes_[true_label]
            seq_info = sequence_info[random_idx]
            
            print(f"📍 序列編號: {random_idx}")
            print(f"📍 序列來源: {seq_info}")
            
            # 預測
            predicted_label, confidence, all_probs = predict_sequence(model, sequence, device)
            predicted_class = label_encoder.classes_[predicted_label]
            
            # 結果
            is_correct = predicted_label == true_label
            if is_correct:
                correct_predictions += 1
            
            # 顯示結果
            print(f"🎯 實際動作: {true_class}")
            print(f"🤖 預測動作: {predicted_class}")
            print(f"📊 預測信心: {confidence:.4f} ({confidence*100:.1f}%)")
            
            if is_correct:
                print("✅ 預測正確! 🎉")
            else:
                print("❌ 預測錯誤...")
                
                # 顯示前3名預測
                prob_pairs = list(zip(label_encoder.classes_, all_probs))
                prob_pairs.sort(key=lambda x: x[1], reverse=True)
                print("前3名預測:")
                for j, (class_name, prob) in enumerate(prob_pairs[:3]):
                    print(f"  {j+1}. {class_name}: {prob:.3f} ({prob*100:.1f}%)")
        
        # 總結
        accuracy = correct_predictions / num_tests
        print(f"\n🏆 測試總結")
        print("=" * 40)
        print(f"總測試次數: {num_tests}")
        print(f"正確預測: {correct_predictions}")
        print(f"預測準確率: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        if accuracy >= 0.8:
            print("🌟 這批隨機樣本表現優秀!")
        elif accuracy >= 0.6:
            print("👍 這批隨機樣本表現良好!")
        else:
            print("📈 這批隨機樣本還有改進空間")
            
    except Exception as e:
        print(f"❌ 測試過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
