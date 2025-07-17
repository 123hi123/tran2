"""
快速隨機測試
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

def load_model_and_data():
    """載入模型和數據"""
    data_folder = "v1/processed_data"
    model_folder = "v1/models"
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
    test_data = pd.read_csv(test_path)
    
    print(f"✅ 模型載入成功，類別: {list(label_encoder.classes_)}")
    print(f"📊 測試數據: {test_data.shape}")
    
    return model, label_encoder, test_data, device

def get_random_sequence(test_data, label_encoder, sequence_length=20):
    """獲取隨機序列"""
    # 隨機選擇類別
    available_classes = test_data['sign_language'].unique()
    random_class = random.choice(available_classes)
    
    # 獲取該類別數據
    class_data = test_data[test_data['sign_language'] == random_class].copy()
    
    # 隨機選擇視頻
    videos = class_data['source_video'].unique()
    random_video = random.choice(videos)
    
    # 獲取視頻數據
    video_data = class_data[class_data['source_video'] == random_video].copy()
    video_data = video_data.sort_values('frame')
    
    if len(video_data) < sequence_length:
        return get_random_sequence(test_data, label_encoder, sequence_length)
    
    # 特徵欄位
    pose_columns = [f'pose_{i}' for i in range(36)]
    left_hand_columns = [f'left_hand_{i}' for i in range(63)]
    right_hand_columns = [f'right_hand_{i}' for i in range(63)]
    feature_columns = pose_columns + left_hand_columns + right_hand_columns
    available_features = [col for col in feature_columns if col in video_data.columns]
    
    # 隨機選擇序列位置
    max_start = len(video_data) - sequence_length
    random_start = random.randint(0, max_start)
    random_end = random_start + sequence_length
    
    # 提取序列
    sequence_data = video_data.iloc[random_start:random_end]
    features = sequence_data[available_features].values
    
    # 處理缺失值
    if np.isnan(features).any():
        # 簡單線性插值
        for i in range(features.shape[1]):
            col_data = features[:, i]
            if np.isnan(col_data).any():
                # 前向填充然後後向填充
                mask = ~np.isnan(col_data)
                if mask.any():
                    features[:, i] = np.interp(
                        np.arange(len(col_data)),
                        np.arange(len(col_data))[mask],
                        col_data[mask]
                    )
                else:
                    features[:, i] = 0  # 如果全是NaN，填充為0
    
    true_label = label_encoder.transform([random_class])[0]
    sequence_info = f"{random_video}[{random_start+1}:{random_end}]"
    
    return features, true_label, random_class, sequence_info

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
        
        # 進行5次隨機測試
        num_tests = 5
        correct_predictions = 0
        
        for i in range(num_tests):
            print(f"\n🎲 第 {i+1} 次隨機測試")
            print("-" * 40)
            
            # 獲取隨機序列
            sequence, true_label, true_class, sequence_info = get_random_sequence(
                test_data, label_encoder
            )
            
            # 預測
            predicted_label, confidence, all_probs = predict_sequence(model, sequence, device)
            predicted_class = label_encoder.classes_[predicted_label]
            
            # 結果
            is_correct = predicted_label == true_label
            if is_correct:
                correct_predictions += 1
            
            # 顯示結果
            print(f"📍 序列來源: {sequence_info}")
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
            print("🌟 模型表現優秀!")
        elif accuracy >= 0.6:
            print("👍 模型表現良好!")
        else:
            print("📈 模型還有改進空間")
            
    except Exception as e:
        print(f"❌ 測試過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
