"""
手語辨識模型訓練腳本 v1
功能：
1. 載入預處理後的訓練資料集
2. 建立和配置深度學習模型
3. 訓練模型
4. 儲存訓練好的模型
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SignLanguageGRU(nn.Module):
    """手語辨識GRU模型"""
    
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

class SignLanguageTrainer:
    def __init__(self, data_folder="v1/processed_data", model_folder="v1/models"):
        self.data_folder = data_folder
        self.model_folder = model_folder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_encoder = None
        
        print(f"使用設備: {self.device}")
        
    def setup_directories(self):
        """建立模型輸出資料夾"""
        os.makedirs(self.model_folder, exist_ok=True)
        print(f"模型資料夾已建立: {self.model_folder}")
    
    def load_data(self):
        """載入預處理後的資料"""
        train_path = os.path.join(self.data_folder, "train_dataset.csv")
        encoder_path = os.path.join(self.data_folder, "label_encoder.pkl")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"找不到訓練資料集: {train_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"找不到標籤編碼器: {encoder_path}")
        
        # 載入訓練資料
        train_data = pd.read_csv(train_path)
        print(f"載入訓練資料: {train_data.shape}")
        
        # 載入標籤編碼器
        self.label_encoder = joblib.load(encoder_path)
        print(f"類別數量: {len(self.label_encoder.classes_)}")
        print(f"類別: {list(self.label_encoder.classes_)}")
        
        return train_data
    
    def analyze_sequence_lengths(self, data):
        """分析各類別的序列長度分布"""
        print("\n📊 序列長度分析:")
        print("-" * 50)
        
        length_stats = {}
        for sign_language in data['sign_language'].unique():
            class_data = data[data['sign_language'] == sign_language]
            length = len(class_data)
            length_stats[sign_language] = length
            
            print(f"{sign_language:<15}: {length:>3} 幀")
        
        # 統計摘要
        lengths = list(length_stats.values())
        print("-" * 50)
        print(f"{'平均長度':<15}: {np.mean(lengths):>6.1f} 幀")
        print(f"{'最短動作':<15}: {np.min(lengths):>6} 幀")
        print(f"{'最長動作':<15}: {np.max(lengths):>6} 幀")
        print(f"{'標準差':<15}: {np.std(lengths):>6.1f} 幀")
        
        # 建議最佳sequence_length
        recommended_length = int(np.percentile(lengths, 70))  # 70%分位數
        print(f"{'建議序列長度':<15}: {recommended_length:>6} 幀 (覆蓋70%動作)")
        print("-" * 50)
        
        return length_stats, recommended_length

    def prepare_sequences(self, data, sequence_length=30):
        """
        準備序列資料 - 將CSV表格轉換為GRU模型所需的時序序列格式
        
        目標: 
        - 輸入: CSV表格 (每行=一幀, 包含163維特徵)
        - 輸出: 3D張量 (樣本數, 時間步, 特徵維度)
        
        Args:
            data: 預處理後的訓練資料 (DataFrame)
            sequence_length: 每個序列的時間步數 (預設30幀)
            
        Returns:
            sequences: 形狀為 (樣本數, sequence_length, 特徵維度) 的3D張量
            labels: 形狀為 (樣本數,) 的標籤陣列
        """
        
        print(f"開始準備序列資料，目標序列長度: {sequence_length}")
        
        # 步驟1: 提取特徵欄位 (移除標籤欄位，只保留座標特徵)
        feature_cols = [col for col in data.columns 
                       if col not in ['sign_language', 'sign_language_encoded']]
        
        print(f"特徵欄位數量: {len(feature_cols)}")
        print(f"特徵欄位包含: frame + 姿態座標 + 左手座標 + 右手座標")
        
        # 步驟2: 初始化序列容器
        sequences = []  # 儲存所有時序序列
        labels = []     # 儲存對應的標籤
        
        # 步驟3: 按手語類別分組處理
        unique_classes = data['sign_language'].unique()
        print(f"處理 {len(unique_classes)} 個手語類別: {list(unique_classes)}")
        
        for sign_language in unique_classes:
            # 提取該類別的所有幀資料
            class_data = data[data['sign_language'] == sign_language]
            num_frames = len(class_data)
            
            print(f"\n處理類別 '{sign_language}': {num_frames} 幀")
            
            # 情況A: 資料充足，使用滑動窗口創建多個序列
            if num_frames >= sequence_length:
                num_sequences = num_frames - sequence_length + 1
                print(f"  → 使用滑動窗口創建 {num_sequences} 個序列")
                
                for i in range(num_sequences):
                    # 提取連續的sequence_length幀作為一個序列
                    seq = class_data.iloc[i:i+sequence_length][feature_cols].values
                    sequences.append(seq)
                    labels.append(class_data.iloc[i]['sign_language_encoded'])
                    
            # 情況B: 資料不足，使用填充策略
            else:
                print(f"  → 資料不足，進行填充 ({num_frames} → {sequence_length} 幀)")
                
                seq_data = class_data[feature_cols].values
                
                if len(seq_data) < sequence_length:
                    # 計算需要填充的幀數
                    padding_needed = sequence_length - len(seq_data)
                    
                    # 使用最後一幀重複填充
                    last_frame = seq_data[-1:] if len(seq_data) > 0 else np.zeros((1, len(feature_cols)))
                    padding = np.repeat(last_frame, padding_needed, axis=0)
                    seq_data = np.vstack([seq_data, padding])
                    
                    print(f"  → 重複最後一幀 {padding_needed} 次進行填充")
                
                sequences.append(seq_data)
                labels.append(class_data.iloc[0]['sign_language_encoded'])
        
        # 步驟4: 轉換為NumPy陣列
        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # 步驟5: 顯示最終結果
        print(f"\n✅ 序列準備完成!")
        print(f"序列形狀: {sequences.shape}")
        print(f"  - 訓練樣本數: {sequences.shape[0]}")
        print(f"  - 序列長度(時間步): {sequences.shape[1]}")
        print(f"  - 特徵維度: {sequences.shape[2]}")
        print(f"標籤形狀: {labels.shape}")
        
        return sequences, labels
    
    def create_model(self, input_size, num_classes):
        """創建模型"""
        self.model = SignLanguageGRU(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3
        ).to(self.device)
        
        print(f"模型參數數量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"可訓練參數數量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        return self.model
    
    def train_model(self, X_train, y_train, epochs=100, batch_size=32, learning_rate=0.001):
        """訓練模型"""
        # 創建資料載入器
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 定義損失函數和優化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        # 訓練記錄
        train_losses = []
        train_accuracies = []
        
        print(f"\n開始訓練 - Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
        print("-" * 70)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # 前向傳播
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # 反向傳播
                loss.backward()
                optimizer.step()
                
                # 統計
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            # 計算平均損失和準確率
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            
            # 學習率調整
            scheduler.step(avg_loss)
            
            # 每10個epoch或最後一個epoch打印結果
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Accuracy: {accuracy:.2f}% | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        print("-" * 70)
        print("訓練完成!")
        
        return train_losses, train_accuracies
    
    def save_model(self, train_losses, train_accuracies):
        """儲存模型和訓練記錄"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 儲存模型
        model_path = os.path.join(self.model_folder, f"sign_language_gru_v1_{timestamp}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.gru.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'num_classes': len(self.label_encoder.classes_)
            },
            'label_encoder': self.label_encoder,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies
        }, model_path)
        
        print(f"模型已儲存: {model_path}")
        
        # 儲存最新模型的路徑（用於測試）
        latest_model_path = os.path.join(self.model_folder, "latest_model.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.gru.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'num_classes': len(self.label_encoder.classes_)
            },
            'label_encoder': self.label_encoder,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies
        }, latest_model_path)
        
        print(f"最新模型連結已更新: {latest_model_path}")
        
        return model_path
    
    def plot_training_curves(self, train_losses, train_accuracies):
        """繪製訓練曲線"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 損失曲線
        ax1.plot(train_losses, 'b-', label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 準確率曲線
        ax2.plot(train_accuracies, 'r-', label='Training Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 儲存圖表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.model_folder, f"training_curves_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"訓練曲線已儲存: {plot_path}")
        
        plt.show()
    
    def run_training(self, epochs=100, batch_size=32, learning_rate=0.001, sequence_length=30):
        """執行完整的訓練流程"""
        print("=" * 70)
        print("手語辨識模型訓練 v1")
        print("=" * 70)
        
        # 1. 建立輸出資料夾
        self.setup_directories()
        
        # 2. 載入資料
        print("\n步驟 1: 載入資料...")
        train_data = self.load_data()
        
        # 3. 分析序列長度分布
        print("\n步驟 2a: 分析序列長度分布...")
        length_stats, recommended_length = self.analyze_sequence_lengths(train_data)
        
        # 根據分析結果調整sequence_length（可選）
        if sequence_length == 20:  # 如果使用預設值，考慮使用建議值
            print(f"\n💡 建議使用序列長度: {recommended_length} (當前使用: {sequence_length})")
        
        # 4. 準備序列資料
        print("\n步驟 2b: 準備序列資料...")
        X_train, y_train = self.prepare_sequences(train_data, sequence_length)
        
        # 5. 創建模型
        print("\n步驟 3: 創建模型...")
        input_size = X_train.shape[2]  # 特徵維度
        num_classes = len(self.label_encoder.classes_)
        self.create_model(input_size, num_classes)
        
        # 6. 訓練模型
        print("\n步驟 4: 訓練模型...")
        train_losses, train_accuracies = self.train_model(
            X_train, y_train, epochs, batch_size, learning_rate
        )
        
        # 7. 儲存模型
        print("\n步驟 5: 儲存模型...")
        model_path = self.save_model(train_losses, train_accuracies)
        
        # 8. 繪製訓練曲線
        print("\n步驟 6: 繪製訓練曲線...")
        self.plot_training_curves(train_losses, train_accuracies)
        
        print("\n" + "=" * 70)
        print("模型訓練完成!")
        print(f"最終訓練準確率: {train_accuracies[-1]:.2f}%")
        print(f"模型檔案: {model_path}")
        print("=" * 70)

def main():
    """主函數"""
    try:
        trainer = SignLanguageTrainer()
        trainer.run_training(
            epochs=50,          # 訓練週期
            batch_size=16,      # 批次大小（考慮到GPU記憶體限制）
            learning_rate=0.001,# 學習率
            sequence_length=20  # 序列長度
        )
    except Exception as e:
        print(f"訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
