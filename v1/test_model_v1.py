"""
手語辨識模型測試腳本 v1
功能：
1. 載入訓練好的模型
2. 使用測試集驗證模型能力
3. 生成詳細的評估報告
4. 顯示混淆矩陣和分類報告
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

class SignLanguageTester:
    def __init__(self, data_folder="v1/processed_data", model_folder="v1/models", results_folder="v1/results"):
        self.data_folder = data_folder
        self.model_folder = model_folder
        self.results_folder = results_folder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_encoder = None
        
        print(f"使用設備: {self.device}")
        
    def setup_directories(self):
        """建立結果輸出資料夾"""
        os.makedirs(self.results_folder, exist_ok=True)
        print(f"結果資料夾已建立: {self.results_folder}")
    
    def load_model(self, model_path=None):
        """載入訓練好的模型"""
        if model_path is None:
            model_path = os.path.join(self.model_folder, "latest_model.pth")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型檔案: {model_path}")
        
        print(f"載入模型: {model_path}")
        
        # 載入模型資料
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint['model_config']
        self.label_encoder = checkpoint['label_encoder']
        
        # 創建模型實例
        self.model = SignLanguageGRU(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_classes=model_config['num_classes']
        ).to(self.device)
        
        # 載入模型權重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"模型架構: {model_config}")
        print(f"類別數量: {len(self.label_encoder.classes_)}")
        print(f"類別: {list(self.label_encoder.classes_)}")
        
        return checkpoint
    
    def load_test_data(self):
        """載入測試資料"""
        test_path = os.path.join(self.data_folder, "test_dataset.csv")
        
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"找不到測試資料集: {test_path}")
        
        test_data = pd.read_csv(test_path)
        print(f"載入測試資料: {test_data.shape}")
        
        if len(test_data) == 0:
            raise ValueError("測試資料集為空")
        
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
                print("⚠️  使用基礎缺失值處理...")
                # 基礎處理：填充 0
                data.fillna(0, inplace=True)
        else:
            print("✅ 測試數據沒有缺失值")
    
    def prepare_test_sequences(self, data, sequence_length=20):
        """準備測試序列資料"""
        # 特徵欄位（排除標籤相關欄位和frame，與訓練時保持一致）
        feature_cols = [col for col in data.columns 
                       if col not in ['sign_language', 'sign_language_encoded', 'frame', 'source_video']]
        
        print(f"測試特徵維度: {len(feature_cols)} (排除: sign_language, sign_language_encoded, frame, source_video)")
        
        # 按類別分組創建序列
        sequences = []
        labels = []
        class_names = []
        
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
        
        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        print(f"測試序列形狀: {sequences.shape}")
        print(f"測試標籤形狀: {labels.shape}")
        
        return sequences, labels, class_names
    
    def evaluate_model(self, X_test, y_test, batch_size=32):
        """評估模型性能"""
        # 創建測試資料載入器
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 模型評估
        self.model.eval()
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # 預測
                outputs = self.model(batch_X)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # 儲存結果
                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(batch_y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_true_labels = np.array(all_true_labels)
        all_probabilities = np.array(all_probabilities)
        
        return all_predictions, all_true_labels, all_probabilities
    
    def generate_classification_report(self, y_true, y_pred):
        """生成分類報告"""
        # 計算準確率
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n整體準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 生成詳細的分類報告
        class_names = self.label_encoder.classes_
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True
        )
        
        # 打印分類報告
        print("\n詳細分類報告:")
        print("-" * 60)
        print(f"{'類別':<15} {'精確度':<10} {'召回率':<10} {'F1分數':<10} {'支援度':<10}")
        print("-" * 60)
        
        for class_name in class_names:
            metrics = report[class_name]
            print(f"{class_name:<15} {metrics['precision']:<10.3f} "
                  f"{metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f} "
                  f"{int(metrics['support']):<10}")
        
        print("-" * 60)
        print(f"{'平均':<15} {report['weighted avg']['precision']:<10.3f} "
              f"{report['weighted avg']['recall']:<10.3f} "
              f"{report['weighted avg']['f1-score']:<10.3f} "
              f"{int(report['weighted avg']['support']):<10}")
        
        return report
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """繪製混淆矩陣"""
        class_names = self.label_encoder.classes_
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('混淆矩陣 (Confusion Matrix)')
        plt.xlabel('預測類別')
        plt.ylabel('真實類別')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 儲存圖表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.results_folder, f"confusion_matrix_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩陣已儲存: {plot_path}")
        
        plt.show()
        
        return cm
    
    def plot_class_performance(self, report):
        """繪製各類別性能圖表"""
        class_names = self.label_encoder.classes_
        metrics = ['precision', 'recall', 'f1-score']
        
        # 準備資料
        performance_data = []
        for class_name in class_names:
            if class_name in report:  # 確保類別存在於報告中
                for metric in metrics:
                    performance_data.append({
                        'Class': class_name,
                        'Metric': metric,
                        'Score': report[class_name][metric]
                    })
        
        df_performance = pd.DataFrame(performance_data)
        
        # 繪製圖表
        plt.figure(figsize=(15, 8))
        sns.barplot(data=df_performance, x='Class', y='Score', hue='Metric')
        plt.title('各類別性能指標')
        plt.xlabel('手語類別')
        plt.ylabel('分數')
        plt.xticks(rotation=45)
        plt.legend(title='指標')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # 儲存圖表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.results_folder, f"class_performance_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"類別性能圖表已儲存: {plot_path}")
        
        plt.show()
    
    def save_results(self, y_true, y_pred, y_prob, class_names, report):
        """儲存詳細測試結果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 創建結果DataFrame
        results_df = pd.DataFrame({
            'True_Label': [self.label_encoder.classes_[label] for label in y_true],
            'Predicted_Label': [self.label_encoder.classes_[label] for label in y_pred],
            'True_Label_Encoded': y_true,
            'Predicted_Label_Encoded': y_pred,
            'Correct': y_true == y_pred
        })
        
        # 添加機率分數
        for i, class_name in enumerate(self.label_encoder.classes_):
            results_df[f'Prob_{class_name}'] = y_prob[:, i]
        
        # 儲存結果
        results_path = os.path.join(self.results_folder, f"test_results_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"詳細測試結果已儲存: {results_path}")
        
        # 儲存分類報告
        report_path = os.path.join(self.results_folder, f"classification_report_{timestamp}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("手語辨識模型測試報告\n")
            f.write("=" * 50 + "\n")
            f.write(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"測試樣本數: {len(y_true)}\n")
            f.write(f"整體準確率: {accuracy_score(y_true, y_pred):.4f}\n\n")
            
            f.write("詳細分類報告:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'類別':<15} {'精確度':<10} {'召回率':<10} {'F1分數':<10} {'支援度':<10}\n")
            f.write("-" * 60 + "\n")
            
            for class_name in self.label_encoder.classes_:
                if class_name in report:
                    metrics = report[class_name]
                    f.write(f"{class_name:<15} {metrics['precision']:<10.3f} "
                           f"{metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f} "
                           f"{int(metrics['support']):<10}\n")
        
        print(f"分類報告已儲存: {report_path}")
    
    def run_testing(self, model_path=None, sequence_length=20, batch_size=32):
        """執行完整的測試流程"""
        print("=" * 70)
        print("手語辨識模型測試 v1")
        print("=" * 70)
        
        # 1. 建立輸出資料夾
        self.setup_directories()
        
        # 2. 載入模型
        print("\n步驟 1: 載入模型...")
        checkpoint = self.load_model(model_path)
        
        # 3. 載入測試資料
        print("\n步驟 2: 載入測試資料...")
        test_data = self.load_test_data()
        
        # 4. 準備測試序列
        print("\n步驟 3: 準備測試序列...")
        X_test, y_test, class_names = self.prepare_test_sequences(test_data, sequence_length)
        
        # 5. 評估模型
        print("\n步驟 4: 評估模型...")
        y_pred, y_true, y_prob = self.evaluate_model(X_test, y_test, batch_size)
        
        # 6. 生成分類報告
        print("\n步驟 5: 生成分類報告...")
        report = self.generate_classification_report(y_true, y_pred)
        
        # 7. 繪製混淆矩陣
        print("\n步驟 6: 繪製混淆矩陣...")
        confusion_mat = self.plot_confusion_matrix(y_true, y_pred)
        
        # 8. 繪製類別性能
        print("\n步驟 7: 繪製類別性能...")
        self.plot_class_performance(report)
        
        # 9. 儲存結果
        print("\n步驟 8: 儲存結果...")
        self.save_results(y_true, y_pred, y_prob, class_names, report)
        
        print("\n" + "=" * 70)
        print("模型測試完成!")
        print(f"整體準確率: {accuracy_score(y_true, y_pred):.4f} ({accuracy_score(y_true, y_pred)*100:.2f}%)")
        print("=" * 70)

def main():
    """主函數"""
    try:
        tester = SignLanguageTester()
        tester.run_testing(
            model_path=None,    # 使用最新模型，或指定特定模型路徑
            sequence_length=20, # 必須與訓練時相同
            batch_size=32
        )
    except Exception as e:
        print(f"測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
