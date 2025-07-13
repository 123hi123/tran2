"""
手語辨識GRU模型訓練腳本
基於數據分析結果的優化訓練策略
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.gru_models import create_model
from src.data_preprocessing import SignLanguagePreprocessor

class SignLanguageDataset(Dataset):
    """手語數據集"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, label_encoder: LabelEncoder):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(label_encoder.transform(labels))
        self.label_encoder = label_encoder
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class EarlyStopping:
    """早停機制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True
            return False

class TrainingConfig:
    """訓練配置"""
    
    def __init__(self):
        # 基於數據分析的配置
        self.batch_size = 16  # 基於記憶體分析
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.early_stopping_patience = 15
        self.sequence_length = 30
        self.stride = 15
        
        # 數據分割
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
        # 設備配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型保存
        self.save_dir = 'models'
        self.log_dir = 'logs'
        
        # 創建目錄
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

class SignLanguageTrainer:
    """手語辨識模型訓練器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device
        self.label_encoder = LabelEncoder()
        
        # 訓練歷史
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"🔧 初始化訓練器")
        print(f"   設備: {self.device}")
        print(f"   批次大小: {config.batch_size}")
        print(f"   學習率: {config.learning_rate}")
    
    def prepare_data(self, csv_files: List[str]) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        準備訓練數據
        
        Args:
            csv_files: CSV檔案列表
            
        Returns:
            train_loader, val_loader, test_loader
        """
        print("📊 準備訓練數據...")
        
        # 初始化預處理器
        preprocessor = SignLanguagePreprocessor(
            sequence_length=self.config.sequence_length,
            stride=self.config.stride
        )
        
        all_sequences = []
        all_labels = []
        
        # 處理每個檔案
        for i, csv_file in enumerate(csv_files):
            print(f"   處理檔案 {i+1}/{len(csv_files)}: {csv_file}")
            
            try:
                # 載入數據（可能需要分批處理大檔案）
                if os.path.getsize(csv_file) > 500 * 1024 * 1024:  # 大於500MB
                    # 分批處理
                    sequences, labels = self._process_large_file(csv_file, preprocessor)
                else:
                    df = pd.read_csv(csv_file)
                    df_clean = preprocessor.handle_missing_values(df)
                    df_normalized = preprocessor.normalize_coordinates(df_clean)
                    sequences, labels = preprocessor.create_sequences(df_normalized)
                
                if len(sequences) > 0:
                    all_sequences.append(sequences)
                    all_labels.extend(labels)
                    print(f"     生成序列: {len(sequences)}")
                
            except Exception as e:
                print(f"     ❌ 處理失敗: {str(e)}")
                continue
        
        if not all_sequences:
            raise ValueError("未能生成任何有效序列")
        
        # 合併所有序列
        final_sequences = np.concatenate(all_sequences, axis=0)
        final_labels = np.array(all_labels)
        
        print(f"📈 總序列數: {len(final_sequences)}")
        print(f"🏷️ 標籤種類: {len(np.unique(final_labels))}")
        
        # 編碼標籤
        self.label_encoder.fit(final_labels)
        num_classes = len(self.label_encoder.classes_)
        print(f"📋 類別列表: {list(self.label_encoder.classes_)}")
        
        # 創建數據集
        dataset = SignLanguageDataset(final_sequences, final_labels, self.label_encoder)
        
        # 分割數據集
        train_size = int(len(dataset) * self.config.train_ratio)
        val_size = int(len(dataset) * self.config.val_ratio)
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"📊 數據分割:")
        print(f"   訓練集: {len(train_dataset)}")
        print(f"   驗證集: {len(val_dataset)}")
        print(f"   測試集: {len(test_dataset)}")
        
        # 創建數據載入器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Windows兼容性
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader, num_classes
    
    def _process_large_file(self, csv_file: str, preprocessor: SignLanguagePreprocessor) -> Tuple[np.ndarray, List]:
        """處理大檔案"""
        print(f"     📦 分批處理大檔案...")
        
        all_sequences = []
        all_labels = []
        
        chunk_size = 10000
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            try:
                chunk_clean = preprocessor.handle_missing_values(chunk)
                chunk_normalized = preprocessor.normalize_coordinates(chunk_clean)
                sequences, labels = preprocessor.create_sequences(chunk_normalized)
                
                if len(sequences) > 0:
                    all_sequences.append(sequences)
                    all_labels.extend(labels)
                    
            except Exception as e:
                print(f"       ⚠️ 塊處理警告: {str(e)}")
                continue
        
        if all_sequences:
            return np.concatenate(all_sequences, axis=0), all_labels
        else:
            return np.array([]), []
    
    def create_model(self, model_type: str, num_classes: int, input_size: int = 162) -> nn.Module:
        """創建模型"""
        model = create_model(
            model_type=model_type,
            input_size=input_size,
            num_classes=num_classes
        )
        
        model = model.to(self.device)
        
        # 打印模型信息
        info = model.get_model_info()
        print(f"🤖 創建模型: {info['model_name']}")
        print(f"   參數數量: {info['total_parameters']:,}")
        print(f"   模型大小: {info['model_size_mb']:.2f} MB")
        
        return model
    
    def calculate_class_weights(self, train_loader: DataLoader) -> torch.Tensor:
        """計算類別權重處理不平衡問題"""
        print("⚖️ 計算類別權重...")
        
        # 統計各類別樣本數
        class_counts = torch.zeros(len(self.label_encoder.classes_))
        
        for _, labels in train_loader:
            for label in labels:
                class_counts[label] += 1
        
        # 計算權重（反比例）
        total_samples = class_counts.sum()
        class_weights = total_samples / (len(class_counts) * class_counts)
        
        # 避免無窮大
        class_weights = torch.where(class_counts == 0, torch.tensor(1.0), class_weights)
        
        print(f"   類別分布: {class_counts.numpy()}")
        print(f"   類別權重: {class_weights.numpy()}")
        
        return class_weights.to(self.device)
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, model_name: str = "model") -> Dict:
        """訓練模型"""
        print(f"🚀 開始訓練模型: {model_name}")
        print("=" * 60)
        
        # 計算類別權重
        class_weights = self.calculate_class_weights(train_loader)
        
        # 損失函數和優化器
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        # 早停機制
        early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)
        
        # 訓練循環
        start_time = time.time()
        best_val_acc = 0.0
        
        for epoch in range(self.config.num_epochs):
            # 訓練階段
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)
            
            # 驗證階段
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)
            
            # 學習率調度
            scheduler.step(val_loss)
            
            # 記錄歷史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            
            # 打印進度
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{self.config.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | "
                  f"Time: {elapsed_time:.1f}s")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_model(model, model_name, epoch, val_acc)
            
            # 早停檢查
            if early_stopping(val_loss, model):
                print(f"🛑 早停於 epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"✅ 訓練完成，總時間: {total_time:.1f}秒")
        print(f"🏆 最佳驗證準確率: {best_val_acc:.3f}")
        
        return {
            'best_val_acc': best_val_acc,
            'total_epochs': epoch + 1,
            'total_time': total_time,
            'final_lr': optimizer.param_groups[0]['lr']
        }
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                    criterion: nn.Module, optimizer: optim.Optimizer) -> Tuple[float, float]:
        """訓練一個epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(self.device), labels.to(self.device)
            
            # 前向傳播
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 統計
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, float]:
        """驗證一個epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _save_model(self, model: nn.Module, model_name: str, epoch: int, val_acc: float):
        """保存模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_epoch{epoch:03d}_acc{val_acc:.3f}_{timestamp}.pth"
        filepath = os.path.join(self.config.save_dir, filename)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'val_acc': val_acc,
            'label_encoder': self.label_encoder,
            'config': self.config.__dict__
        }, filepath)
        
        print(f"💾 保存模型: {filename}")
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict:
        """評估模型"""
        print("📊 模型評估...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(self.device)
                outputs = model(sequences)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # 計算指標
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # 混淆矩陣
        cm = confusion_matrix(all_labels, all_predictions)
        
        # 分類報告
        class_names = self.label_encoder.classes_
        report = classification_report(
            all_labels, all_predictions,
            target_names=class_names,
            output_dict=True
        )
        
        print(f"🎯 測試準確率: {accuracy:.3f}")
        print(f"📈 F1分數: {f1:.3f}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': all_predictions,
            'true_labels': all_labels
        }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """繪製訓練歷史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 損失曲線
        ax1.plot(self.train_history['train_loss'], label='訓練損失')
        ax1.plot(self.train_history['val_loss'], label='驗證損失')
        ax1.set_title('損失曲線')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 準確率曲線
        ax2.plot(self.train_history['train_acc'], label='訓練準確率')
        ax2.plot(self.train_history['val_acc'], label='驗證準確率')
        ax2.set_title('準確率曲線')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 訓練曲線保存至: {save_path}")
        
        plt.show()

def main():
    """主訓練流程"""
    print("🚀 手語辨識GRU模型訓練")
    print("🕐 開始時間:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    # 配置
    config = TrainingConfig()
    trainer = SignLanguageTrainer(config)
    
    # 數據檔案（根據實際路徑調整）
    csv_files = [
        "dataset/sign_language1.csv",
        "dataset/sign_language2.csv",
        # 可以添加更多檔案，但建議先用小樣本測試
    ]
    
    # 檢查檔案存在性
    available_files = [f for f in csv_files if os.path.exists(f)]
    if not available_files:
        print("❌ 未找到任何數據檔案")
        return
    
    print(f"📄 找到 {len(available_files)} 個數據檔案")
    
    try:
        # 準備數據
        train_loader, val_loader, test_loader, num_classes = trainer.prepare_data(available_files)
        
        # 訓練不同複雜度的模型
        models_to_train = [
            ('simple', '簡單GRU模型'),
            ('attention', '注意力GRU模型'),
            # ('bigru', '雙向GRU模型')  # 可選，時間較長
        ]
        
        results = {}
        
        for model_type, model_desc in models_to_train:
            print(f"\n🤖 訓練 {model_desc}")
            print("-" * 40)
            
            # 創建模型
            model = trainer.create_model(model_type, num_classes)
            
            # 訓練
            train_result = trainer.train_model(model, train_loader, val_loader, model_type)
            
            # 評估
            eval_result = trainer.evaluate_model(model, test_loader)
            
            # 保存結果
            results[model_type] = {
                'train': train_result,
                'eval': eval_result
            }
            
            # 繪製訓練曲線
            trainer.plot_training_history(
                save_path=f"{config.log_dir}/{model_type}_training_history.png"
            )
            
            # 重置訓練歷史
            trainer.train_history = {
                'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': []
            }
        
        # 總結結果
        print("\n📊 訓練總結")
        print("=" * 60)
        
        for model_type, result in results.items():
            print(f"🤖 {model_type.upper()} 模型:")
            print(f"   最佳驗證準確率: {result['train']['best_val_acc']:.3f}")
            print(f"   測試準確率: {result['eval']['accuracy']:.3f}")
            print(f"   F1分數: {result['eval']['f1_score']:.3f}")
            print(f"   訓練時間: {result['train']['total_time']:.1f}秒")
            print()
        
        # 保存完整結果
        with open(f"{config.log_dir}/training_results.json", 'w', encoding='utf-8') as f:
            # 序列化結果（移除不可序列化的對象）
            serializable_results = {}
            for model_type, result in results.items():
                serializable_results[model_type] = {
                    'train': result['train'],
                    'eval': {
                        'accuracy': result['eval']['accuracy'],
                        'f1_score': result['eval']['f1_score']
                    }
                }
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print("✅ 訓練流程完成!")
        print("🕐 結束時間:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
    except Exception as e:
        print(f"❌ 訓練過程出錯: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
