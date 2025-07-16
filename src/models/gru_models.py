"""
手語辨識GRU模型集合
包含三個複雜度級別的模型，基於數據分析結果設計
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class SimpleGRU(nn.Module):
    """
    簡單GRU模型 - 快速驗證用
    
    設計邏輯：
    - 單層GRU，快速訓練
    - 目標：>70%準確率，1-2小時訓練
    - 用於驗證數據管道和基本可行性
    """
    
    def __init__(self, 
                 input_size: int = 162,  # 基於數據分析：身體36+左手63+右手63
                 hidden_size: int = 64,
                 num_classes: int = 34,  # 基於數據分析：34種手語
                 dropout_rate: float = 0.3):
        super(SimpleGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # GRU層
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0  # 單層不需要dropout
        )
        
        # 正則化
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分類層
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """權重初始化"""
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入序列 (batch_size, sequence_length, input_size)
            hidden: 隱藏狀態 (可選)
            
        Returns:
            output: 分類結果 (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # GRU處理
        gru_out, _ = self.gru(x, hidden)  # (batch, seq, hidden)
        
        # 使用最後一個時間步的輸出
        last_output = gru_out[:, -1, :]  # (batch, hidden)
        
        # 正則化
        last_output = self.dropout(last_output)
        
        # 分類
        output = self.classifier(last_output)  # (batch, num_classes)
        
        return output
    
    def get_model_info(self) -> dict:
        """獲取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'SimpleGRU',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size, 
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 假設float32
        }

class AttentionGRU(nn.Module):
    """
    注意力GRU模型 - 中等複雜度
    
    設計邏輯：
    - 多層GRU + 注意力機制
    - 目標：>85%準確率，3-5小時訓練
    - 關注手語動作的關鍵幀
    """
    
    def __init__(self,
                 input_size: int = 162,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 34,
                 num_attention_heads: int = 8,
                 dropout_rate: float = 0.4):
        super(AttentionGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_attention_heads = num_attention_heads
        
        # GRU層
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 注意力機制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 正則化
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # 分類器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """權重初始化"""
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入序列 (batch_size, sequence_length, input_size)
            hidden: 隱藏狀態 (可選)
            
        Returns:
            output: 分類結果 (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # GRU處理
        gru_out, _ = self.gru(x, hidden)  # (batch, seq, hidden)
        
        # 層標準化
        gru_out = self.layer_norm(gru_out)
        
        # 自注意力機制
        attn_out, attn_weights = self.attention(
            query=gru_out,
            key=gru_out, 
            value=gru_out
        )  # (batch, seq, hidden)
        
        # 殘差連接
        combined = gru_out + attn_out
        combined = self.dropout(combined)
        
        # 全局平均池化（考慮所有時間步）
        pooled = torch.mean(combined, dim=1)  # (batch, hidden)
        
        # 分類
        output = self.classifier(pooled)  # (batch, num_classes)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """獲取注意力權重用於可視化"""
        with torch.no_grad():
            gru_out, _ = self.gru(x)
            gru_out = self.layer_norm(gru_out)
            _, attn_weights = self.attention(gru_out, gru_out, gru_out)
            return attn_weights
    
    def get_model_info(self) -> dict:
        """獲取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'AttentionGRU',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_attention_heads': self.num_attention_heads,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }

class BiGRUWithSelfAttention(nn.Module):
    """
    雙向GRU + 自注意力模型 - 高複雜度
    
    設計邏輯：
    - 雙向GRU + 多頭自注意力
    - 目標：>90%準確率，8-12小時訓練
    - 最佳性能，適合最終部署
    """
    
    def __init__(self,
                 input_size: int = 162,
                 hidden_size: int = 256,
                 num_layers: int = 3,
                 num_classes: int = 34,
                 num_attention_heads: int = 16,
                 dropout_rate: float = 0.5):
        super(BiGRUWithSelfAttention, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_attention_heads = num_attention_heads
        
        # 雙向GRU
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 雙向輸出的尺寸是 hidden_size * 2
        bigru_output_size = hidden_size * 2
        
        # 多頭自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=bigru_output_size,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 位置編碼（可選）
        self.positional_encoding = self._create_positional_encoding(max_len=50, d_model=bigru_output_size)
        
        # 正則化層
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(bigru_output_size)
        self.layer_norm2 = nn.LayerNorm(bigru_output_size)
        
        # 分類器
        self.classifier = nn.Sequential(
            nn.Linear(bigru_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # 初始化權重
        self._init_weights()
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """創建位置編碼"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def _init_weights(self):
        """權重初始化"""
        for name, param in self.bigru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入序列 (batch_size, sequence_length, input_size)
            hidden: 隱藏狀態 (可選)
            
        Returns:
            output: 分類結果 (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # 雙向GRU處理
        bigru_out, _ = self.bigru(x, hidden)  # (batch, seq, hidden*2)
        
        # 添加位置編碼
        if seq_len <= self.positional_encoding.shape[1]:
            pos_encoding = self.positional_encoding[:, :seq_len, :].to(x.device)
            bigru_out = bigru_out + pos_encoding
        
        # 第一次層標準化
        bigru_out = self.layer_norm1(bigru_out)
        
        # 自注意力機制
        attn_out, attn_weights = self.self_attention(
            query=bigru_out,
            key=bigru_out,
            value=bigru_out
        )  # (batch, seq, hidden*2)
        
        # 殘差連接 + 層標準化
        combined = self.layer_norm2(bigru_out + attn_out)
        combined = self.dropout(combined)
        
        # 多種池化策略的組合
        max_pooled = torch.max(combined, dim=1)[0]  # (batch, hidden*2)
        avg_pooled = torch.mean(combined, dim=1)    # (batch, hidden*2)
        
        # 組合不同的池化結果
        final_representation = max_pooled + avg_pooled  # (batch, hidden*2)
        
        # 分類
        output = self.classifier(final_representation)  # (batch, num_classes)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """獲取注意力權重用於可視化"""
        with torch.no_grad():
            bigru_out, _ = self.bigru(x)
            
            if x.shape[1] <= self.positional_encoding.shape[1]:
                pos_encoding = self.positional_encoding[:, :x.shape[1], :].to(x.device)
                bigru_out = bigru_out + pos_encoding
                
            bigru_out = self.layer_norm1(bigru_out)
            _, attn_weights = self.self_attention(bigru_out, bigru_out, bigru_out)
            return attn_weights
    
    def get_model_info(self) -> dict:
        """獲取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'BiGRUWithSelfAttention',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_attention_heads': self.num_attention_heads,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }

def create_model(model_type: str = 'simple', **kwargs) -> nn.Module:
    """
    模型工廠函數
    
    Args:
        model_type: 模型類型 ('simple', 'attention', 'bigru')
        **kwargs: 模型參數
        
    Returns:
        模型實例
    """
    model_classes = {
        'simple': SimpleGRU,
        'attention': AttentionGRU, 
        'bigru': BiGRUWithSelfAttention
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_classes.keys())}")
    
    return model_classes[model_type](**kwargs)

def test_models():
    """測試所有模型"""
    print("🧪 測試GRU模型")
    print("=" * 50)
    
    # 模擬輸入數據
    batch_size = 8
    sequence_length = 30
    input_size = 162
    
    x = torch.randn(batch_size, sequence_length, input_size)
    print(f"📊 輸入形狀: {x.shape}")
    
    models = {
        'simple': SimpleGRU(),
        'attention': AttentionGRU(),
        'bigru': BiGRUWithSelfAttention()
    }
    
    for name, model in models.items():
        print(f"\n🔍 測試 {name.upper()} 模型:")
        
        # 前向傳播測試
        try:
            output = model(x)
            print(f"   ✅ 輸出形狀: {output.shape}")
            
            # 模型信息
            info = model.get_model_info()
            print(f"   📈 參數數量: {info['total_parameters']:,}")
            print(f"   💾 模型大小: {info['model_size_mb']:.2f} MB")
            
            # 內存使用估算
            memory_mb = batch_size * sequence_length * input_size * 4 / (1024 * 1024)
            print(f"   🧠 輸入記憶體: {memory_mb:.2f} MB")
            
        except Exception as e:
            print(f"   ❌ 測試失敗: {str(e)}")

if __name__ == "__main__":
    test_models()
