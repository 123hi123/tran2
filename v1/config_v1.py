"""
手語辨識系統配置文件 v1
所有可調整的參數都集中在這裡，方便用戶根據硬體環境和需求進行調整
"""

# ============================
# 資料預處理配置
# ============================
DATA_CONFIG = {
    # 輸入資料夾
    "dataset_folder": "dataset",
    
    # 輸出資料夾  
    "output_folder": "v1/processed_data",
    
    # 最小樣本數（不足時進行資料增強）
    "min_samples_per_class": 5,
    
    # 訓練/測試分割比例
    "test_size": 0.2,
    
    # 隨機種子（確保結果可重現）
    "random_state": 42,
    
    # 資料增強的雜訊強度
    "noise_factor": 0.01
}

# ============================
# 模型架構配置
# ============================
MODEL_CONFIG = {
    # GRU隱藏層大小
    "hidden_size": 128,
    
    # GRU層數
    "num_layers": 2,
    
    # Dropout率（防止過擬合）
    "dropout": 0.3,
    
    # 是否使用雙向GRU
    "bidirectional": True
}

# ============================
# 訓練配置
# ============================
TRAINING_CONFIG = {
    # 訓練週期數
    "epochs": 50,
    
    # 批次大小（根據GPU記憶體調整）
    # RTX A2000 (6GB): 建議16-32
    # 記憶體不足時可降至8或4
    "batch_size": 16,
    
    # 學習率
    "learning_rate": 0.001,
    
    # 序列長度（時間步數）
    "sequence_length": 20,
    
    # 權重衰減（L2正規化）
    "weight_decay": 1e-5,
    
    # 學習率調度器參數
    "scheduler_patience": 10,  # 多少個epoch沒改善就降低學習率
    "scheduler_factor": 0.5,   # 學習率調整係數
    
    # 早停參數（可選）
    "early_stopping_patience": 20,  # 多少個epoch沒改善就停止訓練
    
    # 模型儲存資料夾
    "model_folder": "v1/models"
}

# ============================
# 測試配置
# ============================
TESTING_CONFIG = {
    # 測試批次大小
    "batch_size": 32,
    
    # 序列長度（必須與訓練時相同）
    "sequence_length": 20,
    
    # 結果儲存資料夾
    "results_folder": "v1/results",
    
    # 是否顯示圖表
    "show_plots": True,
    
    # 圖表解析度
    "plot_dpi": 300
}

# ============================
# 硬體優化配置
# ============================
HARDWARE_CONFIG = {
    # GPU設定
    "use_gpu": True,  # 是否使用GPU
    
    # 多執行緒設定
    "num_workers": 4,  # DataLoader的工作執行緒數
    
    # 記憶體管理
    "pin_memory": True,  # 是否將資料固定在記憶體中（GPU加速）
    
    # 精度設定
    "mixed_precision": False,  # 是否使用混合精度訓練（節省記憶體）
}

# ============================
# 視覺化配置
# ============================
VISUALIZATION_CONFIG = {
    # 圖表風格
    "style": "seaborn-v0_8",
    
    # 圖表大小
    "figsize": {
        "confusion_matrix": (12, 10),
        "training_curves": (15, 5),
        "class_performance": (15, 8)
    },
    
    # 顏色設定
    "colormap": "Blues",
    
    # 字體大小
    "font_size": 12
}

# ============================
# 特殊硬體環境預設配置
# ============================

# RTX A2000 (6GB) 優化配置
RTX_A2000_CONFIG = {
    "batch_size": 16,
    "sequence_length": 20,
    "hidden_size": 128,
    "epochs": 50
}

# 低記憶體環境配置 (4GB以下)
LOW_MEMORY_CONFIG = {
    "batch_size": 4,
    "sequence_length": 15,
    "hidden_size": 64,
    "epochs": 30
}

# 高性能環境配置 (8GB以上)
HIGH_PERFORMANCE_CONFIG = {
    "batch_size": 64,
    "sequence_length": 30,
    "hidden_size": 256,
    "epochs": 100
}

# ============================
# 實用函數
# ============================

def get_config_for_hardware(gpu_memory_gb):
    """
    根據GPU記憶體大小自動選擇合適的配置
    
    Args:
        gpu_memory_gb (float): GPU記憶體大小（GB）
    
    Returns:
        dict: 適合的配置參數
    """
    if gpu_memory_gb <= 4:
        print(f"檢測到GPU記憶體: {gpu_memory_gb}GB，使用低記憶體配置")
        return LOW_MEMORY_CONFIG
    elif gpu_memory_gb <= 6:
        print(f"檢測到GPU記憶體: {gpu_memory_gb}GB，使用RTX A2000配置")
        return RTX_A2000_CONFIG
    else:
        print(f"檢測到GPU記憶體: {gpu_memory_gb}GB，使用高性能配置")
        return HIGH_PERFORMANCE_CONFIG

def update_config(base_config, hardware_specific_config):
    """
    更新基礎配置with硬體特定配置
    
    Args:
        base_config (dict): 基礎配置
        hardware_specific_config (dict): 硬體特定配置
    
    Returns:
        dict: 更新後的配置
    """
    updated_config = base_config.copy()
    for key, value in hardware_specific_config.items():
        if key in updated_config:
            updated_config[key] = value
    return updated_config

def print_current_config():
    """打印當前配置摘要"""
    print("=" * 60)
    print("當前配置摘要")
    print("=" * 60)
    print(f"資料集資料夾: {DATA_CONFIG['dataset_folder']}")
    print(f"最小樣本數: {DATA_CONFIG['min_samples_per_class']}")
    print(f"測試集比例: {DATA_CONFIG['test_size']}")
    print("-" * 30)
    print(f"隱藏層大小: {MODEL_CONFIG['hidden_size']}")
    print(f"GRU層數: {MODEL_CONFIG['num_layers']}")
    print(f"Dropout率: {MODEL_CONFIG['dropout']}")
    print("-" * 30)
    print(f"訓練週期: {TRAINING_CONFIG['epochs']}")
    print(f"批次大小: {TRAINING_CONFIG['batch_size']}")
    print(f"學習率: {TRAINING_CONFIG['learning_rate']}")
    print(f"序列長度: {TRAINING_CONFIG['sequence_length']}")
    print("=" * 60)

if __name__ == "__main__":
    # 示範配置使用
    print_current_config()
    
    # 示範硬體自適應配置
    print("\n硬體配置示範:")
    config_6gb = get_config_for_hardware(6.0)
    print("RTX A2000配置:", config_6gb)
    
    config_4gb = get_config_for_hardware(3.5)
    print("低記憶體配置:", config_4gb)
