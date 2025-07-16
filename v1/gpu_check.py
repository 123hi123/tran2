"""
GPU使用情況檢測和監控工具
用於驗證和監控手語辨識系統的GPU加速情況
"""

import torch
import numpy as np
import time
import psutil
import os

def check_gpu_availability():
    """檢查GPU可用性和詳細資訊"""
    print("=" * 60)
    print("🔍 GPU環境檢查")
    print("=" * 60)
    
    # 檢查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用性: {'✅ 可用' if cuda_available else '❌ 不可用'}")
    
    if cuda_available:
        # GPU數量
        gpu_count = torch.cuda.device_count()
        print(f"GPU數量: {gpu_count}")
        
        # 當前GPU資訊
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"當前GPU: {gpu_name}")
        
        # CUDA版本
        cuda_version = torch.version.cuda
        print(f"CUDA版本: {cuda_version}")
        
        # PyTorch版本
        pytorch_version = torch.__version__
        print(f"PyTorch版本: {pytorch_version}")
        
        # GPU記憶體資訊
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        print(f"GPU總記憶體: {total_memory / 1024**3:.2f} GB")
        
        # 當前記憶體使用
        allocated_memory = torch.cuda.memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()
        print(f"已分配記憶體: {allocated_memory / 1024**3:.2f} GB")
        print(f"已保留記憶體: {reserved_memory / 1024**3:.2f} GB")
        print(f"可用記憶體: {(total_memory - reserved_memory) / 1024**3:.2f} GB")
        
    else:
        print("❌ GPU不可用，將使用CPU進行計算")
        print("請檢查：")
        print("1. NVIDIA驅動是否正確安裝")
        print("2. CUDA工具包是否安裝")
        print("3. PyTorch是否安裝CUDA版本")
    
    print("=" * 60)
    return cuda_available

def benchmark_gpu_vs_cpu(batch_size=16, sequence_length=20, feature_dim=163):
    """比較GPU和CPU的性能差異"""
    print("\n⚡ GPU vs CPU 性能測試")
    print("-" * 40)
    
    # 創建測試資料
    test_data = torch.randn(batch_size, sequence_length, feature_dim)
    
    # 創建簡單的測試模型
    from train_model_v1 import SignLanguageGRU
    model = SignLanguageGRU(
        input_size=feature_dim,
        hidden_size=128,
        num_layers=2,
        num_classes=10
    )
    
    # CPU測試
    print("測試CPU性能...")
    model_cpu = model.to('cpu')
    data_cpu = test_data.to('cpu')
    
    start_time = time.time()
    for _ in range(10):  # 運行10次取平均
        with torch.no_grad():
            _ = model_cpu(data_cpu)
    cpu_time = (time.time() - start_time) / 10
    print(f"CPU平均推理時間: {cpu_time:.4f}秒")
    
    # GPU測試（如果可用）
    if torch.cuda.is_available():
        print("測試GPU性能...")
        model_gpu = model.to('cuda')
        data_gpu = test_data.to('cuda')
        
        # 預熱GPU
        for _ in range(5):
            with torch.no_grad():
                _ = model_gpu(data_gpu)
        torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = model_gpu(data_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start_time) / 10
        
        print(f"GPU平均推理時間: {gpu_time:.4f}秒")
        speedup = cpu_time / gpu_time
        print(f"GPU加速倍數: {speedup:.2f}x")
        
        return speedup
    else:
        print("GPU不可用，無法進行比較")
        return 1.0

def monitor_gpu_during_training():
    """訓練過程中的GPU監控裝飾器"""
    def decorator(train_func):
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                print("\n📊 開始GPU監控...")
                initial_memory = torch.cuda.memory_allocated()
                max_memory = initial_memory
                
                try:
                    result = train_func(*args, **kwargs)
                    
                    # 訓練後的記憶體統計
                    final_memory = torch.cuda.memory_allocated()
                    peak_memory = torch.cuda.max_memory_allocated()
                    
                    print("\n📈 GPU記憶體使用統計:")
                    print(f"初始記憶體: {initial_memory / 1024**3:.2f} GB")
                    print(f"最終記憶體: {final_memory / 1024**3:.2f} GB")
                    print(f"峰值記憶體: {peak_memory / 1024**3:.2f} GB")
                    print(f"記憶體增長: {(final_memory - initial_memory) / 1024**3:.2f} GB")
                    
                    return result
                    
                except Exception as e:
                    print(f"❌ 訓練過程中發生錯誤: {e}")
                    if "out of memory" in str(e).lower():
                        print("💡 建議：減少batch_size或sequence_length")
                    raise e
            else:
                return train_func(*args, **kwargs)
        return wrapper
    return decorator

def estimate_memory_usage(batch_size, sequence_length, feature_dim, hidden_size=128, num_layers=2):
    """估算模型記憶體使用量"""
    print(f"\n🧮 記憶體使用估算 (batch_size={batch_size}, seq_len={sequence_length})")
    print("-" * 50)
    
    # 模型參數記憶體
    # GRU參數數量估算
    input_size = feature_dim
    gru_params = (input_size + hidden_size + 1) * hidden_size * 6 * num_layers  # 雙向GRU
    fc_params = hidden_size * 2 * hidden_size + hidden_size * 10  # 全連接層
    total_params = gru_params + fc_params
    
    model_memory = total_params * 4 / 1024**3  # 4 bytes per float32
    print(f"模型參數記憶體: {model_memory:.3f} GB")
    
    # 輸入資料記憶體
    input_memory = batch_size * sequence_length * feature_dim * 4 / 1024**3
    print(f"輸入資料記憶體: {input_memory:.3f} GB")
    
    # 中間激活記憶體（粗略估算）
    activation_memory = batch_size * sequence_length * hidden_size * num_layers * 8 / 1024**3  # 雙向
    print(f"激活記憶體: {activation_memory:.3f} GB")
    
    # 梯度記憶體（訓練時）
    gradient_memory = model_memory  # 梯度與參數同大小
    print(f"梯度記憶體: {gradient_memory:.3f} GB")
    
    total_memory = model_memory + input_memory + activation_memory + gradient_memory
    print(f"總計記憶體: {total_memory:.3f} GB")
    
    # RTX A2000建議
    if total_memory > 5.0:
        print("⚠️  記憶體使用可能超過RTX A2000限制(6GB)")
        print("💡 建議調整參數:")
        suggested_batch = max(1, int(batch_size * 4.0 / total_memory))
        print(f"   - 減少batch_size到: {suggested_batch}")
        print(f"   - 或減少sequence_length到: {int(sequence_length * 0.8)}")
    else:
        print("✅ 記憶體使用在RTX A2000可接受範圍內")
    
    return total_memory

def check_system_info():
    """檢查系統硬體資訊"""
    print("\n💻 系統硬體資訊")
    print("-" * 30)
    
    # CPU資訊
    print(f"CPU核心數: {psutil.cpu_count(logical=False)}")
    print(f"CPU邏輯核心: {psutil.cpu_count(logical=True)}")
    
    # 記憶體資訊
    memory = psutil.virtual_memory()
    print(f"系統記憶體: {memory.total / 1024**3:.1f} GB")
    print(f"可用記憶體: {memory.available / 1024**3:.1f} GB")
    
    # 檢查是否有GPU
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  記憶體: {props.total_memory / 1024**3:.1f} GB")
            print(f"  計算能力: {props.major}.{props.minor}")

def main():
    """主檢測程序"""
    print("🚀 手語辨識系統GPU檢測工具")
    
    # 1. 檢查系統資訊
    check_system_info()
    
    # 2. 檢查GPU可用性
    gpu_available = check_gpu_availability()
    
    # 3. 記憶體使用估算
    estimate_memory_usage(
        batch_size=16,
        sequence_length=20,
        feature_dim=163
    )
    
    # 4. 性能測試
    if gpu_available:
        speedup = benchmark_gpu_vs_cpu()
        
        print(f"\n✅ GPU檢測完成!")
        print(f"   - GPU加速: {speedup:.1f}倍")
        print(f"   - 建議使用GPU進行訓練")
    else:
        print(f"\n⚠️  GPU不可用，將使用CPU")
        print(f"   - 訓練速度會較慢")
        print(f"   - 建議檢查CUDA安裝")

if __name__ == "__main__":
    main()
