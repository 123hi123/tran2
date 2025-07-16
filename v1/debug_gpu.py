"""
GPU診斷工具 - 查找GPU無法使用的原因
This script will help identify why GPU is not being used
"""

import sys
import os

def check_basic_python_info():
    """檢查基本Python環境"""
    print("🐍 Python環境檢查")
    print("-" * 40)
    print(f"Python版本: {sys.version}")
    print(f"Python執行檔路徑: {sys.executable}")
    print(f"當前工作目錄: {os.getcwd()}")
    print()

def check_pytorch_installation():
    """檢查PyTorch安裝狀況"""
    print("🔥 PyTorch安裝檢查")
    print("-" * 40)
    
    try:
        import torch
        print(f"✅ PyTorch已安裝: 版本 {torch.__version__}")
        
        # 檢查CUDA版本
        cuda_version = torch.version.cuda
        print(f"PyTorch CUDA版本: {cuda_version if cuda_version else '❌ 無CUDA支援'}")
        
        # 檢查CUDA是否可用
        cuda_available = torch.cuda.is_available()
        print(f"CUDA可用性: {'✅ 可用' if cuda_available else '❌ 不可用'}")
        
        if cuda_available:
            print(f"CUDA設備數量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  設備 {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    記憶體: {props.total_memory / 1024**3:.1f} GB")
                print(f"    計算能力: {props.major}.{props.minor}")
        else:
            print("❌ 無可用的CUDA設備")
            
        return cuda_available
        
    except ImportError as e:
        print(f"❌ PyTorch未安裝: {e}")
        return False
    except Exception as e:
        print(f"❌ PyTorch檢查失敗: {e}")
        return False

def check_nvidia_driver():
    """檢查NVIDIA驅動"""
    print("\n🎯 NVIDIA驅動檢查")
    print("-" * 40)
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA驅動正常")
            print("nvidia-smi輸出:")
            print(result.stdout)
            return True
        else:
            print("❌ nvidia-smi執行失敗")
            print(f"錯誤: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi執行超時")
        return False
    except FileNotFoundError:
        print("❌ 找不到nvidia-smi命令")
        print("可能原因: NVIDIA驅動未安裝或未加入PATH")
        return False
    except Exception as e:
        print(f"❌ 檢查NVIDIA驅動時發生錯誤: {e}")
        return False

def test_simple_gpu_operation():
    """測試簡單的GPU操作"""
    print("\n⚡ GPU操作測試")
    print("-" * 40)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，跳過GPU測試")
            return False
            
        # 測試基本GPU操作
        print("測試1: 創建GPU張量")
        device = torch.device('cuda:0')
        x = torch.randn(3, 3, device=device)
        print(f"✅ GPU張量創建成功: {x.device}")
        
        print("測試2: GPU矩陣運算")
        y = torch.randn(3, 3, device=device)
        z = torch.mm(x, y)
        print(f"✅ GPU運算成功: 結果在 {z.device}")
        
        print("測試3: CPU-GPU資料傳輸")
        cpu_tensor = torch.randn(3, 3)
        gpu_tensor = cpu_tensor.to(device)
        cpu_back = gpu_tensor.cpu()
        print("✅ CPU-GPU資料傳輸成功")
        
        print("測試4: GPU記憶體狀態")
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        print(f"✅ 已分配記憶體: {allocated / 1024**2:.1f} MB")
        print(f"✅ 已保留記憶體: {reserved / 1024**2:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU操作測試失敗: {e}")
        return False

def check_conda_environment():
    """檢查Conda環境"""
    print("\n🐍 Conda環境檢查")
    print("-" * 40)
    
    # 檢查是否在Conda環境中
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"✅ 當前Conda環境: {conda_env}")
    else:
        print("❌ 不在Conda環境中")
        return False
    
    # 檢查環境路徑
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        print(f"環境路徑: {conda_prefix}")
    
    # 檢查Python路徑是否指向Conda環境
    python_path = sys.executable
    if conda_prefix and conda_prefix in python_path:
        print("✅ Python路徑正確指向Conda環境")
        return True
    else:
        print("❌ Python路徑可能不正確")
        print(f"期望包含: {conda_prefix}")
        print(f"實際路徑: {python_path}")
        return False

def check_pytorch_cuda_installation():
    """詳細檢查PyTorch CUDA安裝"""
    print("\n🔍 PyTorch CUDA詳細檢查")
    print("-" * 40)
    
    try:
        import torch
        
        # 檢查編譯資訊
        print("PyTorch編譯資訊:")
        print(f"  版本: {torch.__version__}")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"  是否使用CUDA編譯: {torch.version.cuda is not None}")
        
        # 檢查後端
        print("\nCUDA後端狀態:")
        print(f"  cuDNN可用: {torch.backends.cudnn.enabled}")
        print(f"  cuDNN基準模式: {torch.backends.cudnn.benchmark}")
        
        # 嘗試列出所有可用設備
        print("\n可用設備:")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                print(f"  cuda:{i} - {device_name}")
        else:
            print("  只有CPU可用")
            
        return torch.cuda.is_available()
        
    except Exception as e:
        print(f"❌ 檢查失敗: {e}")
        return False

def provide_solutions():
    """提供解決方案"""
    print("\n💡 常見問題解決方案")
    print("=" * 50)
    
    print("\n問題1: PyTorch沒有CUDA支援")
    print("解決方案:")
    print("  conda uninstall pytorch torchvision torchaudio")
    print("  conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia")
    
    print("\n問題2: NVIDIA驅動問題")
    print("解決方案:")
    print("  1. 更新NVIDIA驅動到最新版本")
    print("  2. 重啟電腦")
    print("  3. 檢查設備管理員中GPU狀態")
    
    print("\n問題3: CUDA版本不匹配")
    print("解決方案:")
    print("  1. 檢查 nvidia-smi 顯示的CUDA版本")
    print("  2. 安裝匹配的PyTorch版本")
    print("  3. 訪問 https://pytorch.org/ 獲取正確安裝命令")
    
    print("\n問題4: 環境變數問題")
    print("解決方案:")
    print("  1. 確保在正確的Conda環境中")
    print("  2. conda activate sign_language")
    print("  3. 重新安裝PyTorch")
    
    print("\n問題5: 權限問題")
    print("解決方案:")
    print("  1. 以管理員身份運行PowerShell")
    print("  2. 檢查防毒軟體是否阻擋")

def main():
    """主診斷程序"""
    print("🚨 GPU診斷工具 - 查找GPU無法使用的原因")
    print("=" * 60)
    
    # 收集所有檢查結果
    results = {}
    
    # 1. 基本環境檢查
    check_basic_python_info()
    
    # 2. Conda環境檢查
    results['conda'] = check_conda_environment()
    
    # 3. PyTorch安裝檢查
    results['pytorch'] = check_pytorch_installation()
    
    # 4. NVIDIA驅動檢查
    results['nvidia'] = check_nvidia_driver()
    
    # 5. PyTorch CUDA詳細檢查
    results['pytorch_cuda'] = check_pytorch_cuda_installation()
    
    # 6. GPU操作測試
    results['gpu_ops'] = test_simple_gpu_operation()
    
    # 總結診斷結果
    print("\n📋 診斷結果總結")
    print("=" * 60)
    
    all_passed = True
    for check_name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{check_name:15}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有檢查都通過！GPU應該可以正常使用。")
        print("如果代碼仍然使用CPU，請檢查代碼中的設備設定。")
    else:
        print("\n⚠️  發現問題，請查看上述失敗的檢查項目。")
        provide_solutions()
    
    # 快速測試代碼
    print("\n🧪 快速測試代碼:")
    print("import torch")
    print("print(f'CUDA可用: {torch.cuda.is_available()}')")
    print("if torch.cuda.is_available():")
    print("    print(f'GPU名稱: {torch.cuda.get_device_name(0)}')")
    print("    x = torch.randn(3, 3, device='cuda')")
    print("    print(f'GPU張量: {x.device}')")

if __name__ == "__main__":
    main()
