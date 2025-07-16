"""
簡單的GPU測試腳本 - 專門測試訓練代碼的GPU使用
Simple GPU test for training code
"""

import torch
import numpy as np

def test_gpu_basic():
    """基本GPU測試"""
    print("=" * 50)
    print("🔍 基本GPU測試")
    print("=" * 50)
    
    # 1. 檢查CUDA可用性
    print(f"1. CUDA可用性: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"2. GPU數量: {torch.cuda.device_count()}")
        print(f"3. 當前GPU: {torch.cuda.get_device_name(0)}")
        print(f"4. CUDA版本: {torch.version.cuda}")
        
        # 創建設備對象 (模擬訓練代碼)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"5. 設備對象: {device}")
        
        return True
    else:
        print("❌ CUDA不可用")
        return False

def test_model_gpu():
    """測試模型GPU部署"""
    print("\n🤖 模型GPU測試")
    print("-" * 30)
    
    try:
        # 導入GRU模型 (如果可能)
        try:
            from train_model_v1 import SignLanguageGRU
            print("✅ 成功導入SignLanguageGRU")
        except ImportError as e:
            print(f"❌ 無法導入模型: {e}")
            return False
        
        # 創建設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"目標設備: {device}")
        
        # 創建模型
        model = SignLanguageGRU(
            input_size=163,  # 特徵維度
            hidden_size=128,
            num_layers=2,
            num_classes=5,   # 假設5個類別
            dropout=0.3
        )
        
        print(f"模型創建成功，參數數量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 移動到GPU
        model = model.to(device)
        print(f"✅ 模型已移動到: {next(model.parameters()).device}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型GPU測試失敗: {e}")
        return False

def test_data_gpu():
    """測試資料GPU傳輸"""
    print("\n📊 資料GPU傳輸測試")
    print("-" * 30)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 創建測試資料 (模擬訓練批次)
        batch_size = 4
        sequence_length = 20
        feature_dim = 163
        num_classes = 5
        
        # 創建假資料
        X = torch.randn(batch_size, sequence_length, feature_dim)
        y = torch.randint(0, num_classes, (batch_size,))
        
        print(f"原始資料設備: X={X.device}, y={y.device}")
        
        # 移動到GPU (模擬訓練代碼)
        X_gpu = X.to(device)
        y_gpu = y.to(device)
        
        print(f"GPU資料設備: X={X_gpu.device}, y={y_gpu.device}")
        
        # 檢查是否真的在GPU上
        if torch.cuda.is_available():
            expected_device = f"cuda:{torch.cuda.current_device()}"
            if str(X_gpu.device) == expected_device and str(y_gpu.device) == expected_device:
                print("✅ 資料成功移動到GPU")
                return True
            else:
                print(f"❌ 資料未正確移動到GPU")
                return False
        else:
            print("⚠️  CUDA不可用，資料在CPU上")
            return False
            
    except Exception as e:
        print(f"❌ 資料GPU測試失敗: {e}")
        return False

def test_training_simulation():
    """模擬訓練過程"""
    print("\n🏋️ 訓練過程模擬")
    print("-" * 30)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        
        # 模擬訓練代碼中的設備檢查
        from train_model_v1 import SignLanguageTrainer
        trainer = SignLanguageTrainer()
        
        print(f"Trainer設備: {trainer.device}")
        
        if torch.cuda.is_available():
            # 檢查GPU記憶體
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            print(f"GPU記憶體 - 已分配: {allocated/1024**2:.1f}MB, 已保留: {reserved/1024**2:.1f}MB")
            
            # 簡單的前向傳播測試
            with torch.no_grad():
                test_input = torch.randn(2, 20, 163).to(device)
                print(f"測試輸入設備: {test_input.device}")
                
                # 這裡應該看到GPU記憶體增加
                new_allocated = torch.cuda.memory_allocated()
                print(f"創建測試張量後GPU記憶體: {new_allocated/1024**2:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 訓練模擬失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_pytorch_installation_detailed():
    """詳細檢查PyTorch安裝"""
    print("\n🔥 PyTorch詳細檢查")
    print("-" * 30)
    
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"是否支援CUDA: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA設備數: {torch.cuda.device_count()}")
            print(f"當前設備: {torch.cuda.current_device()}")
            
            # 檢查設備屬性
            props = torch.cuda.get_device_properties(0)
            print(f"設備名稱: {props.name}")
            print(f"計算能力: {props.major}.{props.minor}")
            print(f"總記憶體: {props.total_memory/1024**3:.1f}GB")
            
            return True
        else:
            print("❌ CUDA不可用")
            
            # 提供可能的原因
            print("\n可能的原因:")
            print("1. PyTorch沒有安裝CUDA版本")
            print("2. CUDA驅動不相容")
            print("3. 環境變數問題")
            
            return False
            
    except Exception as e:
        print(f"❌ PyTorch檢查失敗: {e}")
        return False

def main():
    """主測試程序"""
    print("🚀 GPU使用診斷工具")
    print("專門檢查訓練代碼的GPU使用問題")
    
    results = []
    
    # 1. 基本GPU測試
    results.append(("基本GPU", test_gpu_basic()))
    
    # 2. PyTorch詳細檢查
    results.append(("PyTorch", check_pytorch_installation_detailed()))
    
    # 3. 模型GPU測試
    results.append(("模型GPU", test_model_gpu()))
    
    # 4. 資料GPU測試
    results.append(("資料GPU", test_data_gpu()))
    
    # 5. 訓練模擬
    results.append(("訓練模擬", test_training_simulation()))
    
    # 總結
    print("\n" + "=" * 50)
    print("📋 測試結果總結")
    print("=" * 50)
    
    for test_name, passed in results:
        status = "✅ 通過" if passed else "❌ 失敗"
        print(f"{test_name:<10}: {status}")
    
    failed_tests = [name for name, passed in results if not passed]
    
    if not failed_tests:
        print("\n🎉 所有測試通過！GPU應該可以正常使用。")
        print("\n如果訓練代碼仍然顯示使用CPU，請檢查:")
        print("1. 是否在正確的conda環境中")
        print("2. PyTorch是否為CUDA版本")
        print("3. 重新運行訓練代碼並觀察輸出")
    else:
        print(f"\n⚠️  失敗的測試: {', '.join(failed_tests)}")
        print("\n💡 建議解決步驟:")
        print("1. 運行完整診斷: python v1/debug_gpu.py")
        print("2. 重新安裝PyTorch CUDA版本:")
        print("   conda uninstall pytorch torchvision torchaudio")
        print("   conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia")
        print("3. 確保在sign_language環境中運行")

if __name__ == "__main__":
    main()
