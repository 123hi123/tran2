"""
驗證PyTorch CUDA修復結果
"""
import torch

print("🔍 PyTorch CUDA修復驗證")
print("=" * 40)

try:
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"CUDA可用性: {'✅ 可用' if torch.cuda.is_available() else '❌ 不可用'}")
    
    if torch.cuda.is_available():
        print(f"GPU數量: {torch.cuda.device_count()}")
        print(f"GPU名稱: {torch.cuda.get_device_name(0)}")
        print(f"GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 測試GPU操作
        print("\n🧪 GPU功能測試:")
        device = torch.device('cuda')
        x = torch.randn(3, 3, device=device)
        y = torch.randn(3, 3, device=device)
        z = torch.mm(x, y)
        print(f"✅ GPU矩陣運算成功: 結果在 {z.device}")
        
        print("\n🎉 恭喜！GPU已經可以正常使用了！")
        print("現在可以運行訓練代碼，應該會顯示: 使用設備: cuda:0")
        
    else:
        print("\n❌ 仍然無法使用GPU")
        print("可能需要:")
        print("1. 檢查NVIDIA驅動")
        print("2. 嘗試不同的CUDA版本")
        print("3. 完全重新安裝環境")
        
except Exception as e:
    print(f"❌ 測試失敗: {e}")
    
print("=" * 40)
