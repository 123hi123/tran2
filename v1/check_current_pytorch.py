"""
檢查當前PyTorch安裝狀況
"""
import torch
import sys
import os

print("=" * 60)
print("🔍 當前PyTorch安裝檢查")
print("=" * 60)

# 檢查環境
print(f"當前conda環境: {os.environ.get('CONDA_DEFAULT_ENV', '未知')}")
print(f"Python路徑: {sys.executable}")
print(f"PyTorch版本: {torch.__version__}")
print(f"PyTorch CUDA版本: {torch.version.cuda}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

# 檢查安裝來源
print(f"\nPyTorch安裝資訊:")
try:
    print(f"安裝路徑: {torch.__file__}")
    # 檢查是否為CPU版本
    if 'cpu' in torch.__version__:
        print("⚠️  這是CPU版本的PyTorch!")
    elif torch.version.cuda is None:
        print("⚠️  這個PyTorch版本沒有CUDA支援!")
    else:
        print(f"✅ 這是CUDA版本: {torch.version.cuda}")
except:
    pass

print("\n" + "=" * 60)
print("💡 下一步建議:")
if not torch.cuda.is_available():
    print("需要重新安裝支援CUDA的PyTorch版本")
    print("請按照以下步驟操作...")
