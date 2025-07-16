# GPU問題診斷和解決指南

## 🚨 你的代碼沒有使用GPU的可能原因

### 📋 診斷步驟

#### 第一步：運行診斷腳本
```bash
# 在sign_language環境中運行
conda activate sign_language
cd d:\file\畢業專題\tran2

# 完整診斷
python v1/debug_gpu.py

# 簡單測試
python v1/test_gpu_simple.py
```

#### 第二步：檢查基本輸出
運行訓練代碼時應該看到：
```
使用設備: cuda:0  # ✅ 正確
使用設備: cpu     # ❌ 問題！
```

## 🔍 常見問題和解決方案

### 問題1：PyTorch沒有CUDA支援
**症狀**: `torch.cuda.is_available()` 返回 `False`

**檢查命令**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
```

**解決方案**:
```bash
# 確保在sign_language環境中
conda activate sign_language

# 完全移除現有PyTorch
conda uninstall pytorch torchvision torchaudio

# 重新安裝CUDA版本
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# 或者嘗試不同的CUDA版本
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 問題2：NVIDIA驅動問題
**檢查命令**:
```bash
nvidia-smi
```

**如果失敗**:
1. 更新NVIDIA驅動到最新版本
2. 重啟電腦
3. 檢查設備管理員中GPU狀態

### 問題3：環境混亂
**症狀**: 在錯誤的Python環境中

**檢查命令**:
```bash
# 檢查當前環境
echo $CONDA_DEFAULT_ENV
# 或在PowerShell中
$env:CONDA_DEFAULT_ENV

# 檢查Python路徑
python -c "import sys; print(sys.executable)"
```

**解決方案**:
```bash
# 確保在正確環境中
conda activate sign_language

# 檢查環境中的套件
conda list pytorch
```

### 問題4：CUDA版本不匹配
**檢查CUDA版本**:
```bash
nvidia-smi  # 查看CUDA Driver Version
```

**匹配PyTorch版本**:
- CUDA 11.7/11.8: `pytorch-cuda=11.7` 或 `pytorch-cuda=11.8`
- CUDA 12.x: `pytorch-cuda=12.1`

### 問題5：記憶體不足
**症狀**: 代碼運行但很慢，或者出現記憶體錯誤

**解決方案**:
```python
# 在config_v1.py中調整
TRAINING_CONFIG["batch_size"] = 8  # 減少批次大小
TRAINING_CONFIG["sequence_length"] = 15  # 減少序列長度
```

## 🧪 快速測試代碼

創建 `test_quick.py`:
```python
import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU名稱: {torch.cuda.get_device_name(0)}")
    
    # 測試設備創建（模擬你的代碼）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"設備對象: {device}")
    
    # 測試張量創建
    x = torch.randn(3, 3, device=device)
    print(f"張量設備: {x.device}")
    
    print("✅ GPU測試成功！")
else:
    print("❌ GPU不可用")
```

運行測試:
```bash
python test_quick.py
```

## 🔧 強制重新安裝解決方案

如果其他方法都失敗，嘗試完全重新安裝：

```bash
# 1. 退出環境
conda deactivate

# 2. 刪除環境
conda env remove -n sign_language

# 3. 重新創建環境
conda create -n sign_language python=3.9 -y

# 4. 啟用環境
conda activate sign_language

# 5. 接受服務條款（如果需要）
conda tos accept --all

# 6. 安裝PyTorch CUDA版本
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

# 7. 安裝其他套件
conda install numpy pandas matplotlib seaborn jupyter scikit-learn -y
pip install joblib

# 8. 測試
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 📞 獲取幫助

如果問題仍然存在，請提供以下資訊：

1. **診斷腳本輸出**:
   ```bash
   python v1/debug_gpu.py > gpu_debug.log 2>&1
   ```

2. **環境資訊**:
   ```bash
   conda list > conda_packages.txt
   nvidia-smi > nvidia_info.txt
   ```

3. **錯誤訊息**: 完整的錯誤輸出

4. **系統資訊**: Windows版本、GPU型號

## 🎯 最可能的解決方案

基於經驗，最常見的原因是：

1. **PyTorch沒有CUDA支援** (70%的情況)
2. **不在正確的conda環境中** (20%的情況)
3. **NVIDIA驅動問題** (10%的情況)

**快速修復**:
```bash
conda activate sign_language
conda uninstall pytorch torchvision torchaudio -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

如果這樣還不行，運行完整診斷腳本找出具體問題！
