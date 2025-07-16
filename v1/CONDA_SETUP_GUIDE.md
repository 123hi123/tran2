# Conda環境設置指南 - 解決新版conda問題

## 問題描述
新版本的conda（2024年後）引入了服務條款確認機制，需要用戶明確接受條款才能使用某些頻道。

## 完整解決步驟

### 步驟1：初始化conda
```powershell
# 如果出現 "Run 'conda init' before 'conda activate'" 錯誤
conda init

# ⚠️ 重要：執行init後必須重啟PowerShell
# 關閉當前PowerShell窗口，重新開啟
```

### 步驟2：接受服務條款
```powershell
# 方法1：接受所有預設頻道的條款（推薦）
conda tos accept --all

# 方法2：分別接受各頻道條款
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

### 步驟3：驗證conda正常工作
```powershell
# 檢查conda版本和配置
conda --version
conda info

# 查看可用環境
conda info --envs
```

### 步驟4：創建手語辨識環境
```powershell
# 創建Python 3.9環境
conda create -n sign_language python=3.9 -y

# 啟用環境
conda activate sign_language

# 驗證環境啟用成功
python --version
```

### 步驟5：安裝深度學習套件
```powershell
# 確保在sign_language環境中
conda activate sign_language

# 安裝PyTorch (適用於RTX A2000/CUDA 11.4)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

# 安裝基礎科學計算套件
conda install numpy pandas matplotlib seaborn jupyter scikit-learn -y

# 安裝其他必要套件
pip install joblib
```

## 常見錯誤和解決方案

### 錯誤1：CondaError: Run 'conda init' before 'conda activate'
```powershell
# 解決方案：
conda init
# 然後重啟PowerShell
```

### 錯誤2：Terms of Service have not been accepted
```powershell
# 解決方案：
conda tos accept --all
```

### 錯誤3：conda activate不起作用
```powershell
# 檢查conda是否正確初始化
conda info

# 檢查PowerShell執行策略
Get-ExecutionPolicy

# 如果受限，設置為RemoteSigned
Set-ExecutionPolicy RemoteSigned -CurrentUser
```

### 錯誤4：環境創建失敗
```powershell
# 清理conda緩存
conda clean --all

# 重新嘗試創建環境
conda create -n sign_language python=3.9 -y
```

## 驗證安裝成功

運行以下命令驗證環境正確設置：

```powershell
# 啟用環境
conda activate sign_language

# 檢查Python版本
python --version
# 應顯示：Python 3.9.x

# 檢查關鍵套件
python -c "import torch; print('PyTorch版本:', torch.__version__)"
python -c "import pandas; print('Pandas版本:', pandas.__version__)"
python -c "import numpy; print('NumPy版本:', numpy.__version__)"

# 檢查CUDA可用性
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
```

如果所有命令都成功執行，則環境設置完成，可以開始運行手語辨識系統。

## 額外提示

1. **第一次使用conda**：務必在執行 `conda init` 後重啟PowerShell
2. **企業網路環境**：可能需要配置代理設置
3. **權限問題**：確保有管理員權限或適當的用戶權限
4. **磁盤空間**：確保有足夠空間安裝環境（建議至少5GB可用空間）

---
💡 **重要提醒**：每次使用時都需要先啟用環境 `conda activate sign_language`
