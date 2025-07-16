# 系統硬件與軟件環境調查報告

## 系統基本資訊
- **作業系統**: Windows 10 Enterprise LTSC 2021 (Build 19044)
- **系統製造商**: ASUSTeK COMPUTER INC.
- **系統型號**: ASUSPRO D840MB_M840MB
- **系統類型**: x64-based PC

## CPU 資訊
- **處理器**: Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz
- **核心數**: 8 核心
- **邏輯處理器**: 8 (無超執行緒)
- **最大時脈**: 3001 MHz

## 記憶體 (RAM) 資訊
- **總記憶體**: 48 GB (51,335,266,304 bytes)
- **記憶體配置**:
  - 8 GB × 2 條 (DDR4-2666)
  - 16 GB × 2 條 (DDR4-2666)
- **記憶體頻率**: 2666 MHz
- **記憶體類型**: DDR4

## GPU 資訊 (重要！用於深度學習)
- **主要GPU**: NVIDIA RTX A2000
  - **顯存**: 6 GB GDDR6
  - **驅動版本**: 471.41
  - **CUDA版本**: 11.4
  - **目前記憶體使用**: 135MiB / 6138MiB
  - **GPU使用率**: 0% (待機狀態)
  - **溫度**: 43°C
  - **功耗**: 4W / 70W

- **整合顯卡**: Intel(R) UHD Graphics 630 (1GB)

## 儲存空間資訊
- **主要磁碟**: C: 磁碟機
  - **總容量**: 447 GB (479,384,608,768 bytes)
  - **可用空間**: 327 GB (350,486,941,696 bytes)
  - **檔案系統**: NTFS

## 軟件環境資訊
- **Python版本**: 
  - 系統Python: 3.11.0
  - Conda Base: 3.13.5
- **Conda版本**: conda 25.5.1
- **Conda路徑**: C:\Users\user\anaconda3\Scripts\conda.exe
- **已安裝套件**: numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, scikit-image 0.25.0
- **缺少套件**: PyTorch, TensorFlow, OpenCV, MediaPipe
- **CUDA狀態**: 檢測到CUDA 11.4，但nvcc編譯器不可用

## 環境評估結果

### ✅ **可用資源**
- Conda環境管理器正常運作
- 基礎科學計算套件已安裝 (numpy, pandas, scikit-learn)
- GPU硬體支援CUDA 11.4
- 充足的系統資源 (48GB RAM, 6GB VRAM)

### ⚠️ **需要安裝的核心套件**
- PyTorch (深度學習框架)
- OpenCV (電腦視覺)
- MediaPipe (手部追蹤)
- Jupyter (開發環境)
- matplotlib, seaborn (資料視覺化)

### 🔧 **需要解決的問題**
- 建立專用的Python 3.9環境 (確保相容性)
- 安裝PyTorch with CUDA支援
- 配置CUDA工具包 (如果需要編譯)

## 立即執行建議

### 第一步：建立手語辨識專用環境
```powershell
# 建立Python 3.9環境 (最佳相容性)
conda create -n sign_language python=3.9 -y

# 啟用環境
conda activate sign_language
```

### 第二步：安裝核心深度學習套件
```powershell
# 安裝PyTorch (適用於CUDA 11.4)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

# 安裝基礎科學計算套件
conda install numpy pandas matplotlib seaborn jupyter scikit-learn -y

# 安裝電腦視覺套件
conda install opencv -c conda-forge -y
pip install mediapipe

# 安裝其他必要套件
pip install tqdm pillow
```

### 5. 資料集資訊調查
```powershell
# 檢查資料集檔案數量和大小
Get-ChildItem -Path ".\dataset\" -Name "*.csv" | Measure-Object
Get-ChildItem -Path ".\dataset\" -Name "*.csv" | ForEach-Object { Get-Item ".\dataset\$_" | Select-Object Name, @{Name="Size(KB)";Expression={[math]::Round($_.Length/1KB,2)}} }

# 檢查資料集內容格式 (以第一個檔案為例)
Get-Content ".\dataset\sign_language1.csv" -Head 5
```

### 6. 項目資料夾結構建議
建議在桌面建立以下資料夾結構：
```
手語辨識專案/
├── dataset/          # 資料集檔案 (已存在)
├── models/           # 訓練好的模型檔案
├── notebooks/        # Jupyter notebooks
├── src/              # 原始碼
│   ├── data/         # 數據預處理
│   ├── models/       # 模型定義
│   ├── training/     # 訓練腳本
│   └── inference/    # 推理腳本
├── experiments/      # 實驗記錄和結果
├── logs/            # 訓練日誌
└── requirements.txt  # 套件需求列表
```



git clone https://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@github.com/123hi123/tran2.git