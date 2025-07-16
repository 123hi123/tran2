# 在學校電腦上安全使用私人GitHub倉庫的方法

## 🔑 方法一：Personal Access Token (推薦)

### 步驟1：創建Personal Access Token

1. 登錄您的GitHub帳號
2. 前往：`Settings` → `Developer settings` → `Personal access tokens` → `Tokens (classic)`
3. 點擊 `Generate new token`
4. 設定名稱：`Sign Language Project - School Computer`
5. 選擇權限：
   ```
   ✅ repo (Full control of private repositories)
   ✅ workflow (如果使用GitHub Actions)
   ```
6. 設定過期時間：建議90天（專案完成時間）
7. 複製生成的token（只會顯示一次！）

### 步驟2：在學校電腦上使用

```bash
# 1. 克隆倉庫（使用token）
git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 實際範例（假設）：
git clone https://ghp_xxxxxxxxxxxxxxxxxxxx@github.com/your_username/tran2.git

# 2. 進入專案目錄
cd tran2

# 3. 設置本地git配置（可選，避免每次輸入）
git config credential.helper store

# 4. 第一次推送會要求輸入憑證
# Username: 您的GitHub用戶名
# Password: 您的Personal Access Token (不是密碼!)
```

### 步驟3：日常使用

```bash
# 拉取更新
git pull

# 推送變更
git add .
git commit -m "Training progress update"
git push

# 如果設置了credential.helper store，第二次以後不需要再輸入token
```

### Token管理建議

```bash
# 為了安全，可以設置環境變量
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"

# 使用環境變量克隆
git clone https://$GITHUB_TOKEN@github.com/your_username/tran2.git

# 或者創建一個簡單的腳本
echo 'export GITHUB_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

## 🚀 方法二：SSH Key (次推薦)

如果您不介意在學校電腦上生成SSH key：

### 步驟1：生成SSH Key

```bash
# 在學校電腦上生成新的SSH key
ssh-keygen -t ed25519 -C "school_computer_sign_language_project"

# 按Enter使用默認文件位置，設置密碼（可選）
```

### 步驟2：添加到GitHub

```bash
# 複製公鑰內容
cat ~/.ssh/id_ed25519.pub

# 將輸出的內容複製到GitHub：
# Settings → SSH and GPG keys → New SSH key
```

### 步驟3：使用SSH克隆

```bash
# 使用SSH URL克隆
git clone git@github.com:your_username/tran2.git
```

## 📱 方法三：GitHub CLI (最方便)

```bash
# 安裝GitHub CLI (如果可用)
# Windows
winget install --id GitHub.cli

# 或下載便攜版本到USB

# 使用設備碼登錄（不需要瀏覽器登錄）
gh auth login --web

# 克隆私人倉庫
gh repo clone your_username/tran2
```

## 🔐 方法四：臨時解決方案

如果只是短期使用：

```bash
# 1. 將倉庫臨時設為公開
# 2. 在學校電腦上克隆
git clone https://github.com/your_username/tran2.git

# 3. 完成後再設回私人
# 4. 使用Personal Access Token推送
```

## 💡 最佳實踐建議

### 安全考量
```bash
# 1. 使用專案結束後立即刪除token
# 2. 不要將token保存在代碼中
# 3. 設置適當的過期時間

# 清理腳本（專案完成後）
cat > cleanup.sh << 'EOF'
#!/bin/bash
# 清理Git憑證
git config --global --unset credential.helper
rm -f ~/.git-credentials
# 清理SSH key (如果使用)
rm -f ~/.ssh/id_ed25519*
# 清理環境變量
unset GITHUB_TOKEN
EOF
```

### 工作流程建議
```bash
# 每日工作流程
echo "#!/bin/bash
# 每日開始工作
cd ~/Desktop/tran2
git pull origin main

# 設置token（如果需要）
export GITHUB_TOKEN='your_token'

echo '準備就緒！開始訓練模型...'
" > daily_start.sh

chmod +x daily_start.sh
```

### 團隊協作設置
```bash
# 如果需要團隊協作
# 在GitHub倉庫設置中添加協作者：
# Settings → Manage access → Invite a collaborator

# 協作者也可以使用同樣的token方法
```

## 📋 快速設置指南

### 一步到位腳本

```bash
#!/bin/bash
# 快速設置腳本 - 在學校電腦上運行

echo "手語辨識專案 - GitHub設置助手"
echo "================================"

read -p "請輸入您的GitHub用戶名: " username
read -p "請輸入您的Personal Access Token: " token
read -p "請輸入倉庫名稱 (預設: tran2): " repo_name

repo_name=${repo_name:-tran2}

# 克隆倉庫
echo "正在克隆倉庫..."
git clone https://$token@github.com/$username/$repo_name.git

cd $repo_name

# 設置憑證存儲
git config credential.helper store

# 設置用戶信息
git config user.name "$username"
git config user.email "$username@users.noreply.github.com"

echo "設置完成！"
echo "現在您可以正常使用 git pull 和 git push"
```

## 🎯 推薦流程總結

1. **創建Personal Access Token** (30秒)
2. **在學校電腦運行設置腳本** (1分鐘)  
3. **開始正常的git工作流程** ✅

這樣您就可以在不登錄GitHub的情況下，安全地訪問和推送到私人倉庫了！
