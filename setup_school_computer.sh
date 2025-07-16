#!/bin/bash
# 手語辨識專案 - GitHub私人倉庫快速設置腳本
# 在學校電腦上運行此腳本

echo "🚀 手語辨識專案 - GitHub設置助手"
echo "================================"

# 獲取用戶輸入
read -p "📝 請輸入您的GitHub用戶名: " username
read -p "🔑 請輸入您的Personal Access Token: " token
read -p "📁 請輸入倉庫名稱 (預設: tran2): " repo_name

# 設置預設值
repo_name=${repo_name:-tran2}

echo ""
echo "📊 設置信息："
echo "   用戶名: $username"
echo "   倉庫名: $repo_name"
echo "   Token: ${token:0:8}..." # 只顯示前8位
echo ""

# 創建專案目錄
project_dir="$HOME/Desktop/sign_language_project"
mkdir -p "$project_dir"
cd "$project_dir"

echo "📦 正在克隆倉庫..."
git clone https://$token@github.com/$username/$repo_name.git

if [ $? -eq 0 ]; then
    echo "✅ 倉庫克隆成功！"
    
    cd $repo_name
    
    # 設置Git配置
    echo "⚙️ 配置Git設置..."
    git config credential.helper store
    git config user.name "$username"
    git config user.email "$username@users.noreply.github.com"
    
    # 創建便捷腳本
    echo "📝 創建便捷腳本..."
    
    # 每日啟動腳本
    cat > daily_start.sh << EOF
#!/bin/bash
# 每日訓練開始腳本

echo "🚀 手語辨識專案 - 每日啟動"
echo "========================="

# 切換到專案目錄
cd "$project_dir/$repo_name"

# 拉取最新代碼
echo "📦 拉取最新代碼..."
git pull

# 啟動conda環境
echo "🐍 激活conda環境..."
conda activate sign_language

# 顯示專案狀態
echo "📊 專案狀態："
echo "   當前目錄: \$(pwd)"
echo "   Python版本: \$(python --version 2>&1)"
echo "   Git狀態: \$(git status --porcelain | wc -l) 個未提交變更"

echo ""
echo "✅ 準備就緒！可以開始訓練模型了"
echo "💡 常用命令："
echo "   訓練模型: python src/train.py"
echo "   快速驗證: python verify_preprocessing.py"
echo "   提交代碼: ./daily_commit.sh '訊息'"
EOF

    # 每日提交腳本
    cat > daily_commit.sh << EOF
#!/bin/bash
# 每日提交腳本

if [ -z "\$1" ]; then
    echo "❌ 請提供提交訊息"
    echo "用法: ./daily_commit.sh '今日訓練進度'"
    exit 1
fi

echo "📤 提交今日進度..."

# 添加所有變更（除了.gitignore排除的）
git add .

# 提交
git commit -m "\$1"

# 推送
git push

echo "✅ 提交完成！"
EOF

    # 專案清理腳本
    cat > cleanup.sh << EOF
#!/bin/bash
# 專案完成後清理腳本

echo "🧹 清理Git憑證和敏感信息..."

# 清理憑證
git config --global --unset credential.helper
rm -f ~/.git-credentials

# 清理環境變量
unset GITHUB_TOKEN

echo "✅ 清理完成！"
EOF

    # 設置腳本權限
    chmod +x daily_start.sh daily_commit.sh cleanup.sh
    
    echo ""
    echo "🎉 設置完成！"
    echo ""
    echo "📋 接下來您可以："
    echo "   1. 運行 ./daily_start.sh 開始每日工作"
    echo "   2. 使用 git pull 和 git push 正常操作"
    echo "   3. 使用 ./daily_commit.sh '訊息' 快速提交"
    echo "   4. 專案完成後運行 ./cleanup.sh 清理"
    echo ""
    echo "📁 專案位置: $project_dir/$repo_name"
    
else
    echo "❌ 克隆失敗！請檢查："
    echo "   1. Token是否正確"
    echo "   2. 用戶名是否正確"
    echo "   3. 倉庫名是否正確"
    echo "   4. 網絡連接是否正常"
fi
