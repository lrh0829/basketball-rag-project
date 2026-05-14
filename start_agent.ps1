# 启动篮球规则智能体

Write-Host "🏀 篮球规则智能问答助手启动脚本"
Write-Host "=============================="
Write-Host ""

# 检查Python是否安装
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "错误: 未找到Python，请先安装Python 3.8或更高版本"
    Read-Host "按Enter键退出..."
    exit 1
}

# 检查是否存在.env文件
if (-not (Test-Path ".env")) {
    Write-Host "警告: 未找到.env文件，请先配置环境变量"
    Write-Host "请在.env文件中设置 API_KEY"
    Read-Host "按Enter键退出..."
    exit 1
}

# 检查依赖包是否安装
Write-Host "检查依赖包..."
python -m pip install -r requirements.txt

# 运行智能体
Write-Host ""
Write-Host "启动智能体..."
Write-Host ""
python src\basketball_agent.py

Read-Host "按Enter键退出..."