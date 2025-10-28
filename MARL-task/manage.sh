#!/bin/bash
# 项目管理脚本 - 统一入口
# 作者: Duan
# 日期: 2025-10-28

set -e

PROJECT_ROOT="/Users/duan/mac-miclsirr/Sensors-models/MARL-task"
cd "$PROJECT_ROOT"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# 显示帮助
show_help() {
    cat << EOF
📚 MARL项目管理脚本

用法: ./manage.sh <命令> [选项]

命令:
  📦 项目管理
    check          检查系统环境和依赖
    test           快速测试环境
    archive        归档旧代码 (EPyMARL/MATD3)
    clean          清理日志和临时文件

  🚀 训练管理
    train          启动RLlib训练
    tensorboard    启动TensorBoard监控

  🌐 服务管理
    backend        启动后端服务 (Port 8000)
    frontend       启动前端服务 (Port 3000)
    start          启动所有服务 (后端+前端+TensorBoard)
    stop           停止所有服务

  📊 日志管理
    logs           查看日志 (training/backend/testing)

示例:
  ./manage.sh check           # 检查环境
  ./manage.sh test            # 测试环境
  ./manage.sh train           # 开始训练
  ./manage.sh start           # 启动所有服务
  ./manage.sh logs training   # 查看训练日志

EOF
}

# 检查系统
check_system() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  系统环境检查${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # 检查Python
    if command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1)
        echo -e "${GREEN}✓${NC} Python: $PYTHON_VERSION"
    else
        echo -e "${RED}✗${NC} Python: 未安装"
    fi

    # 检查Conda
    if command -v conda &> /dev/null; then
        CONDA_VERSION=$(conda --version 2>&1)
        echo -e "${GREEN}✓${NC} Conda: $CONDA_VERSION"
        if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
            echo -e "${GREEN}✓${NC} 当前环境: $CONDA_DEFAULT_ENV"
        fi
    else
        echo -e "${YELLOW}⚠${NC} Conda: 未安装"
    fi

    # 检查关键依赖
    echo ""
    echo "检查Python包:"
    for pkg in ray torch gymnasium pandas numpy; do
        if python -c "import $pkg" &> /dev/null; then
            version=$(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null)
            echo -e "${GREEN}✓${NC} $pkg: $version"
        else
            echo -e "${RED}✗${NC} $pkg: 未安装"
        fi
    done

    # 检查GPU
    echo ""
    if python -c "import torch; torch.cuda.is_available()" &> /dev/null; then
        gpu_count=$(python -c "import torch; print(torch.cuda.device_count())")
        echo -e "${GREEN}✓${NC} GPU: $gpu_count 个可用"
    else
        echo -e "${YELLOW}⚠${NC} GPU: 不可用 (将使用CPU)"
    fi

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# 快速测试
quick_test() {
    echo -e "${YELLOW}[测试] 启动环境测试...${NC}"
    python rllib_env/test_env.py
    echo -e "${GREEN}✓ 环境测试完成${NC}"
}

# 归档代码
archive_code() {
    echo -e "${YELLOW}[归档] 开始归档旧代码...${NC}"

    if [ -f "archive_old_code.sh" ]; then
        bash archive_old_code.sh
    else
        echo -e "${RED}✗ archive_old_code.sh 不存在${NC}"
    fi
}

# 清理日志
clean_logs() {
    echo -e "${YELLOW}[清理] 清理日志和临时文件...${NC}"

    # 清理Python缓存
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

    # 清理日志 (保留最近7天)
    if [ -d "logs" ]; then
        find logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
        echo -e "${GREEN}✓ 清理了7天前的日志${NC}"
    fi

    # 清理临时文件
    rm -rf .pytest_cache .mypy_cache .coverage 2>/dev/null || true

    echo -e "${GREEN}✓ 清理完成${NC}"
}

# 启动训练
start_training() {
    echo -e "${YELLOW}[训练] 启动RLlib训练...${NC}"
    echo ""
    echo "使用方法:"
    echo "  基础训练: python rllib_train.py --algo SAC --local"
    echo "  高级训练: python rllib_train_advanced.py --auto-resources"
    echo ""
    read -p "按回车开始基础训练，或Ctrl+C取消..."
    python rllib_train.py --algo SAC --num-vessels 10 --iterations 100 --local
}

# 启动TensorBoard
start_tensorboard() {
    echo -e "${YELLOW}[TensorBoard] 启动监控...${NC}"

    if [ ! -d "ray_results" ]; then
        mkdir -p ray_results
    fi

    echo "TensorBoard地址: http://localhost:6006"
    tensorboard --logdir=./ray_results --port=6006
}

# 启动后端
start_backend() {
    echo -e "${YELLOW}[后端] 启动FastAPI服务...${NC}"
    cd backend
    echo "后端地址: http://localhost:8000"
    echo "API文档: http://localhost:8000/docs"
    uvicorn app:app --reload --port 8000
}

# 启动前端
start_frontend() {
    echo -e "${YELLOW}[前端] 启动Web服务...${NC}"
    cd frontend
    echo "前端地址: http://localhost:3000"
    python -m http.server 3000
}

# 启动所有服务
start_all() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  启动所有服务${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # 检查端口占用
    check_port() {
        if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo -e "${YELLOW}⚠ 端口 $1 已被占用${NC}"
            return 1
        fi
        return 0
    }

    # 检查所有端口
    all_ports_free=true
    for port in 8000 3000 6006; do
        if ! check_port $port; then
            all_ports_free=false
        fi
    done

    if [ "$all_ports_free" = false ]; then
        echo ""
        read -p "是否继续启动? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    echo -e "${GREEN}启动服务...${NC}"
    echo ""

    # 启动后端
    echo -e "${YELLOW}[1/3] 启动后端 (Port 8000)${NC}"
    cd backend
    nohup uvicorn app:app --reload --port 8000 > ../logs/backend.log 2>&1 &
    BACKEND_PID=$!
    cd ..
    sleep 2

    # 启动前端
    echo -e "${YELLOW}[2/3] 启动前端 (Port 3000)${NC}"
    cd frontend
    nohup python -m http.server 3000 > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ..
    sleep 1

    # 启动TensorBoard
    echo -e "${YELLOW}[3/3] 启动TensorBoard (Port 6006)${NC}"
    mkdir -p ray_results
    nohup tensorboard --logdir=./ray_results --port=6006 > logs/tensorboard.log 2>&1 &
    TENSORBOARD_PID=$!
    sleep 2

    # 保存PID
    echo "$BACKEND_PID" > .backend.pid
    echo "$FRONTEND_PID" > .frontend.pid
    echo "$TENSORBOARD_PID" > .tensorboard.pid

    echo ""
    echo -e "${GREEN}✓ 所有服务已启动${NC}"
    echo ""
    echo "访问地址:"
    echo "  后端API:      http://localhost:8000"
    echo "  API文档:      http://localhost:8000/docs"
    echo "  前端界面:     http://localhost:3000"
    echo "  TensorBoard:  http://localhost:6006"
    echo ""
    echo "停止服务: ./manage.sh stop"
    echo ""
}

# 停止所有服务
stop_all() {
    echo -e "${YELLOW}停止所有服务...${NC}"

    # 停止后端
    if [ -f .backend.pid ]; then
        kill $(cat .backend.pid) 2>/dev/null && echo -e "${GREEN}✓ 后端已停止${NC}" || true
        rm .backend.pid
    fi

    # 停止前端
    if [ -f .frontend.pid ]; then
        kill $(cat .frontend.pid) 2>/dev/null && echo -e "${GREEN}✓ 前端已停止${NC}" || true
        rm .frontend.pid
    fi

    # 停止TensorBoard
    if [ -f .tensorboard.pid ]; then
        kill $(cat .tensorboard.pid) 2>/dev/null && echo -e "${GREEN}✓ TensorBoard已停止${NC}" || true
        rm .tensorboard.pid
    fi

    # 强制清理端口
    for port in 8000 3000 6006; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            lsof -ti:$port | xargs kill -9 2>/dev/null || true
        fi
    done

    echo -e "${GREEN}✓ 所有服务已停止${NC}"
}

# 查看日志
view_logs() {
    log_type=${1:-training}

    case $log_type in
        training)
            echo -e "${BLUE}查看训练日志 (最新50行):${NC}"
            if [ -d "logs/training" ]; then
                latest_log=$(ls -t logs/training/*.log 2>/dev/null | head -1)
                if [ ! -z "$latest_log" ]; then
                    tail -50 "$latest_log"
                else
                    echo "没有找到训练日志"
                fi
            else
                echo "训练日志目录不存在"
            fi
            ;;
        backend)
            echo -e "${BLUE}查看后端日志 (最新50行):${NC}"
            if [ -f "logs/backend.log" ]; then
                tail -50 logs/backend.log
            else
                echo "没有找到后端日志"
            fi
            ;;
        testing)
            echo -e "${BLUE}查看测试日志 (最新50行):${NC}"
            if [ -d "logs/testing" ]; then
                latest_log=$(ls -t logs/testing/*.log 2>/dev/null | head -1)
                if [ ! -z "$latest_log" ]; then
                    tail -50 "$latest_log"
                else
                    echo "没有找到测试日志"
                fi
            else
                echo "测试日志目录不存在"
            fi
            ;;
        *)
            echo "未知日志类型: $log_type"
            echo "支持的类型: training, backend, testing"
            ;;
    esac
}

# 主逻辑
case "${1:-help}" in
    check)
        check_system
        ;;
    test)
        quick_test
        ;;
    archive)
        archive_code
        ;;
    clean)
        clean_logs
        ;;
    train)
        start_training
        ;;
    tensorboard)
        start_tensorboard
        ;;
    backend)
        start_backend
        ;;
    frontend)
        start_frontend
        ;;
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    logs)
        view_logs $2
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}错误: 未知命令 '$1'${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
