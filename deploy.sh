#!/bin/bash

# ============================================
# 人脸识别检索系统 - 一键部署脚本
# ============================================
# 功能：自动拉取代码、构建镜像、启动容器、清理环境

set -euo pipefail

# ============================================
# 配置参数
# ============================================
APP_NAME="face-recognition"
DOCKER_COMPOSE_FILE="docker-compose.yml"
GIT_REPO="."  # 使用本地仓库，如需远程拉取改为远程仓库地址

# ============================================
# 颜色定义
# ============================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================
# 日志函数
# ============================================
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================
# 主部署流程
# ============================================
main() {
    info "========== 开始部署 ${APP_NAME} =========="

    # 1. 拉取最新代码
    step_pull_code

    # 2. 停止旧容器
    step_stop_containers

    # 3. 清理无用资源
    step_cleanup

    # 4. 构建镜像
    step_build_image

    # 5. 启动新容器
    step_start_containers

    # 6. 验证部署
    step_verify_deployment

    success "========== 部署完成！=========="
}

# ============================================
# 步骤1: 拉取最新代码
# ============================================
step_pull_code() {
    info "步骤1: 拉取最新代码"
    
    if [ -d ".git" ]; then
        info "正在从 Git 仓库拉取最新代码..."
        git pull origin main
        success "代码拉取完成"
    else
        warning "当前目录不是 Git 仓库，跳过代码拉取"
    fi
}

# ============================================
# 步骤2: 停止旧容器
# ============================================
step_stop_containers() {
    info "步骤2: 停止旧容器"
    
    info "正在停止并删除容器..."
    if docker-compose -f "${DOCKER_COMPOSE_FILE}" down 2>/dev/null; then
        success "容器停止完成"
    else
        warning "未找到运行中的容器，跳过停止步骤"
    fi
}

# ============================================
# 步骤3: 清理无用资源
# ============================================
step_cleanup() {
    info "步骤3: 清理无用资源"
    
    info "清理未使用的 Docker 镜像..."
    docker image prune -f
    
    info "清理未使用的 Docker 卷..."
    docker volume prune -f
    
    info "清理未使用的 Docker 网络..."
    docker network prune -f
    
    success "清理完成"
}

# ============================================
# 步骤4: 构建镜像
# ============================================
step_build_image() {
    info "步骤4: 构建 Docker 镜像"
    
    info "开始构建 ${APP_NAME} 镜像..."
    if docker-compose -f "${DOCKER_COMPOSE_FILE}" build --no-cache; then
        success "镜像构建完成"
    else
        error "镜像构建失败！"
        exit 1
    fi
}

# ============================================
# 步骤5: 启动新容器
# ============================================
step_start_containers() {
    info "步骤5: 启动新容器"
    
    info "启动 ${APP_NAME} 服务..."
    if docker-compose -f "${DOCKER_COMPOSE_FILE}" up -d; then
        success "容器启动完成"
    else
        error "容器启动失败！"
        exit 1
    fi
}

# ============================================
# 步骤6: 验证部署
# ============================================
step_verify_deployment() {
    info "步骤6: 验证部署"
    
    info "等待服务启动..."
    sleep 10
    
    info "检查容器运行状态..."
    if docker-compose -f "${DOCKER_COMPOSE_FILE}" ps; then
        success "容器状态检查通过"
    else
        error "容器状态检查失败！"
        exit 1
    fi
    
    info "检查服务健康状态..."
    # 尝试访问服务
    if curl -f -s http://localhost:8765/ > /dev/null 2>&1; then
        success "服务健康检查通过"
        success "服务已启动，访问地址: http://localhost:8765"
    else
        warning "服务健康检查未通过，可能需要更多启动时间"
        warning "请稍后手动检查: http://localhost:8765"
    fi
}

# ============================================
# 执行主函数
# ============================================
main "$@"
