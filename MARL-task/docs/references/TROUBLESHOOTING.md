# 故障排除指南

## 常见问题

### 1. 启动失败 - "No module named uvicorn"

**原因**: 依赖未安装

**解决**:
```bash
conda activate marl-task
pip install -r requirements.txt
```

### 2. 端口被占用

**错误**: "Address already in use" 或 "[Errno 48] Address already in use"

**说明**: `start_all.sh`脚本现在会自动检测并清理占用的端口，无需手动处理。

**如果仍然遇到端口占用问题**:
```bash
# 方法1: 使用start_all.sh（推荐，自动处理）
./start_all.sh

# 方法2: 手动终止占用端口的进程
lsof -ti:8000 | xargs kill -9  # 后端
lsof -ti:3000 | xargs kill -9  # 前端
lsof -ti:6006 | xargs kill -9  # TensorBoard

# 方法3: 停止所有相关服务
pkill -f uvicorn
pkill -f tensorboard
pkill -f "http.server"
```

### 3. Conda环境问题

**解决**: 重新创建环境
```bash
conda deactivate
conda remove -n marl-task --all -y
conda create -n marl-task python=3.9 -y
conda activate marl-task
pip install -r requirements.txt
```

### 4. 依赖冲突

**解决**: 强制重装
```bash
conda activate marl-task
pip install -r requirements.txt --upgrade --force-reinstall
```

### 5. TensorBoard无数据

**原因**: 未运行训练

**解决**: 运行一次训练生成数据
```bash
python main.py --mode train
```

### 6. Web界面无法连接后端

**检查**:
```bash
# 检查后端是否运行
curl http://localhost:8000/health

# 查看后端日志
./view_backend_logs.sh
```

### 7. 内存不足

**解决**: 修改配置减小资源占用
```yaml
# config/default_config.yaml
training:
  batch_size: 32          # 降低批次大小
  buffer_capacity: 50000  # 减小buffer
```

### 8. 算法运行500错误（已修复）

**问题**: 运行算法时返回500 Internal Server Error

**原因**:
- `calculate_metrics`缺少`num_vessels`字段
- MATD3 agent数量与实际vessel数量不匹配

**状态**: ✅ 已在v2.1中修复

## 查看日志

```bash
# 训练日志
./view_training_logs.sh

# 后端日志
./view_backend_logs.sh

# 实时查看
tail -f logs/backend/api.log
tail -f logs/training/text/training.log
```

## 完全重置

如果问题无法解决，完全重置：

```bash
# 1. 停止所有服务
pkill -f uvicorn
pkill -f tensorboard
pkill -f http.server

# 2. 删除环境
conda deactivate
conda remove -n marl-task --all -y

# 3. 清理日志
./clean_logs.sh  # 选项4

# 4. 重新开始
conda create -n marl-task python=3.9 -y
conda activate marl-task
pip install -r requirements.txt
./start_all.sh
```

## 健康检查

```bash
# 检查后端
curl http://localhost:8000/health

# 检查前端
curl http://localhost:3000/index.html

# 检查TensorBoard
curl http://localhost:6006
```
