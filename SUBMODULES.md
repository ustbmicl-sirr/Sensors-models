# Git子模块管理指南

## 项目子模块

本项目使用git submodule来管理相关子项目。

### 当前子模块列表

#### 1. modelGrow

**路径**: `modelGrow/`
**仓库**: https://github.com/ustbmicl-sirr/modelGrow.git
**用途**: 自动网络增长与结构重参数化 - 模型优化

---

## 📚 Git Submodule使用指南

### 首次克隆包含子模块的仓库

如果是第一次克隆本仓库，有两种方式获取子模块：

#### 方法1: 克隆时同时获取子模块（推荐）
```bash
git clone --recurse-submodules https://github.com/ustbmicl-sirr/Sensors-models.git
```

#### 方法2: 先克隆主仓库，再初始化子模块
```bash
# 克隆主仓库
git clone https://github.com/ustbmicl-sirr/Sensors-models.git
cd Sensors-models

# 初始化并更新子模块
git submodule init
git submodule update
```

### 更新子模块到最新版本

```bash
# 在Sensors-models根目录

# 更新所有子模块到远程最新版本
git submodule update --remote --merge

# 或者进入子模块目录单独更新
cd modelGrow
git pull origin main
cd ..

# 提交子模块更新
git add modelGrow
git commit -m "chore: 更新modelGrow子模块到最新版本"
git push
```

### 查看子模块状态

```bash
# 查看所有子模块状态
git submodule status

# 查看子模块详细信息
git submodule foreach git status
```

### 在子模块中进行开发

```bash
# 进入子模块目录
cd modelGrow

# 创建新分支进行开发
git checkout -b feature/my-feature

# 进行修改后提交
git add .
git commit -m "feat: 添加新功能"

# 推送到子模块仓库
git push origin feature/my-feature

# 返回主仓库
cd ..

# 更新主仓库的子模块引用
git add modelGrow
git commit -m "chore: 更新modelGrow子模块引用"
git push
```

### 删除子模块

如果需要删除子模块：

```bash
# 1. 删除子模块配置
git submodule deinit -f modelGrow

# 2. 删除子模块目录
rm -rf .git/modules/modelGrow

# 3. 删除工作目录中的子模块
git rm -f modelGrow

# 4. 提交更改
git commit -m "chore: 移除modelGrow子模块"
```

---

## ⚠️ 常见问题

### 问题1: 子模块目录为空

**原因**: 克隆仓库时没有使用 `--recurse-submodules` 参数

**解决方法**:
```bash
git submodule init
git submodule update
```

### 问题2: 子模块处于"detached HEAD"状态

**原因**: 子模块默认检出特定的commit，而不是分支

**解决方法**: 如果需要在子模块中开发
```bash
cd modelGrow
git checkout main  # 或其他分支
```

### 问题3: 拉取主仓库后子模块未更新

**原因**: git pull 不会自动更新子模块

**解决方法**:
```bash
git pull
git submodule update --init --recursive
```

### 问题4: 子模块有未提交的更改

**解决方法**:
```bash
cd modelGrow
git status  # 查看更改
git add .
git commit -m "提交说明"
git push
cd ..
git add modelGrow
git commit -m "更新子模块引用"
```

---

## 🔧 项目结构

```
Sensors-models/
├── .gitmodules             # 子模块配置文件
├── modelGrow/              # 子模块：模型优化
│   ├── README.md
│   └── ...（子模块内容）
├── MARL-task/              # 主项目：多智能体强化学习
│   ├── environment/
│   ├── agents/
│   ├── rewards/
│   ├── rllib_env/
│   ├── docs/
│   └── ...
├── SUBMODULES.md           # 本文档
└── README.md
```

---

## 📖 相关链接

- [Git Submodule官方文档](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [modelGrow仓库](https://github.com/ustbmicl-sirr/modelGrow)
- [主仓库](https://github.com/ustbmicl-sirr/Sensors-models)

---

**更新时间**: 2025-10-28
**维护者**: Duan
