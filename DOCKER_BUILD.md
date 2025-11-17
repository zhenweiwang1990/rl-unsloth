# Docker 构建指南

本文档说明如何构建和使用 Email Agent GRPO 训练的 Docker 镜像。

## 快速开始

### 1. 构建镜像

最简单的方式：

```bash
./scripts/build_docker.sh
```

这将构建名为 `email-agent-grpo:latest` 的镜像。

### 2. 自定义构建

#### 指定镜像名称和标签

```bash
./scripts/build_docker.sh --name my-agent --tag v1.0
```

或使用环境变量：

```bash
IMAGE_NAME=my-agent IMAGE_TAG=v1.0 ./scripts/build_docker.sh
```

#### 无缓存构建

如果需要完全重新构建（不使用 Docker 缓存层）：

```bash
./scripts/build_docker.sh --no-cache
```

### 3. 查看帮助

```bash
./scripts/build_docker.sh --help
```

## 缓存机制

### 构建时缓存（BuildKit）

Dockerfile 使用 BuildKit 的缓存挂载功能：

```dockerfile
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt
```

**优势：**
- pip 下载的包在多次构建之间缓存
- 即使修改了 requirements.txt，只下载新增的包
- 加速重复构建

**要求：**
- Docker 18.09+
- 启用 DOCKER_BUILDKIT=1（脚本自动处理）

### 运行时缓存（Volume 挂载）

运行容器时，脚本会自动挂载宿主机缓存目录：

```bash
docker run --gpus all -it \
    -v ~/.cache/pip:/root/.cache/pip \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    --env-file .env \
    email-agent-grpo:latest bash
```

**缓存目录映射：**

| 宿主机 | 容器内 | 用途 |
|--------|--------|------|
| `~/.cache/pip` | `/root/.cache/pip` | pip 包缓存 |
| `~/.cache/huggingface` | `/root/.cache/huggingface` | HuggingFace 模型和数据集 |
| `~/.cache/transformers` | `/root/.cache/transformers` | Transformers 缓存 |

**优势：**
- HuggingFace 模型只下载一次
- 多个容器共享同一份模型和数据
- 容器删除后缓存不丢失

## 构建选项

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--name NAME` | 镜像名称 | `email-agent-grpo` |
| `--tag TAG` | 镜像标签 | `latest` |
| `--no-cache` | 不使用 Docker 缓存层 | `false` |
| `--help` | 显示帮助信息 | - |

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `IMAGE_NAME` | 镜像名称 | `email-agent-grpo` |
| `IMAGE_TAG` | 镜像标签 | `latest` |
| `NO_CACHE` | 不使用缓存 | `false` |

## 使用示例

### 基础构建

```bash
./scripts/build_docker.sh
```

### 开发版本

```bash
./scripts/build_docker.sh --name email-agent --tag dev
```

### 生产版本

```bash
./scripts/build_docker.sh --name email-agent --tag v1.0.0
```

### 清理缓存重建

```bash
./scripts/build_docker.sh --no-cache
```

### 多版本管理

```bash
# 开发版
IMAGE_TAG=dev ./scripts/build_docker.sh

# 测试版
IMAGE_TAG=staging ./scripts/build_docker.sh

# 生产版
IMAGE_TAG=prod ./scripts/build_docker.sh
```

## 构建后使用

### 1. 交互式运行

```bash
docker run --gpus all -it \
    -v ~/.cache/pip:/root/.cache/pip \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    --env-file .env \
    email-agent-grpo:latest bash
```

### 2. 使用现有脚本

项目提供了便捷的运行脚本：

```bash
# 生成数据库
./scripts/generate_database.sh

# 运行训练
./scripts/run_training.sh

# 运行基准测试
./scripts/run_benchmark.sh

# 运行评估
./scripts/run_eval.sh
```

这些脚本会自动处理缓存挂载。

## 故障排除

### 问题 1: BuildKit 不支持

**错误信息：**
```
unknown flag: --mount
```

**解决方案：**
升级 Docker 到 18.09 或更高版本：

```bash
docker --version
```

### 问题 2: 构建过程中下载缓慢

**解决方案：**
- 使用国内镜像源（修改 Dockerfile 中的 apt 源）
- 使用 pip 镜像源（在 requirements.txt 前添加 pip 配置）
- 确保网络连接正常

### 问题 3: 磁盘空间不足

**查看磁盘使用：**
```bash
docker system df
```

**清理未使用的镜像和容器：**
```bash
docker system prune -a
```

**清理 BuildKit 缓存：**
```bash
docker builder prune
```

### 问题 4: 权限问题

**错误信息：**
```
permission denied while trying to connect to the Docker daemon socket
```

**解决方案：**
```bash
sudo usermod -aG docker $USER
newgrp docker
```

## 最佳实践

### 1. 分层缓存

Dockerfile 已经按照最佳实践组织：

1. 基础镜像和系统依赖（变化少）
2. Python 依赖（变化较少）
3. 项目代码（变化频繁）

这样可以最大化缓存利用率。

### 2. .dockerignore

确保 `.dockerignore` 文件排除不必要的文件：

```
__pycache__/
*.pyc
*.pyo
.git/
.env
outputs/
*.log
```

### 3. 版本管理

建议使用语义化版本标签：

```bash
# 开发版本
./scripts/build_docker.sh --tag dev

# 预发布版本
./scripts/build_docker.sh --tag v1.0.0-rc1

# 正式版本
./scripts/build_docker.sh --tag v1.0.0
./scripts/build_docker.sh --tag latest
```

### 4. 镜像大小优化

当前 Dockerfile 已经包含了优化：

- 使用 `--no-cache-dir` 避免保存 pip 缓存到镜像
- 清理 apt 缓存 `rm -rf /var/lib/apt/lists/*`
- 使用多阶段构建（如需要可进一步优化）

## 相关文档

- [README.md](README.md) - 项目主文档
- [BUGFIX_ENV_COMMENTS.md](BUGFIX_ENV_COMMENTS.md) - 环境变量注释处理
- [BENCHMARK_README.md](BENCHMARK_README.md) - 基准测试文档

## 技术支持

如有问题，请检查：
1. Docker 版本是否满足要求（18.09+）
2. 磁盘空间是否充足
3. 网络连接是否正常
4. 缓存目录权限是否正确

