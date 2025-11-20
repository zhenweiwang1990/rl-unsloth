# 环境变量配置说明

本文档列出了训练脚本支持的所有环境变量及其默认值。

## 模型配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `MODEL_NAME` | `OpenPipe/Qwen3-14B-Instruct` | 基础模型名称 |
| `SEED` | `42` | 随机种子 |

## 数据集配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `TRAIN_DATASET_SIZE` | `3000` | 训练集大小 |
| `EVAL_DATASET_SIZE` | `100` | 验证集大小 |

## 训练超参数

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `MAX_STEPS` | `200` | 最大训练步数 |
| `LEARNING_RATE` | `1e-5` | 学习率 |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `4` | 每个设备的批次大小 |
| `NUM_TRAIN_EPOCHS` | `2` | 训练轮数（仅 simple/rollout 模式） |
| `GRADIENT_ACCUMULATION_STEPS` | `1` | 梯度累积步数 |
| `MAX_GRAD_NORM` | `1.0` | 梯度裁剪的最大范数 |

## GRPO 特定配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `NUM_GENERATIONS` | `3` | 每个问题的生成次数（rollout 数量） |
| `BETA` | `0.01` | KL 散度权重（GRPO 的 beta 参数） |

## 评估与检查点

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `EVAL_STEPS` | `10` | 每 N 步评估一次 |
| `SAVE_STEPS` | `10` | 每 N 步保存检查点 |
| `WARMUP_STEPS` | `10` | 学习率预热步数 |
| `SAVE_TOTAL_LIMIT` | `3` | 保留的最大检查点数量 |
| `TARGET_ACCURACY` | `0.95` | 提前停止的目标准确率（0.0 - 1.0） |
| `PATIENCE` | `5` | 提前停止的耐心值（无改进的评估次数） |

## Agent 配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `MAX_TURNS` | `10` | 最大对话轮数 |
| `MAX_TOKENS` | `4096` | 每次生成的最大 token 数 |
| `TEMPERATURE` | `0.7` | 采样温度（仅 simple/rollout 模式） |

## 高级配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `MIN_GROUP_STD` | `0.05` | 保留组的最小标准差 |
| `VERBOSE` | `false` | 详细日志输出 |

## 输出配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `OUTPUT_DIR` | `outputs/grpo` | 输出目录（检查点、日志等） |

## Weights & Biases 配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `WANDB_PROJECT` | `email-agent-grpo` | W&B 项目名称 |
| `WANDB_ENTITY` | （空） | W&B 实体/组织名称 |
| `WANDB_NAME` | `grpo-{mode}` | W&B 运行名称 |
| `WANDB_MODE` | `online` | W&B 模式：`online`, `offline`, 或 `disabled` |

## API 密钥

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `OPENROUTER_API_KEY` | （空） | OpenRouter API 密钥（用于基于 judge 的评估） |

## 使用示例

### 使用 .env 文件

创建 `.env` 文件：

```bash
# .env
MODEL_NAME=OpenPipe/Qwen3-14B-Instruct
MAX_STEPS=500
LEARNING_RATE=5e-6
PER_DEVICE_TRAIN_BATCH_SIZE=8
EVAL_STEPS=20
SAVE_STEPS=20
TARGET_ACCURACY=0.98
WANDB_MODE=online
OPENROUTER_API_KEY=your_api_key_here
```

然后运行训练：

```bash
./scripts/run_training.sh masked
```

### 直接设置环境变量

```bash
export MAX_STEPS=500
export LEARNING_RATE=5e-6
export EVAL_STEPS=20
python train_grpo.py --mode masked
```

### Docker 运行时设置

```bash
docker run --rm -it \
    --gpus all \
    -e MAX_STEPS=500 \
    -e LEARNING_RATE=5e-6 \
    -e EVAL_STEPS=20 \
    email-agent-grpo \
    python train_grpo.py --mode masked
```

## 注意事项

1. **必需的环境变量**：只有 `OPENROUTER_API_KEY` 是可选的（未设置时使用启发式评估），其他都有合理的默认值。

2. **模式差异**：
   - `masked` 模式：推荐使用，支持完整的 token 级别掩码
   - `rollout` 模式：使用真实的 agent rollout
   - `simple` 模式：快速测试模式

3. **资源考虑**：
   - `PER_DEVICE_TRAIN_BATCH_SIZE` × `NUM_GENERATIONS` 决定了每批次的总 rollout 数量
   - 增加 `GRADIENT_ACCUMULATION_STEPS` 可以在有限的 GPU 内存下模拟更大的批次大小

4. **早停机制**：
   - 当验证准确率达到 `TARGET_ACCURACY` 时训练会自动停止
   - `PATIENCE` 参数控制在准确率不再提升后继续训练的评估次数

