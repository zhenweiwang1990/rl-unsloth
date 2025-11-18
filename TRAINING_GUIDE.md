# GRPO 训练指南

## 🚀 统一训练脚本

所有训练功能已合并到单个文件：`train_grpo.py`

## 📋 训练模式

### 1. Masked 模式（推荐）⭐

完整的 token 级别掩码实现，架构最正确。

```bash
# 默认模式
python train_grpo.py --mode masked

# 或者简写
python train_grpo.py
```

**特点**：
- ✅ 真实 agent rollouts
- ✅ Token 级别掩码
- ✅ 只训练模型生成的 tokens
- ✅ Tool results 不参与训练
- ✅ 完整的 GRPO 算法
- ⚠️  训练较慢（但正确）

**适合**：正式训练、追求最佳性能

---

### 2. Rollout 模式

使用真实 rollouts，但没有完整的 token 掩码。

```bash
python train_grpo.py --mode rollout
```

**特点**：
- ✅ 真实 agent rollouts
- ✅ 基于真实性能的 reward
- ❌ 没有 token 掩码（会训练 tool results）
- ⚠️  架构不完全正确

**适合**：快速验证、中等规模测试

---

### 3. Simple 模式

使用 TRL 快速训练，基于启发式 reward。

```bash
python train_grpo.py --mode simple
```

**特点**：
- ❌ 静态数据（没有真实 rollouts）
- ❌ 启发式 reward（不基于真实性能）
- ❌ 没有 token 掩码
- ✅ 训练快速

**适合**：快速原型、功能测试

---

## ⚙️ 配置参数

### 基本配置

```bash
# 数据集大小
export TRAIN_DATASET_SIZE=50
export EVAL_DATASET_SIZE=20

# 训练参数
export MAX_STEPS=200
export LEARNING_RATE=1e-5
export PER_DEVICE_TRAIN_BATCH_SIZE=2

# GRPO 参数
export NUM_GENERATIONS=4
export BETA=0.01

# Agent 参数
export MAX_TURNS=4
export MAX_TOKENS=2048

# 输出目录（会根据模式自动调整）
export OUTPUT_DIR=outputs/grpo

# 运行
python train_grpo.py --mode masked
```

### 推荐配置

#### 快速测试
```bash
export TRAIN_DATASET_SIZE=10
export EVAL_DATASET_SIZE=5
export MAX_STEPS=20
export NUM_GENERATIONS=2
python train_grpo.py --mode masked
```

#### 正式训练
```bash
export TRAIN_DATASET_SIZE=100
export EVAL_DATASET_SIZE=30
export MAX_STEPS=500
export NUM_GENERATIONS=4
export LEARNING_RATE=1e-5
export BETA=0.01
python train_grpo.py --mode masked
```

---

## 📊 训练输出

### Masked 模式日志

```
============================================================
GRPO Training - Mode: MASKED
============================================================
✅ Full implementation with token-level masking (RECOMMENDED)
============================================================
AgentGRPOTrainer initialized
============================================================
Train queries: 50
Eval queries: 20
Rollouts per query: 4
Target accuracy: 95.0%
============================================================

Step 1/200 | Loss: 0.5234 | Policy: 0.5123 | KL: 0.0111 | 
Reward: 0.456 | Acc: 25.0% | 
Trainable tokens: 1234/5678 (21.7%)

────────────────────────────────────────────────────────────
Evaluation at step 50
────────────────────────────────────────────────────────────
Evaluating on 20 queries...
📊 Eval reward: 0.723
📊 Eval accuracy: 45.00%
✨ New best accuracy: 45.00%
💾 Model saved to: outputs/grpo_masked/best_model
────────────────────────────────────────────────────────────
```

### 关键指标

- **Trainable tokens 比例**：20-30% 正常（masked 模式）
- **Loss**：应该逐渐下降
- **Reward**：应该逐渐上升（0.2 → 1.5）
- **Accuracy**：应该逐渐上升（10% → 95%）

---

## 🎯 模式选择建议

| 需求 | 推荐模式 | 原因 |
|-----|---------|------|
| 正式训练 | masked | 架构正确，性能最佳 |
| 快速验证 | rollout | 真实 rollout，速度适中 |
| 功能测试 | simple | 最快速，适合调试 |
| 追求性能 | masked | 唯一正确的实现 |
| 资源有限 | simple | 最省时间和资源 |

---

## 📁 输出文件

训练完成后会生成：

```
outputs/grpo_[mode]/
├── best_model/              # 最佳模型（准确率最高）
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── metadata.json        # 包含准确率等信息
│   └── ...
├── final/                   # 最终模型
│   └── ...
└── checkpoint-*/            # 中间检查点
    └── ...
```

### 使用最佳模型

```python
from unsloth import FastLanguageModel
from email_agent.agent import EmailAgent

# 加载最佳模型
model, tokenizer = FastLanguageModel.from_pretrained(
    "outputs/grpo_masked/best_model"
)

# 创建 agent
agent = EmailAgent(model, tokenizer, policy_config)
```

---

## 🔍 故障排查

### 问题：内存不足

```bash
# 减小批次和 rollouts
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export NUM_GENERATIONS=2
export MAX_TOKENS=1024
```

### 问题：训练太慢

```bash
# 使用 simple 或 rollout 模式
python train_grpo.py --mode rollout

# 或减少数据集
export TRAIN_DATASET_SIZE=20
export NUM_GENERATIONS=2
```

### 问题：准确率不提升

1. 检查 trainable tokens 比例（masked 模式应该 20-30%）
2. 检查 reward 是否增长
3. 尝试调整学习率
4. 验证数据集质量

---

## 🆚 模式对比

| 特性 | Simple | Rollout | Masked |
|-----|--------|---------|--------|
| 真实 Rollout | ❌ | ✅ | ✅ |
| Token 掩码 | ❌ | ❌ | ✅ |
| 训练速度 | 快 | 中等 | 慢 |
| 架构正确性 | ❌ | ⚠️ | ✅ |
| 预期性能 | 差 | 中等 | 最好 |
| 推荐使用 | 测试 | 验证 | 生产 |

---

## 💡 最佳实践

### 训练流程

1. **快速验证**（5-10分钟）
   ```bash
   export TRAIN_DATASET_SIZE=10
   export MAX_STEPS=20
   python train_grpo.py --mode simple
   ```

2. **中等测试**（30-60分钟）
   ```bash
   export TRAIN_DATASET_SIZE=30
   export MAX_STEPS=100
   python train_grpo.py --mode rollout
   ```

3. **正式训练**（2-4小时）
   ```bash
   export TRAIN_DATASET_SIZE=100
   export MAX_STEPS=500
   python train_grpo.py --mode masked
   ```

### 超参数调优

```bash
# 学习率扫描
for lr in 1e-5 5e-6 1e-6; do
    export LEARNING_RATE=$lr
    python train_grpo.py --mode masked
done
```

---

## 📚 相关文档

- `TOKEN_MASKING_GUIDE.md` - Token 掩码详细说明
- `MASKED_TRAINING_GUIDE.md` - Masked 模式完整指南
- `BEFORE_AFTER_COMPARISON.md` - 改进前后对比

---

## ✅ 快速开始

```bash
# 1. 设置环境变量
export TRAIN_DATASET_SIZE=50
export EVAL_DATASET_SIZE=20
export MAX_STEPS=200

# 2. 运行训练（推荐 masked 模式）
python train_grpo.py --mode masked

# 3. 检查输出
ls -la outputs/grpo_masked/best_model/

# 4. 查看元数据
cat outputs/grpo_masked/best_model/metadata.json
```

---

## 🎉 总结

- **一个脚本，三种模式**：`train_grpo.py`
- **推荐使用**：`--mode masked`
- **快速测试**：`--mode simple`
- **中等方案**：`--mode rollout`

立即开始：
```bash
python train_grpo.py --mode masked
```

