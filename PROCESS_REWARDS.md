# Process-Based Rewards Implementation

## 概述

我们实现了**基于过程的奖励（Process-Based Rewards）**系统，这是对原有**基于结果的奖励（Outcome-Based Rewards）**系统的重大改进。

### 核心改进

**之前（Outcome-Based）：**
- 整个 trajectory 获得一个统一的奖励分数
- 所有 assistant tokens 使用相同的 advantage 值
- 模型无法区分哪些中间步骤是好的，哪些是坏的

**现在（Process-Based）：**
- 每个 action（search/read/answer）获得独立的奖励
- 每个 action 的 tokens 有自己的 advantage 值
- 模型可以学习：即使最终答案错了，正确的搜索和阅读也是好的

## 实现细节

### 1. 数据结构修改

#### TrajectorySample
```python
@dataclass
class TrajectorySample:
    query_id: str
    query: SyntheticQuery  # 新增：用于计算过程奖励
    conversation: List[Dict]
    reward: float
    rubric: EvaluationRubric
    rollout_idx: int
    group_id: int
    advantage: Optional[float] = None  # 保留用于group-level计算
```

#### TokenizedTrajectory
```python
@dataclass
class TokenizedTrajectory:
    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    attention_mask: torch.Tensor
    advantage_mask: torch.Tensor  # 改为token-level的advantage mask
    group_id: int
    query_id: str
    old_logprobs: Optional[torch.Tensor] = None
```

### 2. 核心方法：_compute_action_advantage

这是过程奖励的核心逻辑，为每个 action 计算独立的 advantage：

```python
def _compute_action_advantage(
    self,
    msg: Dict,
    rubric: EvaluationRubric,
    query: SyntheticQuery,
    msg_idx: int,
    conversation: List[Dict],
    trajectory_advantage: float,  # Group-level advantage作为baseline
) -> float:
```

#### 奖励规则

| Action | 条件 | Advantage 调整 | 说明 |
|--------|------|----------------|------|
| **search_emails** | 找到正确邮件 | `+0.8` (至少 +0.3) | ✅ 好的搜索！ |
| **search_emails** | 有结果但不对 | `-0.2` | 搜索策略需改进 |
| **search_emails** | 空结果 | `-0.15` | 可能是正常探索 |
| **read_email** | 阅读正确邮件 | `+0.8` (至少 +0.3) | ✅ 好的阅读！ |
| **read_email** | 阅读错误邮件 | `-0.25` | 阅读了错误的邮件 |
| **return_final_answer** | 正确答案 | `+1.5` (至少 +1.0) | ✅✅ 完美！ |
| **return_final_answer** | 错误答案 | `-1.5` (至多 -0.8) | ❌ 答案错误！ |
| **其他工具** | 无效的工具名 | `-1.2` (至多 -1.5) | ❌ 严重错误 |

### 3. Tokenization 修改

`tokenize_conversation_with_mask` 现在接受额外参数并返回 advantage_mask：

```python
def tokenize_conversation_with_mask(
    self,
    conversation: List[Dict],
    rubric: Optional[EvaluationRubric] = None,
    query: Optional[SyntheticQuery] = None,
    trajectory_advantage: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns: input_ids, labels, loss_mask, advantage_mask"""
```

对于每个 assistant message，调用 `_compute_action_advantage` 计算该 action 的 advantage，然后将其应用到该 action 的所有 tokens。

### 4. 批量处理修改

在创建训练批次时，使用 token-level 的 `advantage_mask`：

```python
# 之前：统一的 advantage
advantage_list.append(torch.full_like(trunc_old_logprobs, sample.advantage))

# 现在：token-level 的 advantage_mask
trunc_advantage = sample.advantage_mask[:trunc_len]
shift_advantage = trunc_advantage[1:]
advantage_list.append(shift_advantage[:len(trunc_old_logprobs)])
```

### 5. 损失计算

PPO/GRPO 损失现在使用 token-level 的 advantages：

```python
# advantages 现在是 [batch_size, seq_len]，每个 token 有独立的 advantage
surrogate_1 = prob_ratio * advantages
surrogate_2 = clipped_ratio * advantages
policy_loss = -torch.min(surrogate_1, surrogate_2)
```

## 训练示例

### 示例场景

假设模型在一次 rollout 中：
1. ✅ **搜索** → 找到了正确的邮件（`msg_correct_123`）
2. ✅ **阅读** → 阅读了正确的邮件
3. ❌ **回答** → 但给出了错误的答案

### 奖励分配

使用 **Outcome-Based Rewards（之前）**：
- 整个 trajectory 得分：`-1.0`（错误答案）
- 所有 tokens 的 advantage：`-0.5`（假设）
- **问题**：模型被惩罚搜索和阅读正确邮件！

使用 **Process-Based Rewards（现在）**：
- Search action tokens: advantage = `+0.8` ✅
- Read action tokens: advantage = `+0.8` ✅
- Answer action tokens: advantage = `-1.2` ❌
- **好处**：模型学到搜索和阅读是对的，只有答案需要改进！

## 预期效果

### 1. 更好的信用分配（Credit Assignment）
模型可以明确知道：
- 哪些步骤做对了 → 保持/强化
- 哪些步骤做错了 → 改进

### 2. 更快的收敛
- 更密集的学习信号（每个 action 都有反馈）
- 减少样本浪费（好的中间步骤即使最终失败也能学习）

### 3. 更好的泛化
- 学习正确的中间步骤模式
- 不仅仅关注最终结果

### 4. 潜在的样本效率提升
- 每个 trajectory 提供更多信息
- 部分成功的 trajectory 也有学习价值

## 使用方法

### 训练命令

```bash
# 使用 masked mode（自动启用 process-based rewards）
python train_grpo.py --mode masked

# 或使用环境变量配置
export MAX_STEPS=200
export LEARNING_RATE=1e-5
python train_grpo.py --mode masked
```

### 验证实现

所有修改都经过验证：
- ✅ 数据结构正确更新
- ✅ _compute_action_advantage 逻辑正确
- ✅ Tokenization 返回 advantage_mask
- ✅ 批量处理使用 token-level advantages
- ✅ 损失计算应用 process rewards

## 技术细节

### Advantage 计算公式

对于每个 action：

```
action_advantage = baseline_advantage ± reward_delta

where:
  baseline_advantage = trajectory-level advantage (from GRPO)
  reward_delta = action-specific reward based on outcome
```

### 与 GRPO 的兼容性

Process-based rewards 完全兼容 GRPO 算法：
1. 仍然按 query group 计算 group-level advantages
2. 在 group-level advantage 基础上，调整每个 action 的 advantage
3. PPO clipping 和 KL penalty 照常工作

### Token-level Masking

重要：只有 assistant tokens 有非零的 advantage：
```python
if is_model_generated:  # assistant
    advantages = [action_advantage] * len(tokens)
else:  # user, system, tool
    advantages = [0.0] * len(tokens)
```

这确保只有模型生成的 tokens 参与训练。

## 监控和调试

### Wandb 日志

训练时会记录：
- Group-level advantages（原有）
- Token-level advantage 分布（新增）
- 每个 action 类型的平均 advantage

### 检查点

建议监控：
- 正确 search/read 的 trajectory 是否有正 advantage
- 错误 answer 的 action 是否有负 advantage
- Advantage 分布是否合理（不要全是极值）

## 未来改进方向

### 1. 自适应权重
根据训练进度动态调整奖励权重

### 2. 更细粒度的奖励
- 搜索质量评分（不只是找到/未找到）
- 阅读相关性评分

### 3. 学习过程奖励
使用 meta-learning 自动学习最优的奖励权重

### 4. 多步依赖
考虑 action 序列的依赖关系（例如，好的搜索 + 好的阅读 = 更高奖励）

## 参考文献

Process-based rewards 的概念来源于：
1. **Process Reward Models (PRMs)** - OpenAI
2. **Outcome vs. Process Supervision** - 强化学习文献
3. **Fine-grained Credit Assignment** - RLHF 研究

---

**实现时间**: 2025-11-20  
**版本**: 1.0  
**状态**: ✅ 已完成并验证

