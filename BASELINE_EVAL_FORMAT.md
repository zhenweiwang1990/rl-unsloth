# Baseline Evaluation JSON Format

## 概述

从现在开始，baseline evaluation JSON 文件将包含 `control_reward_map`，使得从 JSON 加载时也能支持 beat-rate 比较功能。

## JSON 格式示例

```json
{
  "step": -1,
  "is_baseline": true,
  "accuracy": 0.85,
  "correct_answers": 85,
  "total_samples": 100,
  "attempted_answer": 90,
  "avg_reward": 2.5,
  "median_reward": 2.8,
  "std_reward": 1.2,
  "min_reward": -5.0,
  "max_reward": 10.0,
  "found_correct_email": 80,
  "read_correct_email": 85,
  "total_repeated_searches": 15,
  "total_unique_searches": 250,
  "total_searches": 265,
  "avg_turns_correct": 3.2,
  "avg_turns_idk": 4.5,
  "avg_search_attempts": 2.65,
  "num_idk": 10,
  "eval_time": 450.5,
  "control_reward_map": {
    "query_001": [2.5, 3.0, 2.8],
    "query_002": [1.5, 2.0],
    "query_003": [4.5, 5.0, 4.8, 5.2]
  }
}
```

## 字段说明

### 基本统计信息
- `step`: 训练步数（-1 表示 baseline）
- `is_baseline`: 是否为 baseline 评估
- `accuracy`: 准确率（0-1）
- `correct_answers`: 正确答案数量
- `total_samples`: 总样本数量
- `attempted_answer`: 尝试回答的数量
- `eval_time`: 评估耗时（秒）

### 奖励统计
- `avg_reward`: 平均奖励
- `median_reward`: 中位数奖励
- `std_reward`: 奖励标准差
- `min_reward`: 最小奖励
- `max_reward`: 最大奖励

### 详细指标
- `found_correct_email`: 找到正确邮件的数量
- `read_correct_email`: 读取正确邮件的数量
- `total_repeated_searches`: 重复搜索总数
- `total_unique_searches`: 唯一搜索总数
- `total_searches`: 搜索总数
- `avg_turns_correct`: 正确答案的平均轮数
- `avg_turns_idk`: "I don't know" 的平均轮数
- `avg_search_attempts`: 平均搜索尝试次数
- `num_idk`: "I don't know" 的数量

### 新增：Reward Map
- `control_reward_map`: 每个查询的奖励列表
  - Key: 查询 ID
  - Value: 该查询在 baseline 评估中的所有 rollout 奖励列表
  - 用于 beat-rate 比较（与后续评估比较）

## 使用方式

### 运行新的 Baseline 评估（保存包含 reward map）
```bash
export RUN_BASELINE_EVAL=true
python train_grpo.py --mode masked
```

### 从 JSON 加载 Baseline（包括 reward map）
```bash
export RUN_BASELINE_EVAL=false
python train_grpo.py --mode masked
```

加载时会自动：
1. 查找最新的 `baseline_eval_*.json` 文件
2. 加载统计信息和 reward map
3. 如果 JSON 中包含 `control_reward_map`，则支持 beat-rate 比较
4. 如果 JSON 中不包含 `control_reward_map`（旧格式），则不支持 beat-rate 比较

## 向后兼容性

- **旧的 JSON 文件**（没有 `control_reward_map`）仍然可以加载
- 加载旧文件时会显示警告：`⚠️ No reward map found: beat-rate comparison unavailable`
- 新运行的 baseline 评估会自动生成包含 `control_reward_map` 的 JSON

## Beat-rate 比较

当 `control_reward_map` 可用时，训练过程中的每次评估都会计算：

- **Beat Rate**: 当前模型在多少比例的查询上超过了 baseline 的最佳表现
- **Avg Delta**: 当前模型与 baseline 之间的平均奖励差异

这些指标会显示在评估日志中，并记录到 wandb（如果启用）。

