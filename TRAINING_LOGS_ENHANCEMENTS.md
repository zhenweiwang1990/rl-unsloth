# Training Logs Enhancements

本文档总结了对 GRPO 训练过程日志输出的增强功能。

## 实现的功能

### 1. Group Statistics 中的 Turn-by-Turn Advantage 表格

**位置**: `grpo/trainer.py` - `_print_turn_advantage_table()` 和 `_compute_turn_advantages()`

**功能描述**:
- 在每个 group 的详细日志中，添加了一个表格展示每个 rollout 的每一轮的 advantage 变化
- 表格格式：行为 rollout 编号，列为 turn 编号
- 每个单元格显示该 rollout 在该 turn 的 advantage 值

**示例输出**:
```
────────────────────────────────────────────────────────────────────────────────
📊 Turn-by-Turn Advantage Table:
────────────────────────────────────────────────────────────────────────────────
Rollout | Turn  1 | Turn  2 | Turn  3 | Turn  4 |
────────────────────────────────────────────────────────────────────────────────
    1  | +0.350 | +0.420 | +0.800 | +1.200 |
    2  | -0.200 | -0.150 | +0.300 | +0.850 |
    3  | +0.150 | +0.250 | -0.300 | -0.450 |
────────────────────────────────────────────────────────────────────────────────
```

### 2. Eval 结束后的详细统计和保存

**位置**: `grpo/trainer.py` - `evaluate()` 方法

**新增统计指标**:
- ✅ 正确率 (已有)
- ✅ 作答次数/总数
- ✅ 总搜索引用正确数 (ever_found_right_email)
- ✅ 总 read 引用正确数 (ever_read_right_email)
- ✅ 重复搜索数 (num_repeated_searches)
- ✅ 正确题目平均轮次
- ✅ I don't know 题目平均轮次
- ✅ 平均尝试搜索次数

**保存位置**: `{output_dir}/eval_logs/eval_step_XXXX.json`

**示例输出**:
```
📊 Detailed Rubric Statistics:
   Attempted answers: 95/100 (95.0%)
   Found correct email: 87/100 (87.0%)
   Read correct email: 82/100 (82.0%)
   Repeated searches: 15 (total: 245, unique: 230)
   Avg turns (correct): 3.45 turns
   Avg turns (I don't know): 4.20 turns (count: 5)
   Avg search attempts: 2.45
   
💾 Eval stats saved to: outputs/grpo_masked/eval_logs/eval_step_0010.json
```

**保存的 JSON 内容**:
```json
{
  "step": 10,
  "accuracy": 0.85,
  "correct_answers": 85,
  "total_samples": 100,
  "attempted_answer": 95,
  "avg_reward": 0.723,
  "median_reward": 0.850,
  "std_reward": 0.312,
  "min_reward": -1.200,
  "max_reward": 1.500,
  "found_correct_email": 87,
  "read_correct_email": 82,
  "total_repeated_searches": 15,
  "total_unique_searches": 230,
  "total_searches": 245,
  "avg_turns_correct": 3.45,
  "avg_turns_idk": 4.20,
  "avg_search_attempts": 2.45,
  "num_idk": 5,
  "eval_time": 234.5,
  "beat_rate": 0.78
}
```

### 3. Step 结束时的 Group 总结信息

**位置**: `grpo/trainer.py` - `training_step()` 方法

**新增统计**:
- 总 group 数量
- 保留用于训练的 group 数量
- 因组内无差异（低方差）过滤掉的 group 数量
- 没有耗尽 turn 就提前退出的 rollout 数量
- Rollout 时间和训练时间
- Token 统计

**示例输出（详细模式）**:
```
────────────────────────────────────────────────────────────────────────────────
STEP PHASE 4: BACKPROPAGATION & OPTIMIZATION
────────────────────────────────────────────────────────────────────────────────
✓ Grad norm (clipped): 0.8234
✓ Tokens trained: 12,345
✓ Total loss: 0.3456

📊 Group Summary:
  - Total groups: 8
  - Groups kept for training: 6
  - Groups filtered (low variance): 2
  - Rollouts that finished early (didn't exhaust turns): 18/48
  - Total rollout time: 45.3s
  - Total training time: 2.1s
  - Total tokens: 45,678
  - Trainable tokens: 12,345 (27.0%)
────────────────────────────────────────────────────────────────────────────────
```

**示例输出（简洁模式）**:
```
📍 Step 10/200
📊 Collecting 3 rollouts for 8 queries...
  Groups: 6/8 kept (2 filtered), 18/48 finished early
```

## 代码修改总结

### 修改的文件

1. **`grpo/trainer.py`**:
   - 在 `TrajectorySample` 中添加了 `turn_advantages` 字段
   - 新增 `_compute_turn_advantages()` 方法来计算每轮的 advantage
   - 新增 `_print_turn_advantage_table()` 方法来打印表格
   - 修改 `_print_group_details()` 调用表格打印方法
   - 修改 `compute_advantages()` 来计算并保存 turn_advantages
   - 增强 `evaluate()` 方法添加详细统计并保存到 JSON 文件
   - 增强 `training_step()` 方法添加 group 总结信息
   - 在 wandb 日志中添加新的指标

2. **`grpo/utils.py`**:
   - 在 `TrainingMetrics` 中添加了三个新字段：
     - `groups_kept`: 保留的 group 数量
     - `groups_filtered`: 过滤的 group 数量
     - `num_early_exit`: 提前退出的 rollout 数量

## Wandb 新增指标

### 训练阶段 (train/*)
- `train/groups_kept`: 保留的 group 数量
- `train/groups_filtered`: 过滤的 group 数量
- `train/num_early_exit`: 提前退出的 rollout 数量

### 评估阶段 (eval/*)
- `eval/total_repeated_searches`: 总重复搜索数
- `eval/total_unique_searches`: 总唯一搜索数
- `eval/total_searches`: 总搜索次数
- `eval/avg_turns_correct`: 正确题目平均轮次
- `eval/avg_turns_idk`: I don't know 题目平均轮次
- `eval/avg_search_attempts`: 平均搜索尝试次数
- `eval/num_idk`: I don't know 的题目数量

## 使用方法

### 启用详细日志

在环境变量中设置 `VERBOSE=true` 来启用详细的日志输出，包括：
- Turn-by-Turn Advantage 表格
- 详细的 rollout 信息
- 完整的 group 总结

```bash
export VERBOSE=true
python train_grpo.py --mode masked
```

### 查看保存的 Eval 日志

```bash
# 查看最新的 eval 日志
cat outputs/grpo_masked/eval_logs/eval_step_*.json | jq .

# 查看所有 eval 的准确率趋势
for f in outputs/grpo_masked/eval_logs/eval_step_*.json; do
    echo -n "$(basename $f): "
    jq -r '.accuracy' $f
done
```

### 分析训练过程

评估日志文件可以用于后续分析，例如：
- 追踪准确率变化
- 分析搜索策略的效率
- 了解模型何时学会提前退出
- 比较不同 checkpoint 的详细表现

## 注意事项

1. **Turn Advantage 计算**：advantage 值是基于 process-based rewards，每个 turn 根据其对最终目标的贡献独立计算

2. **过滤的 Group**：方差过低的 group（`std < min_group_std`）会被过滤，因为它们对梯度更新没有贡献

3. **提前退出**：rollout 可能因为以下原因提前退出：
   - 成功找到答案并返回
   - 返回 "I don't know"
   - 遇到错误（格式错误、非法工具调用等）

4. **文件大小**：eval_logs 目录会随着训练步数增长，定期清理旧的 eval 日志或设置合适的 `eval_steps` 间隔

## 示例分析脚本

```python
import json
import glob
from pathlib import Path

def analyze_eval_logs(output_dir):
    """分析 eval 日志文件"""
    eval_logs = sorted(glob.glob(f"{output_dir}/eval_logs/eval_step_*.json"))
    
    for log_file in eval_logs:
        with open(log_file) as f:
            data = json.load(f)
        
        print(f"Step {data['step']:4d}: "
              f"Acc={data['accuracy']*100:5.1f}%, "
              f"Found={data['found_correct_email']}/{data['total_samples']}, "
              f"Read={data['read_correct_email']}/{data['total_samples']}, "
              f"Repeats={data['total_repeated_searches']}")

# 使用示例
analyze_eval_logs("outputs/grpo_masked")
```

## 版本信息

- 修改日期: 2024-11-20
- 影响范围: GRPO masked mode 训练
- 兼容性: 向后兼容，不影响现有训练流程

