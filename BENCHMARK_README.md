# Benchmark 使用说明

本文档说明如何对 Email Agent 模型进行 benchmark 评估。

## 快速开始

### 使用 Docker（推荐）

```bash
# 基础 benchmark（100 个查询）
./scripts/run_benchmark.sh outputs/grpo/final

# 自定义配置
RUN_ID=002 TEST_SET_SIZE=200 VERBOSE=true ./scripts/run_benchmark.sh outputs/grpo/final

# Benchmark 基础模型（未微调）
./scripts/run_benchmark.sh
```

### 直接运行

```bash
# 需要先设置环境变量
export OPENAI_API_KEY=sk-...
export EMAIL_DB_PATH=data/enron_emails.db

# 运行 benchmark
python benchmark.py --model-path outputs/grpo/final --limit 100 --verbose

# Benchmark 基础模型
python benchmark.py --limit 100
```

## 配置参数

### 环境变量

```bash
# 运行标识
RUN_ID=001              # 默认: 001

# 测试集大小
TEST_SET_SIZE=100       # 默认: 100

# 详细日志
VERBOSE=false           # 默认: false

# Agent 配置
MAX_TURNS=10            # 默认: 10
MAX_TOKENS=2048         # 默认: 2048

# OpenAI API（用于 judge）
OPENAI_API_KEY=sk-...   # 必需
```

### 命令行参数

```bash
python benchmark.py \
  --model-path outputs/grpo/final \  # 模型路径（可选）
  --limit 100 \                      # 查询数量
  --output results.csv \             # 输出文件
  --verbose                          # 详细日志
```

## 输出结果

### CSV 文件

结果保存为 `benchmark_results_{RUN_ID}.csv`，包含以下列：

| 列名 | 说明 |
|------|------|
| `query_id` | 查询 ID |
| `question` | 问题（截断到 100 字符） |
| `answer` | 正确答案（截断到 100 字符） |
| `reward` | 奖励分数（-2 到 +2） |
| `answer_correct` | 答案是否正确（0/1） |
| `sources_correct` | Sources 是否正确（0/1） |
| `num_turns` | 使用的轮数 |
| `attempted_answer` | 是否尝试回答（0/1） |
| `ever_found_right_email` | 是否找到正确邮件（0/1） |
| `ever_read_right_email` | 是否读取正确邮件（0/1） |
| `ran_out_of_turns` | 是否用完轮数（0/1） |
| `returned_i_dont_know` | 是否返回"不知道"（0/1） |
| `cant_parse_tool_call` | 无法解析工具调用（0/1） |
| `bad_tool_call_name` | 错误的工具名（0/1） |
| `bad_tool_call_args` | 错误的工具参数（0/1） |
| `duration_seconds` | 查询耗时（秒） |

### 控制台输出

```
==============================================================
BENCHMARK RESULTS
==============================================================
Total queries: 100
Total duration: 1234.56s
Avg duration per query: 12.35s

Average reward: 0.456
Answer accuracy: 45.0%
Source accuracy: 38.0%
Average turns: 3.42

Attempted answers: 85/100
Found right email: 72/100
Read right email: 68/100

Ran out of turns: 12
Returned 'I don't know': 3
Parse errors: 2
Bad tool name: 1
Bad tool args: 3
==============================================================
```

## 指标说明

### 核心指标

- **Average reward**: 平均奖励分数（-2 到 +2）
- **Answer accuracy**: 答案正确率
- **Source accuracy**: Source 引用正确率
- **Average turns**: 平均使用轮数

### 行为指标

- **Attempted answers**: 尝试回答的数量
- **Found right email**: 搜索到正确邮件的数量
- **Read right email**: 读取正确邮件的数量

### 错误指标

- **Ran out of turns**: 用完最大轮数
- **Returned 'I don't know'**: 返回"不知道"
- **Parse errors**: 无法解析工具调用
- **Bad tool name**: 工具名错误
- **Bad tool args**: 工具参数错误

## 使用场景

### 1. 评估训练效果

在不同训练 checkpoint 上运行 benchmark：

```bash
# Checkpoint 100
./scripts/run_benchmark.sh outputs/grpo/checkpoint-100

# Checkpoint 500
./scripts/run_benchmark.sh outputs/grpo/checkpoint-500

# Final model
./scripts/run_benchmark.sh outputs/grpo/final
```

### 2. 对比基础模型

```bash
# 基础模型（未微调）
RUN_ID=base ./scripts/run_benchmark.sh

# 微调后模型
RUN_ID=finetuned ./scripts/run_benchmark.sh outputs/grpo/final
```

### 3. 不同配置测试

```bash
# 更多轮数
MAX_TURNS=15 RUN_ID=turns15 ./scripts/run_benchmark.sh outputs/grpo/final

# 更少轮数
MAX_TURNS=5 RUN_ID=turns5 ./scripts/run_benchmark.sh outputs/grpo/final
```

### 4. 大规模评估

```bash
# 完整测试集
TEST_SET_SIZE=1000 RUN_ID=full ./scripts/run_benchmark.sh outputs/grpo/final
```

## 分析结果

### 使用 Python

```python
import polars as pl

# 加载结果
df = pl.read_csv("benchmark_results_001.csv")

# 基础统计
print(df.describe())

# 按奖励分组
print(df.group_by("reward").agg([
    pl.count("query_id").alias("count"),
    pl.col("answer_correct").mean().alias("accuracy")
]))

# 找出失败的案例
failed = df.filter(pl.col("answer_correct") == 0)
print(failed.select(["query_id", "question", "reward", "num_turns"]))
```

### 使用命令行工具

```bash
# 查看统计
csvstat benchmark_results_001.csv

# 过滤正确答案
csvgrep -c answer_correct -m 1 benchmark_results_001.csv | csvlook

# 按 reward 排序
csvsort -c reward benchmark_results_001.csv | csvlook
```

## 与 Evaluation 的区别

| 特性 | Evaluation | Benchmark |
|------|-----------|-----------|
| 目的 | 快速验证 | 详细分析 |
| 输出 | 控制台 | CSV + 控制台 |
| 指标 | 基础指标 | 完整指标 |
| 用途 | 开发调试 | 正式评估 |

推荐流程：
1. 开发时使用 `eval.py` 快速验证
2. 重要节点使用 `benchmark.py` 详细评估
3. 最终发布前运行完整 benchmark

## 性能优化

### 加快 Benchmark

```bash
# 减少查询数量
TEST_SET_SIZE=50 ./scripts/run_benchmark.sh

# 使用 GPU
# 自动检测和使用

# 并行处理（TODO）
# 未来版本支持
```

### 降低成本

```bash
# 使用简单 reward（不调用 GPT-4o judge）
export STUPID_SIMPLE_REWARD_FN=true
./scripts/run_benchmark.sh
```

## 故障排除

### OpenAI API 错误

```bash
# 检查 API key
echo $OPENAI_API_KEY

# 或在 .env 中设置
vim .env
```

### 数据库错误

```bash
# 检查数据库
ls -lh data/enron_emails.db

# 重新生成
./scripts/generate_database.sh
```

### GPU 内存不足

```bash
# 使用 CPU
# Docker 会自动处理，但会很慢
```

## 最佳实践

1. **定期 Benchmark**: 每个重要 checkpoint 运行一次
2. **保存结果**: 使用不同的 RUN_ID 保存历史结果
3. **版本控制**: 在 git commit 中记录 benchmark 结果
4. **分析趋势**: 比较不同版本的结果，识别改进方向
5. **记录配置**: 在结果文件名或 git tag 中记录训练配置

## 示例工作流

```bash
# 1. 训练模型
./scripts/run_training.sh

# 2. Benchmark checkpoint
for step in 100 200 500 1000; do
  RUN_ID=step${step} ./scripts/run_benchmark.sh outputs/grpo/checkpoint-${step}
done

# 3. Benchmark final model
RUN_ID=final TEST_SET_SIZE=500 ./scripts/run_benchmark.sh outputs/grpo/final

# 4. 分析结果
python -c "
import polars as pl
import matplotlib.pyplot as plt

# 加载所有结果
results = {}
for step in [100, 200, 500, 1000]:
    df = pl.read_csv(f'benchmark_results_step{step}.csv')
    results[step] = df['answer_correct'].mean()

# 绘制趋势
plt.plot(results.keys(), results.values())
plt.xlabel('Training Step')
plt.ylabel('Accuracy')
plt.title('Model Performance Over Training')
plt.savefig('benchmark_trend.png')
print('Saved to benchmark_trend.png')
"

# 5. Git commit
git add benchmark_results_*.csv
git commit -m "Add benchmark results for training run"
```

## 参考

- [eval.py](eval.py) - 简单评估脚本
- [train_grpo.py](train_grpo.py) - 训练脚本
- [email_agent/rollout.py](email_agent/rollout.py) - Reward 计算逻辑

