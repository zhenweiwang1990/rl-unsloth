# Rollout Logging Quick Start

This guide shows you how to enable and use detailed rollout logging.

## Enable During Training

Simply add the `--enable-detailed-logging` flag when running training:

```bash
python train_grpo.py --mode masked --enable-detailed-logging
```

This will save detailed JSON logs for every rollout to `outputs/rollout_logs/`.

## Directory Structure After Training

After training with logging enabled, you'll see:

```
outputs/rollout_logs/
├── step_0/           # Initial evaluation
│   ├── query_0/
│   │   ├── rollout_0.json
│   │   ├── rollout_1.json
│   │   └── rollout_2.json
│   ├── query_1/
│   └── ...
├── step_10/          # After 10 training steps
│   └── ...
└── step_20/
    └── ...
```

## Quick Analysis

### View Latest Step Summary

```bash
python scripts/analyze_rollout_logs.py
```

This will show statistics for the latest training step.

### Compare Multiple Steps

```bash
# Compare steps 0, 10, and 20
python scripts/analyze_rollout_logs.py --compare "0,10,20"
```

### Analyze All Steps

```bash
python scripts/analyze_rollout_logs.py --all
```

### View Failure Cases

```bash
# Show failure cases from step 10
python scripts/analyze_rollout_logs.py --step 10 --failures
```

## Inspect Individual Logs

### Load and Inspect in Python

```python
import json
from pathlib import Path

# Load a specific rollout
log_path = Path("outputs/rollout_logs/step_10/query_0/rollout_0.json")
with open(log_path) as f:
    log = json.load(f)

# Print query info
print(f"Question: {log['query_question']}")
print(f"Answer: {log['query_answer']}")

# Print tool call sequence
print("\nTool Calls:")
for tc in log['tool_calls']:
    print(f"  Turn {tc['turn_number']}: {tc['tool_name']}")
    
    if tc['tool_name'] == 'search_emails':
        print(f"    → Found {tc['result_count']} emails")
        print(f"    → Correct email in results: {tc['is_correct_email_found']}")
    
    elif tc['tool_name'] == 'read_email':
        print(f"    → Read correct email: {tc['is_correct_email_read']}")
    
    elif tc['tool_name'] == 'return_final_answer':
        print(f"    → Answer: {log['final_answer'][:60]}...")

# Print judge decision
if log.get('judge_log'):
    judge = log['judge_log']
    print(f"\nJudge Decision: {'✓ CORRECT' if judge['is_correct'] else '✗ WRONG'}")
    print(f"Agent: {judge['agent_answer'][:60]}...")
    print(f"Truth: {judge['ground_truth_answer'][:60]}...")

# Print reward breakdown
print(f"\nReward: {log['reward']:.3f}")
print(f"Turns Used: {log['rubric']['num_turns']}")
print(f"Answer Correct: {log['rubric']['answer_correct']}")
print(f"Sources Correct: {log['rubric']['sources_correct']}")
```

### View as Pretty JSON

```bash
cat outputs/rollout_logs/step_10/query_0/rollout_0.json | python -m json.tool | less
```

### Search for Specific Patterns

Find all rollouts where agent gave up early:

```bash
grep -r '"gave_up_too_early": true' outputs/rollout_logs/
```

Find all rollouts with high rewards:

```python
import json
from pathlib import Path

high_reward_rollouts = []

for log_file in Path("outputs/rollout_logs").glob("*/*/rollout_*.json"):
    with open(log_file) as f:
        log = json.load(f)
    
    if log['reward'] >= 1.5:
        high_reward_rollouts.append({
            'file': str(log_file),
            'reward': log['reward'],
            'question': log['query_question'][:60],
        })

# Sort by reward
high_reward_rollouts.sort(key=lambda x: x['reward'], reverse=True)

print(f"Found {len(high_reward_rollouts)} high-reward rollouts:")
for rollout in high_reward_rollouts[:10]:
    print(f"  {rollout['reward']:.3f} - {rollout['question']}... ({rollout['file']})")
```

## Common Analysis Tasks

### 1. Find Why Model Is Repeating Searches

```python
import json
from pathlib import Path

for log_file in Path("outputs/rollout_logs/step_20").glob("*/rollout_*.json"):
    with open(log_file) as f:
        log = json.load(f)
    
    if log['rubric']['num_repeated_searches'] > 0:
        print(f"\nQuery: {log['query_question'][:60]}...")
        print(f"Repeated searches: {log['rubric']['num_repeated_searches']}")
        
        # Show search sequence
        searches = [tc for tc in log['tool_calls'] if tc['tool_name'] == 'search_emails']
        for i, search in enumerate(searches, 1):
            keywords = search['tool_arguments'].get('keywords', [])
            print(f"  Search {i}: keywords={keywords}, results={search['result_count']}")
```

### 2. Analyze Judge Disagreements

```python
import json
from pathlib import Path

disagreements = []

for log_file in Path("outputs/rollout_logs").glob("*/*/rollout_*.json"):
    with open(log_file) as f:
        log = json.load(f)
    
    # Check if answer was attempted but judge said wrong
    if log.get('judge_log'):
        judge = log['judge_log']
        if not judge['is_correct']:
            disagreements.append({
                'file': str(log_file),
                'question': log['query_question'],
                'agent_answer': judge['agent_answer'],
                'ground_truth': judge['ground_truth_answer'],
            })

print(f"Found {len(disagreements)} incorrect answers")
for d in disagreements[:5]:
    print(f"\nQuestion: {d['question'][:60]}...")
    print(f"Agent: {d['agent_answer'][:60]}...")
    print(f"Truth: {d['ground_truth'][:60]}...")
```

### 3. Track Tool Usage Evolution

```python
import json
from pathlib import Path
from collections import Counter

def analyze_tool_usage(step):
    tool_counts = Counter()
    
    step_dir = Path(f"outputs/rollout_logs/step_{step}")
    for log_file in step_dir.glob("*/rollout_*.json"):
        with open(log_file) as f:
            log = json.load(f)
        
        for tc in log['tool_calls']:
            tool_counts[tc['tool_name']] += 1
    
    return tool_counts

# Compare tool usage across steps
for step in [0, 10, 20]:
    counts = analyze_tool_usage(step)
    total = sum(counts.values())
    
    print(f"\nStep {step} Tool Usage:")
    for tool, count in counts.most_common():
        print(f"  {tool:25s} {count:4d} ({count/total*100:5.1f}%)")
```

## Performance Impact

- **Storage**: ~10-50 KB per rollout (depends on conversation length)
- **Time**: Negligible (<1ms per rollout to write JSON)
- **Recommendation**: Enable during development and debugging, disable for production

## Tips

1. **Disk Space**: Logs can grow large. For a 200-step training with 4 queries × 4 rollouts per step:
   - ~3,200 log files
   - ~50-150 MB total (depending on conversation lengths)

2. **Periodic Cleanup**: Archive old logs periodically:
   ```bash
   # Archive logs from steps 0-99
   tar czf rollout_logs_steps_0-99.tar.gz outputs/rollout_logs/step_{0..99}
   rm -rf outputs/rollout_logs/step_{0..99}
   ```

3. **Selective Logging**: To log only specific steps, modify the code to check `training_step`:
   ```python
   # In grpo/trainer.py, add condition:
   enable_logging = self.enable_detailed_logging and (step_num % 10 == 0)
   ```

4. **Custom Analysis**: Create your own analysis scripts using the log structure documented in `ROLLOUT_LOGGING.md`

