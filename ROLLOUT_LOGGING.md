# Detailed Rollout Logging

This document describes the detailed rollout logging feature that records complete execution traces of each rollout during training.

## Overview

The rollout logging system creates detailed JSON files for each rollout, organized by training step and query. These logs are invaluable for:

- Debugging agent behavior
- Understanding why certain rewards were assigned
- Analyzing judge decisions
- Tracking tool call patterns
- Identifying failure modes

## Directory Structure

Logs are saved in the following structure:

```
outputs/rollout_logs/
├── step_0/
│   ├── query_0/
│   │   ├── rollout_0.json
│   │   ├── rollout_1.json
│   │   ├── rollout_2.json
│   │   └── rollout_3.json
│   ├── query_1/
│   │   ├── rollout_0.json
│   │   └── ...
│   └── ...
├── step_1/
│   └── ...
└── ...
```

- **step_X**: Training step number (0 for initial evaluation)
- **query_Y**: Query ID from the dataset
- **rollout_Z.json**: Individual rollout file (Z = 0 to num_rollouts-1)

## Log File Format

Each `rollout_Z.json` file contains:

### Query Information
- `query_id`: Unique query identifier
- `query_question`: The question asked
- `query_answer`: Ground truth answer
- `query_inbox`: Inbox address
- `correct_message_ids`: List of correct email IDs

### Configuration
- `system_prompt`: Full system prompt sent to model
- `max_turns`: Maximum allowed turns
- `policy_config`: Policy configuration dict
- `temperature`: Sampling temperature used
- `repetition_penalty`: Repetition penalty used

### Execution Trace
- `conversation_history`: Complete conversation in OpenAI format
  - System message
  - User message
  - Assistant messages with tool calls
  - Tool result messages

### Tool Calls
- `tool_calls`: Array of tool call logs, each containing:
  - `turn_number`: Which turn this occurred in (1-indexed)
  - `tool_name`: Name of tool called (`search_emails`, `read_email`, `return_final_answer`)
  - `tool_arguments`: Arguments passed to tool
  - `tool_result`: Result returned by tool
  - `is_correct_email_found`: Whether search found the correct email (for `search_emails`)
  - `is_correct_email_read`: Whether agent read the correct email (for `read_email`)
  - `result_count`: Number of results (for `search_emails`)
  - `error`: Error message if tool call failed
  - `timestamp`: ISO timestamp

### Answer and Judging
- `final_answer`: Agent's final answer (if provided)
- `final_answer_sources`: List of email IDs cited as sources
- `judge_log`: Judge evaluation details
  - `system_prompt`: System prompt sent to judge model
  - `user_prompt`: User prompt with question, ground truth, and agent answer
  - `agent_answer`: Answer provided by agent
  - `ground_truth_answer`: Ground truth answer
  - `judge_response`: Raw response from judge model
  - `is_correct`: Boolean judge decision
  - `timestamp`: ISO timestamp

### Evaluation Results
- `rubric`: Complete evaluation rubric as dict
  - `answer_correct`: Whether answer was correct
  - `sources_correct`: Whether sources were correct
  - `num_turns`: Number of turns used
  - `attempted_answer`: Whether agent attempted to answer
  - `ever_found_right_email`: Whether agent found correct email in search
  - `ever_read_right_email`: Whether agent read correct email
  - `cant_parse_tool_call`: Parse errors
  - `bad_tool_call_name`: Invalid tool names
  - `bad_tool_call_args`: Invalid tool arguments
  - `ran_out_of_turns`: Whether agent exhausted turn budget
  - `returned_i_dont_know`: Whether agent said "I don't know"
  - `num_sources`: Number of sources cited
  - `ever_tried_to_read_invalid_email`: Whether agent tried invalid email
  - `total_input_tokens`: Total input tokens used
  - `total_output_tokens`: Total output tokens used
  - Search effort metrics (repetitions, retries, etc.)
- `reward`: Final reward value

### Metadata
- `step`: Training step number
- `rollout_index`: Index of this rollout (0 to num_rollouts-1)
- `start_time`: ISO timestamp when rollout started
- `end_time`: ISO timestamp when rollout finished
- `duration_seconds`: Total duration in seconds
- `total_input_tokens`: Total input tokens
- `total_output_tokens`: Total output tokens

## Enabling Detailed Logging

### In Training Script

Add the `enable_detailed_logging=True` parameter when creating the trainer:

```python
from grpo.trainer import AgentGRPOTrainer

trainer = AgentGRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_queries=train_queries,
    eval_queries=eval_queries,
    policy_config=policy_config,
    openai_client=openai_client,
    # ... other parameters ...
    enable_detailed_logging=True,  # Enable detailed logging
)

# Run training - logs will be saved to outputs/rollout_logs/
trainer.train()
```

### In train_grpo.py

The training script can be modified to accept a command-line flag:

```bash
# Example (after adding argparse flag):
python train_grpo.py --enable-detailed-logging
```

Or modify the script directly:

```python
# In train_grpo.py
trainer = AgentGRPOTrainer(
    # ... parameters ...
    enable_detailed_logging=True,  # Add this line
)
```

## Example Log Entry

Here's a simplified example of what a log file looks like:

```json
{
  "query_id": "query_123",
  "query_question": "What is the project deadline?",
  "query_answer": "March 15, 2024",
  "query_inbox": "john@example.com",
  "correct_message_ids": ["msg_456"],
  
  "system_prompt": "You are an email assistant...",
  "max_turns": 10,
  "temperature": 0.7,
  "repetition_penalty": 1.0,
  
  "tool_calls": [
    {
      "turn_number": 1,
      "tool_name": "search_emails",
      "tool_arguments": {
        "keywords": ["project", "deadline"]
      },
      "tool_result": [
        {
          "message_id": "msg_456",
          "subject": "Project Timeline Update",
          "from": "manager@example.com"
        }
      ],
      "is_correct_email_found": true,
      "result_count": 1,
      "timestamp": "2024-01-15T10:30:00"
    },
    {
      "turn_number": 2,
      "tool_name": "read_email",
      "tool_arguments": {
        "message_id": "msg_456"
      },
      "tool_result": {
        "subject": "Project Timeline Update",
        "body": "The project deadline is March 15, 2024.",
        "from": "manager@example.com"
      },
      "is_correct_email_read": true,
      "timestamp": "2024-01-15T10:30:05"
    },
    {
      "turn_number": 3,
      "tool_name": "return_final_answer",
      "tool_arguments": {
        "answer": "March 15, 2024",
        "source_message_ids": ["msg_456"]
      },
      "tool_result": {
        "status": "Final answer submitted"
      },
      "timestamp": "2024-01-15T10:30:10"
    }
  ],
  
  "final_answer": "March 15, 2024",
  "final_answer_sources": ["msg_456"],
  
  "judge_log": {
    "system_prompt": "You will be given a question and two different answers...",
    "user_prompt": "Question: What is the project deadline?\nCorrect answer: March 15, 2024\nAI answer: March 15, 2024",
    "agent_answer": "March 15, 2024",
    "ground_truth_answer": "March 15, 2024",
    "judge_response": "True",
    "is_correct": true,
    "timestamp": "2024-01-15T10:30:15"
  },
  
  "rubric": {
    "answer_correct": true,
    "sources_correct": true,
    "num_turns": 3,
    "attempted_answer": true,
    "ever_found_right_email": true,
    "ever_read_right_email": true,
    "num_repeated_searches": 0,
    "num_zero_result_searches": 0
  },
  
  "reward": 2.0,
  
  "step": 0,
  "rollout_index": 0,
  "start_time": "2024-01-15T10:30:00",
  "end_time": "2024-01-15T10:30:15",
  "duration_seconds": 15.0,
  "total_input_tokens": 1250,
  "total_output_tokens": 180
}
```

## Analyzing Logs

### Python Script Example

```python
import json
from pathlib import Path

# Load a specific rollout
log_path = Path("outputs/rollout_logs/step_5/query_0/rollout_0.json")
with open(log_path) as f:
    log = json.load(f)

# Analyze tool calls
for tool_call in log["tool_calls"]:
    print(f"Turn {tool_call['turn_number']}: {tool_call['tool_name']}")
    if tool_call['tool_name'] == 'search_emails':
        print(f"  Found correct email: {tool_call['is_correct_email_found']}")
        print(f"  Results: {tool_call['result_count']}")

# Check judge decision
if log.get("judge_log"):
    judge = log["judge_log"]
    print(f"\nJudge Decision: {judge['is_correct']}")
    print(f"Agent Answer: {judge['agent_answer']}")
    print(f"Ground Truth: {judge['ground_truth_answer']}")

# View reward and metrics
print(f"\nReward: {log['reward']}")
print(f"Turns Used: {log['rubric']['num_turns']}")
print(f"Answer Correct: {log['rubric']['answer_correct']}")
```

### Aggregating Statistics

```python
from pathlib import Path
import json
from collections import defaultdict

# Analyze all logs from a step
step_dir = Path("outputs/rollout_logs/step_5")
stats = defaultdict(list)

for rollout_file in step_dir.glob("*/rollout_*.json"):
    with open(rollout_file) as f:
        log = json.load(f)
    
    stats["rewards"].append(log["reward"])
    stats["turns"].append(log["rubric"]["num_turns"])
    stats["correct"].append(log["rubric"]["answer_correct"])
    stats["found_email"].append(log["rubric"]["ever_found_right_email"])

# Print summary
print(f"Step 5 Statistics:")
print(f"  Average Reward: {sum(stats['rewards']) / len(stats['rewards']):.3f}")
print(f"  Average Turns: {sum(stats['turns']) / len(stats['turns']):.1f}")
print(f"  Accuracy: {sum(stats['correct']) / len(stats['correct']) * 100:.1f}%")
print(f"  Found Correct Email: {sum(stats['found_email']) / len(stats['found_email']) * 100:.1f}%")
```

## Performance Considerations

- **Storage**: Each log file is typically 5-50 KB depending on conversation length
- **I/O Impact**: Logs are written asynchronously at the end of each rollout
- **Recommended**: Enable only when needed for debugging or analysis
- **Disable in production**: Set `enable_detailed_logging=False` for production runs

## Use Cases

1. **Debugging Training Issues**: When accuracy plateaus or degrades, examine logs to understand agent behavior
2. **Judge Analysis**: Review judge decisions to ensure correctness
3. **Tool Usage Patterns**: Identify if agent is using tools efficiently
4. **Error Analysis**: Find systematic errors in tool calls or reasoning
5. **Reward Engineering**: Validate that rewards align with desired behavior
6. **Model Comparison**: Compare logs across different checkpoints

## Notes

- Logs are saved immediately after each rollout completes
- If training crashes, logs up to the last completed rollout are preserved
- Log directory can be changed by modifying `get_rollout_logger()` call in code
- Logs from evaluation rollouts are also saved (with `step` indicating the checkpoint number)

