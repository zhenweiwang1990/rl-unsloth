# Email Agent GRPO Training with Unsloth

This project implements GRPO (Group Relative Policy Optimization) training for an email search agent using [unsloth](https://github.com/unslothai/unsloth) for fast LoRA fine-tuning and [TRL](https://github.com/huggingface/trl) for the GRPO algorithm.

## Overview

The email agent is trained to search through a database of emails (Enron dataset) and answer questions about email content using a tool-calling interface. The agent learns through reinforcement learning, receiving rewards based on whether it correctly answers queries.

### Key Features

- **Fast Training with Unsloth**: Uses unsloth's optimized LoRA implementation for 2x faster training with less memory
- **GRPO Algorithm**: Implements Group Relative Policy Optimization from the DeepSeekMath paper
- **Tool Use**: Agent learns to use search and read tools to find information
- **Reward Shaping**: Complex reward function with partial credit for intermediate progress
- **Base Model**: Qwen3-14B-Base with LoRA fine-tuning

## Architecture

```
email_agent/
├── data/
│   ├── types.py              # Data models (Email, SyntheticQuery)
│   ├── query_loader.py       # Load queries from HuggingFace
│   └── local_email_db.py     # Generate SQLite database
├── tools.py                  # Email search and read tools
├── config.py                 # Training and policy configuration
└── rollout.py                # Agent rollout and reward calculation

train_grpo.py                 # Main training script
train_grpo_advanced.py        # Advanced training with custom rollout
scripts/
├── setup.sh                  # Initial setup
├── generate_database.sh      # Generate email database
└── run_training.sh           # Run training
```

## Setup

### 1. Install Dependencies

```bash
./scripts/setup.sh
```

This will:
- Install Python dependencies from `requirements.txt`
- Create a `.env` file for configuration
- Optionally generate the email database

### 2. Configure Environment

Edit `.env` and add your OpenAI API key (required for judge model):

```bash
OPENAI_API_KEY=your_key_here
```

### 3. Generate Email Database

If not done during setup:

```bash
./scripts/generate_database.sh
```

This downloads the Enron email dataset and creates a local SQLite database (~10-15 minutes).

## Training

### Quick Start

```bash
./scripts/run_training.sh
```

Or directly with Python:

```bash
python train_grpo.py
```

For multi-GPU training with accelerate:

```bash
accelerate launch train_grpo.py
```

### Configuration

Key parameters in `.env`:

```bash
# Training
TRAIN_DATASET_SIZE=1000
EVAL_DATASET_SIZE=100
MAX_STEPS=1000
LEARNING_RATE=1e-5
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4

# Agent
MAX_TURNS=10
MAX_TOKENS=2048

# Output
OUTPUT_DIR=outputs/grpo
RUN_NAME=email_agent_grpo
```

## How It Works

### 1. Data Loading

Queries are loaded from the [corbt/enron_emails_sample_questions](https://huggingface.co/datasets/corbt/enron_emails_sample_questions) dataset. Each query contains:
- Question about email content
- Ground truth answer
- Message IDs of relevant emails

### 2. Agent Loop

For each query, the agent:
1. Receives the question and system prompt
2. Calls tools (search_emails, read_email) to find information
3. Returns a final answer with sources
4. Continues for up to `MAX_TURNS` turns

### 3. Tools

- **search_emails**: Full-text search with filters (keywords, sender, recipient, date)
- **read_email**: Retrieve complete email by message_id
- **return_final_answer**: Submit answer and sources

### 4. Reward Calculation

Rewards range from -2 to +2:

- **-2 to -1**: Formatting errors (can't parse tool calls)
- **-1 to 0**: Wrong answer
- **0 to 1**: No answer or "I don't know"
- **1 to 2**: Correct answer (bonus for correct sources, fewer turns)

Partial credit awarded for:
- Finding the right email in search results (+0.1)
- Reading the right email (+0.1)
- Not attempting to read invalid emails (+0.1)
- Correct source citations (+0.1)

### 5. GRPO Algorithm

GRPO generates multiple completions per prompt and uses group-relative advantage:

```
advantage_i = (reward_i - mean(rewards)) / std(rewards)
```

This makes rewards more comparable across different queries.

## Model Details

- **Base Model**: [unsloth/Qwen3-14B-Base](https://huggingface.co/unsloth/Qwen3-14B-Base)
- **Method**: LoRA fine-tuning (rank 16, alpha 16)
- **Quantization**: 4-bit quantization with unsloth
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Evaluation

The model is evaluated using GPT-4o as a judge to determine if answers are semantically correct. Metrics tracked:
- Answer correctness
- Source correctness
- Number of turns
- Various error types (bad tool calls, invalid emails, etc.)

## Advanced Usage

### Custom Reward Function

Edit `email_agent/rollout.py` and modify `calculate_reward()`:

```python
def calculate_reward(policy_config: PolicyConfig, rubric: EvaluationRubric) -> float:
    # Your custom reward logic here
    return reward
```

### Custom Rollout Function

Use `train_grpo_advanced.py` for more control over the rollout process:

```bash
python train_grpo_advanced.py
```

### Training on Subset

Modify dataset size in code or `.env`:

```python
train_queries = load_synthetic_queries(
    split="train",
    limit=100,  # Smaller for testing
    shuffle=True,
)
```

## Docker Support

Build and run with Docker:

```bash
# Build
docker build -t email-agent-grpo -f Dockerfile .

# Run training
docker run --gpus all -v $(pwd)/outputs:/workspace/outputs email-agent-grpo
```

## Performance Tips

1. **Multi-GPU Training**: Use `accelerate` for distributed training across multiple GPUs
2. **Gradient Accumulation**: Increase `GRADIENT_ACCUMULATION_STEPS` if memory is limited
3. **Mixed Precision**: BF16 is automatically enabled on supported GPUs
4. **Batch Size**: Start with batch_size=1 and increase if memory allows
5. **Max Turns**: Reduce `MAX_TURNS` to speed up rollouts during development

## Troubleshooting

### Database Connection Error

Make sure the database is generated:
```bash
./scripts/generate_database.sh
```

### Out of Memory

Reduce batch size or increase gradient accumulation:
```bash
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=8
```

### Judge API Errors

Check your OpenAI API key in `.env`:
```bash
echo $OPENAI_API_KEY
```

## References

- [DeepSeekMath Paper](https://arxiv.org/abs/2402.03300) - Original GRPO algorithm
- [TRL GRPO Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) - HuggingFace implementation
- [Unsloth](https://github.com/unslothai/unsloth) - Fast LoRA training
- [Enron Email Dataset](https://huggingface.co/datasets/corbt/enron-emails) - Training data

## License

Apache 2.0

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{email-agent-grpo,
  title={Email Agent GRPO Training with Unsloth},
  year={2025},
  publisher={GitHub},
  url={https://github.com/your-repo/rl-unsloth}
}
```

