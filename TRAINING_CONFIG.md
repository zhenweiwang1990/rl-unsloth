# GRPO Training Configuration

## Training Process Overview

The training follows this process:

1. **Load a batch of 8 questions** from the dataset (with their correct answers)
2. **Run the agent 6 times** for each question using the rollout function → 48 trajectories total per step
3. **Score all 48 trajectories** with the reward function
4. **Calculate GRPO loss** using all trajectories and their rewards, then **update the model**
5. **Every 30 steps**: Run 100 validation questions and calculate accuracy
6. **Stop** when the model stops improving on the validation set (early stopping)

## Environment Variables

You can set these via environment variables or a `.env` file:

```bash
# Model Configuration
MODEL_NAME=unsloth/Qwen3-14B-Base

# Dataset Configuration
TRAIN_DATASET_SIZE=3000          # Number of training questions
EVAL_DATASET_SIZE=100            # 100 validation questions for evaluation

# Training Configuration  
PER_DEVICE_TRAIN_BATCH_SIZE=8   # 8 questions per batch
NUM_GENERATIONS=6                # 6 rollouts per question
                                 # Total: 8 × 6 = 48 trajectories per step
MAX_STEPS=200                    # Maximum training steps
LEARNING_RATE=1e-5               # Learning rate
BETA=0.01                        # KL penalty coefficient

# Agent Configuration
MAX_TURNS=4                      # Maximum conversation turns per rollout
MAX_TOKENS=4096                  # Maximum tokens per generation

# Training Control
PATIENCE=5                       # Early stopping: stop if no improvement for N evaluations

# Output
OUTPUT_DIR=outputs/grpo_masked   # Output directory for checkpoints

# Other Settings
SEED=42                          # Random seed for reproducibility
VERBOSE=false                    # Verbose logging

# OpenRouter API (for judge-based evaluation)
OPENROUTER_API_KEY=your_key_here
```

## Key Changes from Default

The following parameters have been updated to match the training process:

1. **`PER_DEVICE_TRAIN_BATCH_SIZE=8`** (was 2)
   - Now processes 8 questions per batch

2. **`EVAL_DATASET_SIZE=100`** (was 1000)
   - Uses 100 validation questions for faster evaluation

3. **`eval_steps=30`** (was 50, hardcoded in AgentGRPOTrainer)
   - Evaluates every 30 steps instead of 50

4. **`PATIENCE=5`** (new parameter)
   - Implements early stopping: training stops if no improvement for 5 consecutive evaluations

## Running Training

```bash
# Using masked mode (recommended)
python train_grpo.py --mode masked

# Or with custom environment variables
export PER_DEVICE_TRAIN_BATCH_SIZE=8
export EVAL_DATASET_SIZE=100
export PATIENCE=5
python train_grpo.py --mode masked
```

## Training Metrics

The training will automatically:
- Track best accuracy on validation set
- Save best model checkpoint when accuracy improves
- Stop early if no improvement for 5 evaluations
- Log detailed statistics including:
  - Rewards (mean, std, median, range)
  - Loss (total, policy, KL)
  - Accuracy (current, moving average, best)
  - Token statistics (trainable vs total)
  - Timing information

