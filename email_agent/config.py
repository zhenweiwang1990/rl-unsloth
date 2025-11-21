"""Configuration classes for Email Agent training and evaluation."""

from pydantic import BaseModel
from typing import Optional


class GRPOConfig(BaseModel):
    """GRPO training configuration."""
    
    # Model settings
    model_name: str = "OpenPipe/Qwen3-14B-Instruct"
    max_seq_length: int = 16384
    load_in_4bit: bool = True
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    
    # Training settings
    num_generations: int = 4  # G in GRPO paper
    num_iterations: int = 1  # μ in GRPO paper
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    max_steps: int = 1000
    warmup_steps: int = 10
    logging_steps: int = 1
    eval_steps: int = 10
    save_steps: int = 10
    
    # GRPO-specific
    beta: float = 0.0  # KL divergence weight (0.0 = no KL penalty)
    scale_rewards: bool = False  # Whether to scale by std(rewards)
    
    # Agent settings
    max_turns: int = 10
    max_tokens: int = 4096
    
    # Dataset settings
    train_dataset_size: int = 1000
    eval_dataset_size: int = 100
    
    # Output
    output_dir: str = "outputs/grpo"
    run_name: str = "email_agent_grpo"
    
    # Misc
    seed: int = 42
    verbose: bool = True


class PolicyConfig(BaseModel):
    """Policy configuration for the email agent."""
    
    max_turns: int = 10
    max_tokens: int = 4096
    use_tools: bool = True
    stupid_simple_reward_fn: bool = False
    verbose: bool = True
    
    # Sampling diversity settings (for GRPO exploration)
    enable_dynamic_temperature: bool = True
    base_temperature: float = 0.5
    temperature_increment: float = 0.2
    base_repetition_penalty: float = 1.0
    repetition_penalty_increment: float = 0.05

