"""Utility functions and data classes for GRPO training."""

import os
from dataclasses import dataclass
from typing import List

from datasets import Dataset
from email_agent.data import SyntheticQuery
from email_agent.prompts import create_system_prompt


def get_env_int(key: str, default: str) -> int:
    """Get integer from environment variable."""
    value = os.environ.get(key, default)
    value = value.split('#')[0].strip()
    return int(value)


def get_env_float(key: str, default: str) -> float:
    """Get float from environment variable."""
    value = os.environ.get(key, default)
    value = value.split('#')[0].strip()
    return float(value)


@dataclass
class TrainingMetrics:
    """Metrics for a training step."""
    loss: float
    policy_loss: float
    kl_loss: float
    avg_reward: float
    max_reward: float
    min_reward: float
    accuracy: float
    num_trainable_tokens: int = 0
    num_total_tokens: int = 0
    rollout_time: float = 0.0
    training_time: float = 0.0
    reward_std: float = 0.0
    median_reward: float = 0.0


def prepare_dataset(queries: List[SyntheticQuery]) -> Dataset:
    """Prepare dataset for TRL training."""
    prompts = []
    for query in queries:
        system_prompt = create_system_prompt(query, max_turns=10)
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query.question},
        ]
        prompts.append({
            "prompt": prompt,
            "query_id": query.id,
        })
    
    return Dataset.from_list(prompts)

