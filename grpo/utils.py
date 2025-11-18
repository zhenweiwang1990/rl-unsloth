"""Utility functions and data classes for GRPO training."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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


def find_checkpoints(output_dir: str) -> List[Tuple[int, Path]]:
    """Find all checkpoints in output directory.
    
    Returns:
        List of (step_number, checkpoint_path) tuples, sorted by step number.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return []
    
    checkpoints = []
    for path in output_path.iterdir():
        if path.is_dir() and path.name.startswith("checkpoint-"):
            try:
                step = int(path.name.split("-")[1])
                checkpoints.append((step, path))
            except (IndexError, ValueError):
                continue
    
    return sorted(checkpoints, key=lambda x: x[0])


def find_latest_checkpoint(output_dir: str) -> Optional[Path]:
    """Find the latest checkpoint in output directory.
    
    Returns:
        Path to latest checkpoint, or None if no checkpoints found.
    """
    checkpoints = find_checkpoints(output_dir)
    if not checkpoints:
        return None
    return checkpoints[-1][1]


def find_best_checkpoint(output_dir: str) -> Optional[Tuple[Path, float]]:
    """Find the best checkpoint based on accuracy/metrics.
    
    Returns:
        Tuple of (checkpoint_path, accuracy), or None if no checkpoints found.
    """
    checkpoints = find_checkpoints(output_dir)
    if not checkpoints:
        return None
    
    best_checkpoint = None
    best_accuracy = -1.0
    
    for step, ckpt_path in checkpoints:
        metadata_file = ckpt_path / "training_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                accuracy = metadata.get("accuracy", 0.0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_checkpoint = ckpt_path
            except (json.JSONDecodeError, KeyError):
                continue
    
    # If no metadata found, return latest checkpoint
    if best_checkpoint is None and checkpoints:
        return checkpoints[-1][1], 0.0
    
    return (best_checkpoint, best_accuracy) if best_checkpoint else None

