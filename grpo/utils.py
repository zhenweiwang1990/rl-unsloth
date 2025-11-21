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
    groups_kept: int = 0
    groups_filtered: int = 0
    num_early_exit: int = 0  # Rollouts that finished without exhausting turns


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
    found_valid_accuracy = False
    
    for step, ckpt_path in checkpoints:
        metadata_file = ckpt_path / "training_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                # Check if accuracy exists in metadata (not just default 0.0)
                if "accuracy" in metadata:
                    accuracy = metadata["accuracy"]
                    found_valid_accuracy = True
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_checkpoint = ckpt_path
                # Also check in nested metrics dict
                elif "metrics" in metadata and "accuracy" in metadata["metrics"]:
                    accuracy = metadata["metrics"]["accuracy"]
                    found_valid_accuracy = True
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_checkpoint = ckpt_path
            except (json.JSONDecodeError, KeyError):
                continue
    
    # If no valid accuracy found, return latest checkpoint
    if not found_valid_accuracy and checkpoints:
        return checkpoints[-1][1], 0.0
    
    return (best_checkpoint, best_accuracy) if best_checkpoint else None


def find_checkpoint_with_best_marker(output_dir: str) -> Optional[Path]:
    """Find checkpoint that has best_model_path marker in training_state.json.
    
    This finds checkpoints that were marked as the best model during training.
    If multiple checkpoints have the marker, returns the one with highest step.
    
    Returns:
        Path to checkpoint with best marker, or None if no such checkpoint found.
    """
    checkpoints = find_checkpoints(output_dir)
    if not checkpoints:
        return None
    
    # Check all checkpoints, find ones with best_model_path marker
    marked_checkpoints = []
    for step, ckpt_path in checkpoints:
        state_file = ckpt_path / "training_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                best_model_path = state.get("best_model_path")
                if best_model_path and best_model_path != "None":
                    # This checkpoint has a best model marker
                    marked_checkpoints.append((step, ckpt_path))
            except (json.JSONDecodeError, KeyError):
                continue
    
    # Return the one with highest step if any found
    if marked_checkpoints:
        # Sort by step and return the latest one with marker
        marked_checkpoints.sort(key=lambda x: x[0])
        return marked_checkpoints[-1][1]
    
    return None


def find_auto_resume_checkpoint(output_dir: str) -> Optional[Path]:
    """Automatically find checkpoint to resume from.
    
    Priority:
    1. Checkpoint with best_model_path marker (if exists)
    2. Latest checkpoint (if exists)
    3. None (start from scratch)
    
    Returns:
        Path to checkpoint to resume from, or None if no checkpoints found.
    """
    # First, try to find checkpoint with best marker
    best_marked = find_checkpoint_with_best_marker(output_dir)
    if best_marked:
        return best_marked
    
    # Otherwise, use latest checkpoint
    return find_latest_checkpoint(output_dir)

