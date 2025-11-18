"""Unified GRPO training script for Email Agent.

This script supports multiple training modes:
1. simple: Fast training with TRL (for quick testing)
2. rollout: Training with real agent rollouts
3. masked: Full implementation with token-level masking (RECOMMENDED)

Usage:
    # Masked training (default, recommended)
    python train_grpo.py --mode masked
    
    # Simple training (fast, for testing)
    python train_grpo.py --mode simple
    
    # Rollout training (without full masking)
    python train_grpo.py --mode rollout
"""

import os
import sys
import torch
import torch.nn.functional as F
import json
import asyncio
import argparse
import time
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import logging
from dataclasses import dataclass
import numpy as np
from functools import partial
from datetime import datetime, timedelta

# TRL imports
from datasets import Dataset
from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer
from transformers import TrainerCallback, TrainerState, TrainerControl

# Unsloth imports
from unsloth import FastLanguageModel
from openai import AsyncOpenAI

# Local imports
from email_agent.config import GRPOConfig, PolicyConfig
from email_agent.data import load_synthetic_queries, SyntheticQuery
from email_agent.agent import EmailAgent
from email_agent.rollout import calculate_reward, EvaluationRubric
from email_agent.prompts import create_system_prompt, get_tools_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


class AccuracyStopCallback(TrainerCallback):
    """Custom callback to stop training when accuracy reaches target and save best model."""
    
    def __init__(
        self,
        target_accuracy: float = 0.95,
        output_dir: str = "outputs/grpo",
        reward_tracker: Dict = None,
    ):
        self.target_accuracy = target_accuracy
        self.output_dir = output_dir
        self.best_accuracy = 0.0
        self.best_model_path = None
        self.reward_tracker = reward_tracker or {}
        
        logger.info(f"🎯 Target accuracy: {self.target_accuracy * 100:.1f}%")
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics is None:
            return control
        
        accuracy = self._calculate_accuracy_from_rewards()
        logger.info(f"📊 Step {state.global_step} - Current accuracy: {accuracy * 100:.2f}%")
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self._save_best_model(state, kwargs.get("model"), kwargs.get("tokenizer"))
            logger.info(f"✨ New best accuracy: {accuracy * 100:.2f}%")
        
        if accuracy >= self.target_accuracy:
            logger.info(f"🎉 Target accuracy {self.target_accuracy * 100:.1f}% reached!")
            logger.info(f"🏆 Final accuracy: {accuracy * 100:.2f}%")
            control.should_training_stop = True
        
        return control
    
    def _calculate_accuracy_from_rewards(self) -> float:
        """Calculate accuracy from tracked rewards."""
        if not self.reward_tracker.get("eval_rewards"):
            return 0.0
        
        rewards = self.reward_tracker["eval_rewards"]
        if len(rewards) == 0:
            return 0.0
        
        # Consider reward > 0.8 as correct answer
        correct_count = sum(1 for r in rewards if r > 0.8)
        accuracy = correct_count / len(rewards)
        
        return accuracy
    
    def _save_best_model(self, state: TrainerState, model, tokenizer):
        """Save the best model checkpoint."""
        if model is None:
            return
        
        best_model_dir = Path(self.output_dir) / "best_model"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(str(best_model_dir))
        if tokenizer:
            tokenizer.save_pretrained(str(best_model_dir))
        
        metadata = {
            "step": state.global_step,
            "accuracy": float(self.best_accuracy),
            "epoch": state.epoch,
        }
        
        with open(best_model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.best_model_path = str(best_model_dir)
        logger.info(f"💾 Best model saved to: {best_model_dir}")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training."""
        logger.info("="*60)
        logger.info("Training Summary")
        logger.info("="*60)
        logger.info(f"Best accuracy achieved: {self.best_accuracy * 100:.2f}%")
        logger.info(f"Target accuracy: {self.target_accuracy * 100:.1f}%")
        
        if self.best_model_path:
            logger.info(f"Best model saved at: {self.best_model_path}")
        
        if self.best_accuracy >= self.target_accuracy:
            logger.info("✅ Target accuracy reached!")
        else:
            logger.info("⚠️  Target accuracy not reached")
        
        logger.info("="*60)
        
        return control


# ==================== Mode 1: Simple Training (TRL-based) ====================

def simple_reward_function(
    completions,
    prompts,
    queries_dict,
    policy_config,
    reward_tracker=None,
    eval_dataset_size=100,
    **kwargs
):
    """Simple reward function for fast training (heuristic-based)."""
    rewards = []
    
    for completion, prompt in zip(completions, prompts):
        try:
            query_id = prompt.get("query_id") if isinstance(prompt, dict) else None
            
            if query_id is None or query_id not in queries_dict:
                content = completion[0]["content"] if completion else ""
                reward = 0.5 if len(content) > 20 else 0.0
                rewards.append(reward)
                continue
            
            query = queries_dict[query_id]
            rubric = EvaluationRubric()
            
            completion_content = completion[0]["content"] if completion else ""
            
            # Simple heuristic
            reward = 0.0
            if len(completion_content) > 20:
                reward += 0.3
            if "search_emails" in completion_content:
                reward += 0.2
                rubric.ever_found_right_email = True
            if "read_email" in completion_content:
                reward += 0.2
                rubric.ever_read_right_email = True
            if "return_final_answer" in completion_content:
                reward += 0.3
                rubric.attempted_answer = True
            
            final_reward = calculate_reward(policy_config, rubric)
            rewards.append(final_reward)
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            rewards.append(0.0)
    
    # Track rewards
    if reward_tracker is not None:
        if len(rewards) <= eval_dataset_size:
            reward_tracker["eval_rewards"] = rewards.copy()
        else:
            reward_tracker["train_rewards"].extend(rewards)
    
    return rewards


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


# ==================== Mode 2 & 3: Rollout Training ====================

async def execute_rollout(
    query: SyntheticQuery,
    model,
    tokenizer,
    policy_config: PolicyConfig,
    openai_client: Optional[AsyncOpenAI],
    verbose: bool = False,
    log_turns: bool = False,
) -> Tuple[List[Dict], float, EvaluationRubric]:
    """Execute a real agent rollout."""
    agent = EmailAgent(
        model=model,
        tokenizer=tokenizer,
        policy_config=policy_config,
        openai_client=openai_client,
    )
    
    rubric, conversation = await agent.run_query(query, verbose=verbose)
    reward = calculate_reward(policy_config, rubric)
    
    # Log turn-by-turn summary if requested
    if log_turns and len(conversation) > 2:  # Skip system and user messages
        logger.info(f"  Query: {query.question[:60]}...")
        turn_num = 0
        for i, msg in enumerate(conversation[2:], start=2):  # Skip system and initial user
            role = msg.get('role', '')
            if role == 'assistant':
                turn_num += 1
                tool_calls = msg.get('tool_calls', [])
                content = msg.get('content', '')
                if tool_calls:
                    tools_str = ', '.join([tc.get('function', {}).get('name', 'unknown') for tc in tool_calls])
                    logger.info(f"    Turn {turn_num}: {tools_str}")
                elif content:
                    logger.info(f"    Turn {turn_num}: text response ({len(content)} chars)")
        logger.info(f"  Result: reward={reward:.2f}, correct={rubric.answer_correct}")
    
    return conversation, reward, rubric


def rollout_reward_function(
    completions,
    prompts,
    model,
    tokenizer,
    queries_dict,
    policy_config,
    openai_client,
    reward_tracker=None,
    eval_dataset_size=100,
    **kwargs
):
    """Reward function with real agent rollouts."""
    rewards = []
    loop = asyncio.get_event_loop()
    
    for prompt in prompts:
        try:
            query_id = prompt.get("query_id") if isinstance(prompt, dict) else None
            
            if query_id is None or query_id not in queries_dict:
                logger.warning(f"Query ID not found: {query_id}")
                rewards.append(0.0)
                continue
            
            query = queries_dict[query_id]
            
            # Execute real rollout
            conversation, reward, metrics = loop.run_until_complete(
                execute_rollout(
                    query=query,
                    model=model,
                    tokenizer=tokenizer,
                    policy_config=policy_config,
                    openai_client=openai_client,
                    verbose=False,
                )
            )
            
            rewards.append(reward)
            
            if len(rewards) % 10 == 0:
                logger.info(f"Rollout {len(rewards)}: reward={reward:.3f}")
        
        except Exception as e:
            logger.error(f"Error in rollout: {e}")
            rewards.append(0.0)
    
    # Track rewards
    if reward_tracker is not None:
        if len(rewards) <= eval_dataset_size:
            reward_tracker["eval_rewards"] = rewards.copy()
        else:
            reward_tracker["train_rewards"].extend(rewards)
    
    logger.info(f"Batch rewards: mean={np.mean(rewards):.3f}, min={np.min(rewards):.3f}, max={np.max(rewards):.3f}")
    
    return rewards


# ==================== Mode 3: Masked Training (Full Implementation) ====================

class AgentGRPOTrainer:
    """Custom GRPO Trainer with token-level masking."""
    
    def __init__(
        self,
        model,
        tokenizer,
        train_queries: List[SyntheticQuery],
        eval_queries: List[SyntheticQuery],
        policy_config: PolicyConfig,
        openai_client: Optional[AsyncOpenAI],
        num_rollouts: int = 4,
        learning_rate: float = 1e-5,
        beta: float = 0.01,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        output_dir: str = "outputs/grpo",
        target_accuracy: float = 0.95,
        eval_steps: int = 50,
        save_steps: int = 100,
        max_steps: int = 1000,
        warmup_steps: int = 10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_queries = train_queries
        self.eval_queries = eval_queries
        self.policy_config = policy_config
        self.openai_client = openai_client
        self.num_rollouts = num_rollouts
        self.beta = beta
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = Path(output_dir)
        self.target_accuracy = target_accuracy
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Reference model for KL divergence
        self.ref_model = None
        if beta > 0:
            logger.info("Creating reference model for KL divergence...")
            from copy import deepcopy
            self.ref_model = deepcopy(model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            logger.info("✓ Reference model created")
        
        self.global_step = 0
        self.best_accuracy = 0.0
        self.best_model_path = None
        
        logger.info("="*60)
        logger.info("AgentGRPOTrainer initialized")
        logger.info("="*60)
        logger.info(f"Train queries: {len(train_queries)}")
        logger.info(f"Eval queries: {len(eval_queries)}")
        logger.info(f"Rollouts per query: {num_rollouts}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"KL penalty (beta): {beta}")
        logger.info(f"Target accuracy: {target_accuracy*100:.1f}%")
        logger.info("="*60)
    
    def tokenize_conversation_with_mask(
        self,
        conversation: List[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize conversation and create loss mask."""
        all_tokens = []
        all_masks = []
        
        for msg in conversation:
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            
            is_model_generated = (role == "assistant")
            
            # Serialize message
            if role == "system":
                text = f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                text = f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                if tool_calls:
                    tool_calls_str = json.dumps(tool_calls)
                    text = f"<|im_start|>assistant\n{tool_calls_str}<|im_end|>\n"
                else:
                    text = f"<|im_start|>assistant\n{content}<|im_end|>\n"
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                text = f"<|im_start|>tool\ntool_call_id: {tool_call_id}\n{content}<|im_end|>\n"
            else:
                logger.warning(f"Unknown role: {role}")
                text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            
            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Create mask
            if is_model_generated:
                mask = [1.0] * len(tokens)
            else:
                mask = [0.0] * len(tokens)
            
            all_tokens.extend(tokens)
            all_masks.extend(mask)
        
        # Add EOS
        all_tokens.append(self.tokenizer.eos_token_id)
        all_masks.append(0.0)
        
        input_ids = torch.tensor(all_tokens, dtype=torch.long)
        labels = input_ids.clone()
        loss_mask = torch.tensor(all_masks, dtype=torch.float)
        
        return input_ids, labels, loss_mask
    
    def compute_loss_for_trajectory(
        self,
        conversation: List[Dict],
        advantage: float,
        log_mask: bool = False,
        trajectory_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a single trajectory with token masking."""
        input_ids, labels, loss_mask = self.tokenize_conversation_with_mask(conversation)
        
        device = next(self.model.parameters()).device
        input_ids = input_ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        loss_mask = loss_mask.to(device)
        
        # Log token mask visualization
        if log_mask:
            self._log_token_mask(input_ids[0], loss_mask, trajectory_idx)
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=labels)
        logits = outputs.logits
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = loss_mask[1:].contiguous()
        
        # Compute token losses
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Apply mask
        masked_losses = token_losses * shift_mask
        
        num_trainable_tokens = shift_mask.sum().item()
        num_total_tokens = shift_mask.numel()
        
        if num_trainable_tokens == 0:
            policy_loss = torch.tensor(0.0, device=device)
        else:
            avg_loss = masked_losses.sum() / shift_mask.sum()
            policy_loss = advantage * avg_loss
        
        # KL divergence
        kl_loss = torch.tensor(0.0, device=device)
        if self.beta > 0 and self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(input_ids=input_ids)
                ref_logits = ref_outputs.logits[:, :-1, :].contiguous()
            
            kl_div = F.kl_div(
                F.log_softmax(shift_logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction='none',
            ).sum(-1)
            
            masked_kl = (kl_div.squeeze() * shift_mask).sum() / shift_mask.sum()
            kl_loss = self.beta * masked_kl
        
        total_loss = policy_loss + kl_loss
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "num_trainable_tokens": num_trainable_tokens,
            "num_total_tokens": num_total_tokens,
            "trainable_ratio": num_trainable_tokens / num_total_tokens if num_total_tokens > 0 else 0,
        }
        
        return total_loss, metrics
    
    def _log_token_mask(self, input_ids: torch.Tensor, loss_mask: torch.Tensor, trajectory_idx: int):
        """Log visualization of token mask."""
        logger.info(f"\n{'─'*80}")
        logger.info(f"🎭 Token Mask Visualization (Trajectory {trajectory_idx+1}):")
        logger.info(f"{'─'*80}")
        
        # Decode tokens
        tokens = input_ids.cpu().tolist()
        mask = loss_mask.cpu().tolist()
        
        # Find role boundaries
        role_starts = []
        current_text = ""
        for i, token_id in enumerate(tokens):
            token_text = self.tokenizer.decode([token_id])
            current_text += token_text
            
            # Detect role markers
            if '<|im_start|>' in current_text:
                role_starts.append(i)
                current_text = ""
        
        # Group by roles and display
        logger.info("Legend: ✅ = trainable (model output), ❌ = not trainable (context/tool results)\n")
        
        current_role = "unknown"
        role_tokens = []
        role_mask_values = []
        
        for i, (token_id, mask_val) in enumerate(zip(tokens, mask)):
            token_text = self.tokenizer.decode([token_id])
            
            # Check for role change
            if '<|im_start|>' in token_text:
                # Log previous role if exists
                if role_tokens:
                    self._log_role_section(current_role, role_tokens, role_mask_values)
                    role_tokens = []
                    role_mask_values = []
                
                # Try to extract role name
                remaining_text = self.tokenizer.decode(tokens[i:min(i+10, len(tokens))])
                if 'assistant' in remaining_text:
                    current_role = 'assistant'
                elif 'user' in remaining_text:
                    current_role = 'user'
                elif 'system' in remaining_text:
                    current_role = 'system'
                elif 'tool' in remaining_text:
                    current_role = 'tool'
            
            role_tokens.append(token_text)
            role_mask_values.append(mask_val)
        
        # Log last role
        if role_tokens:
            self._log_role_section(current_role, role_tokens, role_mask_values)
        
        # Statistics
        trainable = sum(mask)
        total = len(mask)
        logger.info(f"\n📊 Mask Statistics:")
        logger.info(f"  Total tokens: {total}")
        logger.info(f"  Trainable tokens: {trainable} ({trainable/total*100:.1f}%)")
        logger.info(f"  Masked tokens: {total-trainable} ({(total-trainable)/total*100:.1f}%)")
        logger.info(f"{'─'*80}\n")
    
    def _log_role_section(self, role: str, tokens: List[str], mask_values: List[float]):
        """Log a role section with mask visualization."""
        if not tokens:
            return
        
        # Determine role emoji
        role_emoji = {
            'system': '⚙️',
            'user': '👤',
            'assistant': '🤖',
            'tool': '🔧',
        }.get(role, '❓')
        
        # Count trainable tokens
        trainable = sum(1 for m in mask_values if m > 0.5)
        total = len(mask_values)
        
        # Get text preview
        text = ''.join(tokens).strip()
        text_preview = text[:100] + '...' if len(text) > 100 else text
        text_preview = text_preview.replace('\n', ' ')
        
        mask_indicator = '✅' if trainable > 0 else '❌'
        logger.info(f"{role_emoji} {role.upper():10s} {mask_indicator} ({trainable}/{total} tokens trainable)")
        logger.info(f"   {text_preview}")
    
    async def collect_rollouts_for_batch(
        self,
        queries: List[SyntheticQuery],
        log_rollouts: bool = False,
    ) -> Tuple[List[List[Dict]], List[float], List[EvaluationRubric]]:
        """Collect rollouts for a batch of queries."""
        trajectories = []
        rewards = []
        rubrics = []
        
        total_rollouts = len(queries) * self.num_rollouts
        
        if log_rollouts:
            logger.info(f"\n{'─'*80}")
            logger.info(f"🎲 Rollout Details:")
            logger.info(f"{'─'*80}")
        
        with tqdm(total=total_rollouts, desc="🎲 Collecting rollouts", 
                  unit="rollout", leave=False, ncols=100, disable=log_rollouts) as pbar:
            for query_idx, query in enumerate(queries):
                if log_rollouts:
                    logger.info(f"\n📋 Query {query_idx+1}/{len(queries)}: Collecting {self.num_rollouts} rollouts")
                
                for rollout_idx in range(self.num_rollouts):
                    conversation, reward, rubric = await execute_rollout(
                        query, self.model, self.tokenizer, 
                        self.policy_config, self.openai_client, 
                        verbose=False,
                        log_turns=log_rollouts
                    )
                    trajectories.append(conversation)
                    rewards.append(reward)
                    rubrics.append(rubric)
                    
                    # Update progress bar with current stats
                    if not log_rollouts:
                        pbar.set_postfix({
                            'query': f'{query_idx+1}/{len(queries)}',
                            'reward': f'{reward:.2f}',
                            'avg': f'{np.mean(rewards):.2f}'
                        })
                        pbar.update(1)
        
        if log_rollouts:
            logger.info(f"{'─'*80}\n")
        
        return trajectories, rewards, rubrics
    
    def compute_advantages(self, rewards: List[float], log_details: bool = False) -> List[float]:
        """Compute advantages using GRPO."""
        rewards_array = np.array(rewards)
        num_groups = len(rewards) // self.num_rollouts
        advantages = []
        
        if log_details:
            logger.info(f"\n{'─'*80}")
            logger.info(f"📊 Group Advantages Computation:")
            logger.info(f"{'─'*80}")
        
        for i in range(num_groups):
            start_idx = i * self.num_rollouts
            end_idx = start_idx + self.num_rollouts
            group_rewards = rewards_array[start_idx:end_idx]
            
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            
            group_advantages = (group_rewards - group_mean) / group_std
            advantages.extend(group_advantages.tolist())
            
            if log_details:
                logger.info(f"\nQuery Group {i+1}/{num_groups}:")
                logger.info(f"  Rewards: {[f'{r:.3f}' for r in group_rewards]}")
                logger.info(f"  Mean: {group_mean:.3f}, Std: {group_std:.3f}")
                logger.info(f"  Advantages: {[f'{a:+.3f}' for a in group_advantages]}")
                
                # Highlight best and worst
                best_idx = np.argmax(group_rewards)
                worst_idx = np.argmin(group_rewards)
                logger.info(f"  Best rollout: #{best_idx+1} (reward={group_rewards[best_idx]:.3f}, adv={group_advantages[best_idx]:+.3f})")
                logger.info(f"  Worst rollout: #{worst_idx+1} (reward={group_rewards[worst_idx]:.3f}, adv={group_advantages[worst_idx]:+.3f})")
        
        if log_details:
            logger.info(f"{'─'*80}\n")
        
        return advantages
    
    def training_step(
        self,
        queries: List[SyntheticQuery],
        log_details: bool = False,
    ) -> TrainingMetrics:
        """Execute a single training step."""
        self.model.train()
        
        # Collect rollouts with timing
        if not log_details:
            logger.info(f"📊 Collecting {self.num_rollouts} rollouts for {len(queries)} queries...")
        rollout_start = time.time()
        trajectories, rewards, rubrics = asyncio.run(
            self.collect_rollouts_for_batch(queries, log_rollouts=log_details)
        )
        rollout_time = time.time() - rollout_start
        
        # Compute advantages
        advantages = self.compute_advantages(rewards, log_details=log_details)
        
        # Training with timing
        training_start = time.time()
        total_loss = 0
        total_policy_loss = 0
        total_kl_loss = 0
        total_trainable_tokens = 0
        total_tokens = 0
        
        # Show token mask for first trajectory if logging details
        show_mask_for_first = log_details
        
        # Progress bar for gradient computation
        with tqdm(total=len(trajectories), desc="⚡ Computing gradients", 
                  unit="traj", leave=False, ncols=100, disable=log_details) as pbar:
            for i, (traj, adv) in enumerate(zip(trajectories, advantages)):
                log_mask = show_mask_for_first and i == 0
                loss, metrics = self.compute_loss_for_trajectory(
                    traj, adv, log_mask=log_mask, trajectory_idx=i
                )
                
                total_loss += loss / len(trajectories)
                total_policy_loss += metrics["policy_loss"]
                total_kl_loss += metrics["kl_loss"]
                total_trainable_tokens += metrics["num_trainable_tokens"]
                total_tokens += metrics["num_total_tokens"]
                
                if not log_details:
                    pbar.update(1)
        
        # Backpropagation
        if not log_details:
            logger.info("🔄 Backpropagating gradients...")
        else:
            logger.info(f"\n{'─'*80}")
            logger.info("🔄 Backpropagating gradients...")
            logger.info(f"{'─'*80}")
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        training_time = time.time() - training_start
        
        # Compute metrics
        correct_answers = sum(1 for r in rubrics if r.answer_correct)
        accuracy = correct_answers / len(rubrics)
        
        metrics = TrainingMetrics(
            loss=total_loss.item(),
            policy_loss=total_policy_loss / len(trajectories),
            kl_loss=total_kl_loss / len(trajectories),
            avg_reward=np.mean(rewards),
            max_reward=np.max(rewards),
            min_reward=np.min(rewards),
            accuracy=accuracy,
            num_trainable_tokens=total_trainable_tokens,
            num_total_tokens=total_tokens,
            rollout_time=rollout_time,
            training_time=training_time,
            reward_std=np.std(rewards),
            median_reward=np.median(rewards),
        )
        
        return metrics
    
    async def evaluate(self) -> Tuple[float, float]:
        """Evaluate the model."""
        self.model.eval()
        
        all_rewards = []
        correct_answers = 0
        all_rubrics = []
        
        logger.info(f"📊 Evaluating on {len(self.eval_queries)} queries...")
        eval_start = time.time()
        
        with torch.no_grad():
            with tqdm(total=len(self.eval_queries), desc="🔍 Evaluation", 
                      unit="query", ncols=100) as pbar:
                for query in self.eval_queries:
                    conversation, reward, rubric = await execute_rollout(
                        query, self.model, self.tokenizer,
                        self.policy_config, self.openai_client, verbose=False
                    )
                    all_rewards.append(reward)
                    all_rubrics.append(rubric)
                    
                    if rubric.answer_correct:
                        correct_answers += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'reward': f'{reward:.2f}',
                        'correct': f'{correct_answers}/{len(all_rewards)}',
                        'acc': f'{correct_answers/len(all_rewards)*100:.1f}%'
                    })
                    pbar.update(1)
        
        eval_time = time.time() - eval_start
        avg_reward = np.mean(all_rewards)
        accuracy = correct_answers / len(self.eval_queries)
        
        # Detailed evaluation statistics
        logger.info(f"\n{'─'*80}")
        logger.info(f"📊 EVALUATION RESULTS")
        logger.info(f"{'─'*80}")
        logger.info(f"Time taken: {eval_time:.1f}s ({eval_time/len(self.eval_queries):.2f}s per query)")
        logger.info(f"Average reward: {avg_reward:.3f} (std: {np.std(all_rewards):.3f})")
        logger.info(f"Median reward: {np.median(all_rewards):.3f}")
        logger.info(f"Reward range: [{np.min(all_rewards):.3f}, {np.max(all_rewards):.3f}]")
        logger.info(f"Accuracy: {accuracy*100:.2f}% ({correct_answers}/{len(self.eval_queries)})")
        
        # Rubric statistics
        attempted = sum(1 for r in all_rubrics if r.attempted_answer)
        found_email = sum(1 for r in all_rubrics if r.ever_found_right_email)
        read_email = sum(1 for r in all_rubrics if r.ever_read_right_email)
        logger.info(f"\nRubric Statistics:")
        logger.info(f"  - Attempted answer: {attempted}/{len(all_rubrics)} ({attempted/len(all_rubrics)*100:.1f}%)")
        logger.info(f"  - Found correct email: {found_email}/{len(all_rubrics)} ({found_email/len(all_rubrics)*100:.1f}%)")
        logger.info(f"  - Read correct email: {read_email}/{len(all_rubrics)} ({read_email/len(all_rubrics)*100:.1f}%)")
        logger.info(f"{'─'*80}\n")
        
        return avg_reward, accuracy
    
    def save_model(self, path: Path, metrics: Optional[Dict] = None):
        """Save model checkpoint."""
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        
        if metrics:
            metadata = {
                "step": self.global_step,
                "metrics": metrics,
            }
            
            with open(path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Model saved to: {path}")
    
    def train(self):
        """Main training loop."""
        logger.info("\n" + "="*80)
        logger.info("🚀 Starting Training with Token-Level Masking")
        logger.info("="*80)
        logger.info(f"🎯 Target accuracy: {self.target_accuracy*100:.1f}%")
        logger.info(f"📊 Max steps: {self.max_steps}")
        logger.info(f"📚 Train queries: {len(self.train_queries)}")
        logger.info(f"📝 Eval queries: {len(self.eval_queries)}")
        logger.info(f"🎲 Rollouts per query: {self.num_rollouts}")
        logger.info(f"📦 Batch size: {self.batch_size}")
        logger.info(f"✅ Only model-generated tokens will be trained")
        logger.info(f"❌ Tool results will NOT be trained")
        logger.info("="*80)
        
        step = 0
        training_start_time = time.time()
        cumulative_rollout_time = 0
        cumulative_training_time = 0
        
        # Track moving averages
        recent_rewards = []
        recent_accuracies = []
        window_size = 5
        
        while step < self.max_steps:
            step_start_time = time.time()
            
            batch_queries = np.random.choice(
                self.train_queries,
                size=min(self.batch_size, len(self.train_queries)),
                replace=False
            ).tolist()
            
            # Log detailed information for first step and every 10 steps
            log_details = (step == 0) or (step % 10 == 0)
            
            metrics = self.training_step(batch_queries, log_details=log_details)
            
            step += 1
            self.global_step = step
            
            cumulative_rollout_time += metrics.rollout_time
            cumulative_training_time += metrics.training_time
            step_time = time.time() - step_start_time
            
            # Update moving averages
            recent_rewards.append(metrics.avg_reward)
            recent_accuracies.append(metrics.accuracy)
            if len(recent_rewards) > window_size:
                recent_rewards.pop(0)
                recent_accuracies.pop(0)
            
            # Calculate ETA
            elapsed_time = time.time() - training_start_time
            avg_step_time = elapsed_time / step
            remaining_steps = self.max_steps - step
            eta_seconds = avg_step_time * remaining_steps
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            
            # GPU memory usage
            if torch.cuda.is_available():
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                gpu_mem_str = f"GPU: {gpu_mem_allocated:.1f}/{gpu_mem_reserved:.1f}GB"
            else:
                gpu_mem_str = "GPU: N/A"
            
            logger.info(f"\n{'─'*80}")
            logger.info(f"📍 Step {step}/{self.max_steps} ({step/self.max_steps*100:.1f}%) | ETA: {eta_str}")
            logger.info(f"{'─'*80}")
            logger.info(f"⏱️  Timing:")
            logger.info(f"   Step time: {step_time:.1f}s (rollout: {metrics.rollout_time:.1f}s, training: {metrics.training_time:.1f}s)")
            logger.info(f"   Avg step: {avg_step_time:.1f}s | Total: {str(timedelta(seconds=int(elapsed_time)))}")
            logger.info(f"\n💰 Rewards:")
            logger.info(f"   Mean: {metrics.avg_reward:.3f} ± {metrics.reward_std:.3f}")
            logger.info(f"   Median: {metrics.median_reward:.3f}")
            logger.info(f"   Range: [{metrics.min_reward:.3f}, {metrics.max_reward:.3f}]")
            logger.info(f"   Moving avg (last {len(recent_rewards)}): {np.mean(recent_rewards):.3f}")
            logger.info(f"\n📊 Loss:")
            logger.info(f"   Total: {metrics.loss:.4f}")
            logger.info(f"   Policy: {metrics.policy_loss:.4f}")
            logger.info(f"   KL: {metrics.kl_loss:.4f}")
            logger.info(f"\n🎯 Accuracy:")
            logger.info(f"   Current: {metrics.accuracy*100:.1f}%")
            logger.info(f"   Moving avg (last {len(recent_accuracies)}): {np.mean(recent_accuracies)*100:.1f}%")
            logger.info(f"   Best: {self.best_accuracy*100:.1f}%")
            logger.info(f"\n🔢 Tokens:")
            logger.info(f"   Trainable: {metrics.num_trainable_tokens:,}/{metrics.num_total_tokens:,} "
                       f"({metrics.num_trainable_tokens/metrics.num_total_tokens*100:.1f}%)")
            logger.info(f"\n💾 {gpu_mem_str}")
            logger.info(f"{'─'*80}")
            
            # Evaluate
            if step % self.eval_steps == 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"🔍 EVALUATION at step {step}")
                logger.info(f"{'='*80}")
                
                avg_reward, accuracy = asyncio.run(self.evaluate())
                
                improvement = ""
                if accuracy > self.best_accuracy:
                    improvement = f" (+{(accuracy - self.best_accuracy)*100:.2f}%)"
                    self.best_accuracy = accuracy
                    self.best_model_path = self.output_dir / "best_model"
                    self.save_model(
                        self.best_model_path,
                        {"accuracy": accuracy, "reward": avg_reward, "step": step}
                    )
                    logger.info(f"✨ New best accuracy: {accuracy*100:.2f}%{improvement}")
                
                progress_to_target = accuracy / self.target_accuracy * 100
                logger.info(f"📈 Progress to target: {progress_to_target:.1f}% "
                           f"({accuracy*100:.1f}% / {self.target_accuracy*100:.1f}%)")
                
                if accuracy >= self.target_accuracy:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"🎉 TARGET ACCURACY REACHED!")
                    logger.info(f"🏆 Final accuracy: {accuracy*100:.2f}%")
                    logger.info(f"🎯 Target: {self.target_accuracy*100:.1f}%")
                    logger.info(f"⏱️  Total time: {str(timedelta(seconds=int(time.time() - training_start_time)))}")
                    logger.info(f"{'='*80}")
                    break
                
                logger.info(f"{'='*80}\n")
            
            # Save checkpoint
            if step % self.save_steps == 0:
                checkpoint_path = self.output_dir / f"checkpoint-{step}"
                self.save_model(checkpoint_path, {"step": step})
                logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Final evaluation
        total_training_time = time.time() - training_start_time
        logger.info(f"\n{'='*80}")
        logger.info("🏁 FINAL EVALUATION")
        logger.info(f"{'='*80}")
        
        avg_reward, accuracy = asyncio.run(self.evaluate())
        
        logger.info(f"\n{'='*80}")
        logger.info("📊 TRAINING SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total steps: {step}")
        logger.info(f"Total time: {str(timedelta(seconds=int(total_training_time)))}")
        logger.info(f"  - Rollout time: {str(timedelta(seconds=int(cumulative_rollout_time)))} "
                   f"({cumulative_rollout_time/total_training_time*100:.1f}%)")
        logger.info(f"  - Training time: {str(timedelta(seconds=int(cumulative_training_time)))} "
                   f"({cumulative_training_time/total_training_time*100:.1f}%)")
        logger.info(f"Avg time per step: {total_training_time/step:.1f}s")
        logger.info(f"\nFinal accuracy: {accuracy*100:.2f}%")
        logger.info(f"Best accuracy: {self.best_accuracy*100:.2f}%")
        logger.info(f"Final reward: {avg_reward:.3f}")
        
        if self.best_model_path:
            logger.info(f"\n💾 Best model saved at: {self.best_model_path}")
        
        # Save final model
        final_path = self.output_dir / "final"
        self.save_model(final_path, {
            "accuracy": accuracy, 
            "reward": avg_reward,
            "step": step,
            "training_time": total_training_time
        })
        logger.info(f"💾 Final model saved at: {final_path}")
        
        logger.info(f"{'='*80}")
        logger.info("✅ Training Complete!")
        logger.info(f"{'='*80}\n")


# ==================== Main Training Function ====================

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="GRPO Training for Email Agent")
    parser.add_argument(
        "--mode",
        type=str,
        default="masked",
        choices=["simple", "rollout", "masked"],
        help="Training mode: simple (fast), rollout (real agent), masked (full, recommended)"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = GRPOConfig(
        model_name=os.environ.get("MODEL_NAME", "unsloth/Qwen3-14B-Base").split('#')[0].strip(),
        train_dataset_size=get_env_int("TRAIN_DATASET_SIZE", "3000"),
        eval_dataset_size=get_env_int("EVAL_DATASET_SIZE", "1000"),
        max_steps=get_env_int("MAX_STEPS", "200"),
        learning_rate=get_env_float("LEARNING_RATE", "1e-5"),
        per_device_train_batch_size=get_env_int("PER_DEVICE_TRAIN_BATCH_SIZE", "2"),
        num_generations=get_env_int("NUM_GENERATIONS", "6"),
        beta=get_env_float("BETA", "0.01"),
        max_turns=get_env_int("MAX_TURNS", "4"),
        max_tokens=get_env_int("MAX_TOKENS", "4096"),
        output_dir=os.environ.get("OUTPUT_DIR", f"outputs/grpo_{args.mode}").split('#')[0].strip(),
        seed=get_env_int("SEED", "42"),
        verbose=os.environ.get("VERBOSE", "false").split('#')[0].strip().lower() == "true",
    )
    
    policy_config = PolicyConfig(
        max_turns=config.max_turns,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        stupid_simple_reward_fn=True,  # Use simple reward for speed
    )
    
    logger.info("="*60)
    logger.info(f"GRPO Training - Mode: {args.mode.upper()}")
    logger.info("="*60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Train dataset: {config.train_dataset_size}")
    logger.info(f"Eval dataset: {config.eval_dataset_size}")
    logger.info(f"Max steps: {config.max_steps}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Batch size: {config.per_device_train_batch_size}")
    logger.info(f"Output dir: {config.output_dir}")
    
    if args.mode == "simple":
        logger.info("⚡ Fast training with heuristic rewards")
    elif args.mode == "rollout":
        logger.info("🔄 Training with real agent rollouts")
    elif args.mode == "masked":
        logger.info("✅ Full implementation with token-level masking (RECOMMENDED)")
    
    logger.info("="*60)
    
    # Initialize OpenRouter client (for judge)
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_api_key and openrouter_api_key != "your_openrouter_api_key_here":
        openai_client = AsyncOpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        logger.info("✓ OpenRouter client initialized (for judge)")
    else:
        logger.warning("⚠ OpenRouter API key not set - using heuristic evaluation")
        logger.warning("⚠ Set OPENROUTER_API_KEY environment variable to enable judge-based evaluation")
        openai_client = None
    
    # Load model
    logger.info("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        dtype=None,
    )
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        max_seq_length=config.max_seq_length,
    )
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    logger.info("✓ Model loaded")
    
    # Load datasets
    logger.info("Loading datasets...")
    # Use train split for training, test split for evaluation to avoid overfitting
    train_queries = load_synthetic_queries(
        split="train",
        limit=config.train_dataset_size,
        shuffle=True,
        max_messages=1,
    )
    eval_queries = load_synthetic_queries(
        split="test",
        limit=config.eval_dataset_size,
        shuffle=True,
        max_messages=1,
    )
    
    logger.info(f"✓ Loaded {len(train_queries)} train queries")
    logger.info(f"✓ Loaded {len(eval_queries)} eval queries")
    
    # Training based on mode
    if args.mode == "masked":
        # Full implementation with token masking
        trainer = AgentGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            train_queries=train_queries,
            eval_queries=eval_queries,
            policy_config=policy_config,
            openai_client=openai_client,
            num_rollouts=config.num_generations,
            learning_rate=config.learning_rate,
            beta=config.beta,
            batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            output_dir=config.output_dir,
            target_accuracy=0.95,
            eval_steps=50,
            save_steps=100,
            max_steps=config.max_steps,
            warmup_steps=10,
        )
        trainer.train()
        
    else:
        # Simple or rollout mode using TRL
        reward_tracker = {
            "eval_rewards": [],
            "train_rewards": [],
        }
        
        queries_dict = {q.id: q for q in train_queries + eval_queries}
        
        train_dataset = prepare_dataset(train_queries)
        eval_dataset = prepare_dataset(eval_queries)
        
        training_args = TRLGRPOConfig(
            output_dir=config.output_dir,
            run_name=f"email_agent_grpo_{args.mode}",
            num_train_epochs=2,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            max_steps=config.max_steps,
            warmup_steps=config.warmup_steps,
            logging_steps=config.logging_steps,
            eval_strategy="steps",
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            save_total_limit=3,
            seed=config.seed,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            num_generation_per_prompt=config.num_generations,
            max_new_tokens=config.max_tokens,
            temperature=0.7,
        )
        
        # Select reward function
        if args.mode == "simple":
            reward_fn = partial(
                simple_reward_function,
                queries_dict=queries_dict,
                policy_config=policy_config,
                reward_tracker=reward_tracker,
                eval_dataset_size=config.eval_dataset_size,
            )
        else:  # rollout
            reward_fn = partial(
                rollout_reward_function,
                model=model,
                tokenizer=tokenizer,
                queries_dict=queries_dict,
                policy_config=policy_config,
                openai_client=openai_client,
                reward_tracker=reward_tracker,
                eval_dataset_size=config.eval_dataset_size,
            )
        
        accuracy_callback = AccuracyStopCallback(
            target_accuracy=0.95,
            output_dir=config.output_dir,
            reward_tracker=reward_tracker,
        )
        
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            reward_funcs=reward_fn,
            callbacks=[accuracy_callback],
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("Training complete!")
        
        # Save final model
        final_model_dir = f"{config.output_dir}/final"
        logger.info(f"Saving final model to {final_model_dir}")
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        # Print model locations
        logger.info("")
        logger.info("="*60)
        logger.info("📁 Model Locations:")
        logger.info("="*60)
        logger.info(f"Final model: {final_model_dir}")
        if accuracy_callback.best_model_path:
            logger.info(f"Best model: {accuracy_callback.best_model_path}")
            logger.info(f"Best accuracy: {accuracy_callback.best_accuracy * 100:.2f}%")
        logger.info("="*60)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
