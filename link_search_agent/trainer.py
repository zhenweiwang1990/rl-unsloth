"""Custom GRPO Trainer for Link Search Agent with token-level masking."""

import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from link_search_agent.config import PolicyConfig
from link_search_agent.data.types import LinkSearchQuery
from link_search_agent.rollout import LinkSearchRubric, calculate_reward, normalize_handle
from link_search_agent.grpo_utils import execute_rollout
from link_search_agent.rollout_logger import save_rollout_logs

logger = logging.getLogger(__name__)


@dataclass
class TrajectorySample:
    """Single trajectory collected from a rollout."""
    query_id: str
    query: LinkSearchQuery
    conversation: List[Dict]
    reward: float
    rubric: LinkSearchRubric
    rollout_idx: int
    group_id: int
    advantage: Optional[float] = None
    turn_advantages: Optional[List[float]] = None
    rollout_log: Optional[object] = None


@dataclass
class TrajectoryGroup:
    """Grouped trajectories for GRPO."""
    query: LinkSearchQuery
    group_id: int
    samples: List[TrajectorySample] = field(default_factory=list)


@dataclass
class TokenizedTrajectory:
    """Tokenized trajectory ready for batching."""
    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    attention_mask: torch.Tensor
    advantage_mask: torch.Tensor
    group_id: int
    query_id: str
    old_logprobs: Optional[torch.Tensor] = None


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
    avg_score: float = 0.0
    avg_hits: float = 0.0
    num_trainable_tokens: int = 0
    num_total_tokens: int = 0
    rollout_time: float = 0.0
    training_time: float = 0.0
    reward_std: float = 0.0
    groups_kept: int = 0
    groups_filtered: int = 0


class LinkSearchGRPOTrainer:
    """Custom GRPO Trainer for Link Search Agent."""
    
    def __init__(
        self,
        model,
        tokenizer,
        train_queries: List[LinkSearchQuery],
        eval_queries: List[LinkSearchQuery],
        policy_config: PolicyConfig,
        num_rollouts: int = 4,
        learning_rate: float = 1e-5,
        beta: float = 0.01,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        output_dir: str = "outputs/grpo_linksearch",
        target_accuracy: float = 0.80,
        eval_steps: int = 10,
        save_steps: int = 10,
        max_steps: int = 1000,
        warmup_steps: int = 10,
        patience: int = 5,
        min_group_std: float = 0.05,
        resume_from_checkpoint: Optional[str] = None,
        clip_epsilon: float = 0.2,
        rollout_concurrency: int = 4,
        eval_rollouts: int = 1,
        max_seq_length: Optional[int] = None,
        use_wandb: bool = False,
        run_baseline_eval: bool = True,
        enable_detailed_logging: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_queries = train_queries
        self.eval_queries = eval_queries
        self.policy_config = policy_config
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
        self.patience = patience
        self.min_group_std = min_group_std
        self.verbose = policy_config.verbose
        self.clip_epsilon = clip_epsilon
        self.rollout_concurrency = max(1, rollout_concurrency)
        self.eval_rollouts = max(1, eval_rollouts)
        self.max_seq_length = max_seq_length or getattr(
            getattr(self.model, "config", {}), "max_position_embeddings", 4096
        )
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.run_baseline_eval = run_baseline_eval
        self.enable_detailed_logging = enable_detailed_logging
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Reference model state for KL
        self.ref_model_state = None
        if beta > 0:
            logger.info("Saving reference model state for KL divergence...")
            self.ref_model_state = {
                name: param.detach().cpu().clone() 
                for name, param in model.named_parameters()
            }
        
        self.global_step = 0
        self.best_accuracy = 0.0
        self.best_model_path = None
        self.evals_without_improvement = 0
        
        # Resume if checkpoint provided
        if resume_from_checkpoint:
            self._load_checkpoint(Path(resume_from_checkpoint))
        
        logger.info("="*60)
        logger.info("LinkSearchGRPOTrainer initialized")
        logger.info("="*60)
        logger.info(f"Train queries: {len(train_queries)}")
        logger.info(f"Eval queries: {len(eval_queries)}")
        logger.info(f"Rollouts per query: {num_rollouts}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"KL penalty (beta): {beta}")
        logger.info(f"Target accuracy: {target_accuracy*100:.1f}%")
        logger.info("="*60)
    
    def _load_checkpoint(self, checkpoint_path: Path):
        """Load training state from checkpoint."""
        state_file = checkpoint_path / "training_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
            self.global_step = state.get("global_step", 0)
            self.best_accuracy = state.get("best_accuracy", 0.0)
            self.best_model_path = state.get("best_model_path")
            logger.info(f"Loaded training state: step={self.global_step}, best_acc={self.best_accuracy:.2%}")
    
    def _save_checkpoint(self, metrics: TrainingMetrics):
        """Save model checkpoint and training state."""
        checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(checkpoint_dir))
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "best_accuracy": self.best_accuracy,
            "best_model_path": str(self.best_model_path) if self.best_model_path else None,
            "metrics": {
                "loss": metrics.loss,
                "accuracy": metrics.accuracy,
                "avg_reward": metrics.avg_reward,
                "avg_score": metrics.avg_score,
            }
        }
        with open(checkpoint_dir / "training_state.json", 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_dir}")
        return checkpoint_dir
    
    async def _collect_rollouts(
        self,
        queries: List[LinkSearchQuery],
        is_eval: bool = False,
    ) -> List[TrajectoryGroup]:
        """Collect rollouts for a batch of queries."""
        groups = []
        all_rollout_logs = []
        
        num_rollouts = 1 if is_eval else self.num_rollouts
        
        for group_id, query in enumerate(queries):
            group = TrajectoryGroup(query=query, group_id=group_id)
            
            for rollout_idx in range(num_rollouts):
                try:
                    conversation, reward, rubric, rollout_log = await execute_rollout(
                        query=query,
                        model=self.model,
                        tokenizer=self.tokenizer,
                        policy_config=self.policy_config,
                        verbose=self.verbose,
                        log_turns=self.verbose,
                        rollout_index=rollout_idx,
                        num_rollouts=num_rollouts,
                        enable_detailed_logging=self.enable_detailed_logging,
                        training_step=self.global_step,
                    )
                    
                    sample = TrajectorySample(
                        query_id=query.id,
                        query=query,
                        conversation=conversation,
                        reward=reward,
                        rubric=rubric,
                        rollout_idx=rollout_idx,
                        group_id=group_id,
                        rollout_log=rollout_log,
                    )
                    group.samples.append(sample)
                    
                    if rollout_log:
                        all_rollout_logs.append(rollout_log)
                    
                except Exception as e:
                    logger.error(f"Rollout failed for query {query.id}: {e}")
                    continue
            
            if group.samples:
                groups.append(group)
        
        # Save rollout logs
        if self.enable_detailed_logging and all_rollout_logs:
            save_rollout_logs(all_rollout_logs, str(self.output_dir / "rollout_logs"))
        
        return groups
    
    def _compute_advantages(self, groups: List[TrajectoryGroup]) -> List[TrajectoryGroup]:
        """Compute GRPO advantages for each group."""
        for group in groups:
            rewards = [s.reward for s in group.samples]
            
            if len(rewards) < 2:
                for s in group.samples:
                    s.advantage = 0.0
                continue
            
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            
            # Filter low-variance groups
            if std_reward < self.min_group_std:
                for s in group.samples:
                    s.advantage = 0.0
                continue
            
            # Normalize advantages
            for s in group.samples:
                s.advantage = (s.reward - mean_reward) / (std_reward + 1e-8)
        
        return groups
    
    def _compute_action_advantage(
        self,
        msg: Dict,
        rubric: LinkSearchRubric,
        query: LinkSearchQuery,
        msg_idx: int,
        conversation: List[Dict],
        trajectory_advantage: float,
    ) -> float:
        """Compute action-level advantage for process rewards."""
        tool_calls = msg.get("tool_calls", [])
        
        if not tool_calls:
            return trajectory_advantage
        
        next_msg = conversation[msg_idx + 1] if msg_idx + 1 < len(conversation) else None
        gold_set = set(normalize_handle(h) for h in query.gold_handles)
        
        for tc in tool_calls:
            func_name = tc.get("function", {}).get("name")
            func_args_str = tc.get("function", {}).get("arguments", "{}")
            
            try:
                func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
            except:
                return -2.0
            
            if func_name == "search_profile":
                if next_msg and next_msg.get("role") == "tool":
                    try:
                        tool_result = json.loads(next_msg.get("content", "{}"))
                        if isinstance(tool_result, dict) and tool_result.get("success"):
                            rows = tool_result.get("rows", [])
                            
                            # Check if found correct handles
                            found_correct = False
                            for row in rows:
                                handle = row.get("handle") or row.get("linkedin_handle", "")
                                if normalize_handle(handle) in gold_set:
                                    found_correct = True
                                    break
                            
                            if found_correct:
                                return +1.0  # Found correct profile
                            elif len(rows) > 0:
                                return +0.1  # Found something
                            else:
                                return -0.2  # Zero results
                    except:
                        pass
                return trajectory_advantage * 0.5
            
            elif func_name == "read_profile":
                handle = func_args.get("linkedin_handle", "").lower()
                if normalize_handle(handle) in gold_set:
                    return +1.2  # Read correct profile
                else:
                    return -0.3  # Read wrong profile
            
            elif func_name == "return_results":
                results = func_args.get("results", {})
                if isinstance(results, dict):
                    num_correct = 0
                    for h in results.keys():
                        if normalize_handle(h) in gold_set:
                            num_correct += 1
                    
                    if num_correct > 0:
                        return +1.5 + 0.2 * num_correct
                    else:
                        return -0.5
                return trajectory_advantage
            
            else:
                return -2.0
        
        return trajectory_advantage * 0.5
    
    def tokenize_conversation_with_mask(
        self,
        conversation: List[Dict],
        rubric: Optional[LinkSearchRubric] = None,
        query: Optional[LinkSearchQuery] = None,
        trajectory_advantage: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize conversation and create loss mask with advantages."""
        all_tokens = []
        all_masks = []
        all_advantages = []
        
        for msg_idx, msg in enumerate(conversation):
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
                text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            if is_model_generated:
                mask = [1.0] * len(tokens)
                
                if rubric is not None and query is not None:
                    action_advantage = self._compute_action_advantage(
                        msg, rubric, query, msg_idx, conversation, trajectory_advantage
                    )
                else:
                    action_advantage = trajectory_advantage
                
                advantages = [action_advantage] * len(tokens)
            else:
                mask = [0.0] * len(tokens)
                advantages = [0.0] * len(tokens)
            
            all_tokens.extend(tokens)
            all_masks.extend(mask)
            all_advantages.extend(advantages)
        
        # Add EOS
        all_tokens.append(self.tokenizer.eos_token_id)
        all_masks.append(0.0)
        all_advantages.append(0.0)
        
        input_ids = torch.tensor(all_tokens, dtype=torch.long)
        labels = input_ids.clone()
        loss_mask = torch.tensor(all_masks, dtype=torch.float)
        advantage_mask = torch.tensor(all_advantages, dtype=torch.float)
        
        return input_ids, labels, loss_mask, advantage_mask
    
    def compute_loss_for_trajectory(
        self,
        conversation: List[Dict],
        advantage: float,
        rubric: Optional[LinkSearchRubric] = None,
        query: Optional[LinkSearchQuery] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a single trajectory."""
        input_ids, labels, loss_mask, advantage_mask = self.tokenize_conversation_with_mask(
            conversation, rubric, query, advantage
        )
        
        device = next(self.model.parameters()).device
        input_ids = input_ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        loss_mask = loss_mask.to(device)
        advantage_mask = advantage_mask.to(device)
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=labels)
        logits = outputs.logits
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = loss_mask[1:].contiguous()
        shift_advantage = advantage_mask[1:].contiguous()
        
        # Compute losses
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        masked_losses = token_losses * shift_mask * shift_advantage
        
        num_trainable = shift_mask.sum().item()
        
        if num_trainable == 0:
            policy_loss = torch.tensor(0.0, device=device)
        else:
            policy_loss = masked_losses.sum() / shift_mask.sum()
        
        # KL divergence (simplified - no ref model copy)
        kl_loss = torch.tensor(0.0, device=device)
        
        total_loss = policy_loss + kl_loss
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "num_trainable_tokens": num_trainable,
            "num_total_tokens": shift_mask.numel(),
        }
        
        return total_loss, metrics
    
    def _run_evaluation(self) -> TrainingMetrics:
        """Run evaluation on eval set."""
        logger.info("Running evaluation...")
        
        self.model.eval()
        
        loop = asyncio.get_event_loop()
        groups = loop.run_until_complete(
            self._collect_rollouts(self.eval_queries[:min(50, len(self.eval_queries))], is_eval=True)
        )
        
        rewards = []
        scores = []
        hits = []
        
        for group in groups:
            for sample in group.samples:
                rewards.append(sample.reward)
                scores.append(sample.rubric.score)
                hits.append(sample.rubric.num_correct_handles)
        
        avg_reward = np.mean(rewards) if rewards else 0.0
        avg_score = np.mean(scores) if scores else 0.0
        avg_hits = np.mean(hits) if hits else 0.0
        
        # Accuracy = percentage with score > 0.5
        accuracy = np.mean([1 if s > 0.5 else 0 for s in scores]) if scores else 0.0
        
        self.model.train()
        
        return TrainingMetrics(
            loss=0.0,
            policy_loss=0.0,
            kl_loss=0.0,
            avg_reward=avg_reward,
            max_reward=max(rewards) if rewards else 0.0,
            min_reward=min(rewards) if rewards else 0.0,
            accuracy=accuracy,
            avg_score=avg_score,
            avg_hits=avg_hits,
        )
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        self.model.train()
        
        query_idx = 0
        
        for step in range(self.global_step, self.max_steps):
            self.global_step = step + 1
            step_start = time.time()
            
            # Get batch of queries
            batch_queries = []
            for _ in range(self.batch_size):
                batch_queries.append(self.train_queries[query_idx % len(self.train_queries)])
                query_idx += 1
            
            # Collect rollouts
            rollout_start = time.time()
            loop = asyncio.get_event_loop()
            groups = loop.run_until_complete(self._collect_rollouts(batch_queries))
            rollout_time = time.time() - rollout_start
            
            # Compute advantages
            groups = self._compute_advantages(groups)
            
            # Training step
            train_start = time.time()
            
            total_loss = 0.0
            total_policy_loss = 0.0
            num_samples = 0
            
            self.optimizer.zero_grad()
            
            for group in groups:
                for sample in group.samples:
                    if sample.advantage is None or sample.advantage == 0:
                        continue
                    
                    loss, metrics = self.compute_loss_for_trajectory(
                        sample.conversation,
                        sample.advantage,
                        sample.rubric,
                        sample.query,
                    )
                    
                    if loss.item() != 0:
                        (loss / self.gradient_accumulation_steps).backward()
                        total_loss += loss.item()
                        total_policy_loss += metrics["policy_loss"]
                        num_samples += 1
            
            if num_samples > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            train_time = time.time() - train_start
            
            # Collect metrics
            all_rewards = [s.reward for g in groups for s in g.samples]
            all_scores = [s.rubric.score for g in groups for s in g.samples]
            
            metrics = TrainingMetrics(
                loss=total_loss / max(num_samples, 1),
                policy_loss=total_policy_loss / max(num_samples, 1),
                kl_loss=0.0,
                avg_reward=np.mean(all_rewards) if all_rewards else 0.0,
                max_reward=max(all_rewards) if all_rewards else 0.0,
                min_reward=min(all_rewards) if all_rewards else 0.0,
                accuracy=np.mean([1 if s > 0.5 else 0 for s in all_scores]) if all_scores else 0.0,
                avg_score=np.mean(all_scores) if all_scores else 0.0,
                rollout_time=rollout_time,
                training_time=train_time,
                groups_kept=sum(1 for g in groups if any(s.advantage and s.advantage != 0 for s in g.samples)),
                groups_filtered=sum(1 for g in groups if all(s.advantage is None or s.advantage == 0 for s in g.samples)),
            )
            
            # Log progress
            print(
                f"Step {self.global_step}/{self.max_steps} | "
                f"Loss: {metrics.loss:.4f} | "
                f"Score: {metrics.avg_score:.3f} | "
                f"Reward: {metrics.avg_reward:.3f} | "
                f"Rollout: {rollout_time:.1f}s | "
                f"Train: {train_time:.1f}s",
                flush=True
            )
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "train/loss": metrics.loss,
                    "train/policy_loss": metrics.policy_loss,
                    "train/avg_reward": metrics.avg_reward,
                    "train/avg_score": metrics.avg_score,
                    "train/accuracy": metrics.accuracy,
                    "train/rollout_time": rollout_time,
                    "train/training_time": train_time,
                }, step=self.global_step)
            
            # Evaluation
            if self.global_step % self.eval_steps == 0:
                eval_metrics = self._run_evaluation()
                
                print(
                    f"\n📊 Eval Step {self.global_step}: "
                    f"Score={eval_metrics.avg_score:.3f}, "
                    f"Accuracy={eval_metrics.accuracy:.2%}, "
                    f"Hits={eval_metrics.avg_hits:.2f}\n",
                    flush=True
                )
                
                if self.use_wandb:
                    wandb.log({
                        "eval/avg_score": eval_metrics.avg_score,
                        "eval/accuracy": eval_metrics.accuracy,
                        "eval/avg_reward": eval_metrics.avg_reward,
                        "eval/avg_hits": eval_metrics.avg_hits,
                    }, step=self.global_step)
                
                # Check for improvement
                if eval_metrics.accuracy > self.best_accuracy:
                    self.best_accuracy = eval_metrics.accuracy
                    self.evals_without_improvement = 0
                    self.best_model_path = self._save_checkpoint(eval_metrics)
                    print(f"🎯 New best accuracy: {self.best_accuracy:.2%}", flush=True)
                else:
                    self.evals_without_improvement += 1
                
                # Early stopping
                if self.evals_without_improvement >= self.patience:
                    print(f"Early stopping: no improvement for {self.patience} evaluations", flush=True)
                    break
                
                # Target reached
                if eval_metrics.accuracy >= self.target_accuracy:
                    print(f"🎉 Target accuracy {self.target_accuracy:.2%} reached!", flush=True)
                    self._save_checkpoint(eval_metrics)
                    break
            
            # Save checkpoint
            if self.global_step % self.save_steps == 0:
                self._save_checkpoint(metrics)
        
        # Final save
        final_dir = self.output_dir / "final"
        self.model.save_pretrained(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        
        print(f"\n✅ Training complete! Best accuracy: {self.best_accuracy:.2%}", flush=True)
        print(f"📁 Final model: {final_dir}", flush=True)
        if self.best_model_path:
            print(f"📁 Best model: {self.best_model_path}", flush=True)

