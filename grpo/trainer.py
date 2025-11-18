"""Custom GRPO Trainer with token-level masking."""

import asyncio
import json
import logging
import sys
import time
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from openai import AsyncOpenAI
from tqdm import tqdm

from email_agent.config import PolicyConfig
from email_agent.data import SyntheticQuery
from email_agent.rollout import EvaluationRubric

from grpo.reward_functions import execute_rollout
from grpo.utils import TrainingMetrics

logger = logging.getLogger(__name__)


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
        eval_steps: int = 2,
        save_steps: int = 2,
        max_steps: int = 1000,
        warmup_steps: int = 10,
        patience: int = 5,  # Early stopping: stop if no improvement for N evaluations
        min_group_std: float = 0.05,  # Minimum reward std to keep a group for training
        resume_from_checkpoint: Optional[str] = None,  # Path to checkpoint to resume from
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
        self.patience = patience
        self.min_group_std = min_group_std
        self.verbose = policy_config.verbose  # Control logging verbosity
        
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
            self.ref_model = deepcopy(model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            logger.info("✓ Reference model created")
        
        self.global_step = 0
        self.best_accuracy = 0.0
        self.best_model_path = None
        self.evals_without_improvement = 0  # For early stopping
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self._load_checkpoint(Path(resume_from_checkpoint))
        
        logger.info("="*60)
        logger.info("AgentGRPOTrainer initialized")
        logger.info("="*60)
        logger.info(f"Train queries: {len(train_queries)}")
        logger.info(f"Eval queries: {len(eval_queries)}")
        logger.info(f"Rollouts per query: {num_rollouts}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Total trajectories per step: {batch_size * num_rollouts}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"KL penalty (beta): {beta}")
        logger.info(f"Target accuracy: {target_accuracy*100:.1f}%")
        logger.info(f"Early stopping patience: {patience} evaluations")
        logger.info(f"Min group std for training: {min_group_std:.3f} (filters low-variance groups)")
        logger.info(f"Verbose logging: {self.verbose} (detailed rollout logs: {'enabled' if self.verbose else 'disabled'})")
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
    
    def _get_lora_name(self) -> Optional[str]:
        """Extract LoRA adapter name from model if available."""
        try:
            # Check if model has PEFT adapters
            if hasattr(self.model, 'peft_config'):
                adapter_names = list(self.model.peft_config.keys())
                if adapter_names:
                    return adapter_names[0]
            # Check for active adapters
            if hasattr(self.model, 'active_adapters'):
                adapters = self.model.active_adapters
                if adapters and len(adapters) > 0:
                    return adapters[0] if isinstance(adapters, list) else str(adapters)
            # Fallback: check if it's a PEFT model
            if hasattr(self.model, 'base_model'):
                return "LoRA"
        except:
            pass
        return None
    
    async def collect_rollouts_for_batch(
        self,
        queries: List[SyntheticQuery],
        log_rollouts: bool = False,
        step_num: int = 0,
    ) -> Tuple[List[List[Dict]], List[float], List[EvaluationRubric]]:
        """Collect rollouts for a batch of queries."""
        trajectories = []
        rewards = []
        rubrics = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        total_rollouts = len(queries) * self.num_rollouts
        batch_start_time = time.time()
        
        if log_rollouts:
            print(f"\n{'='*80}", flush=True)
            print(f"🎲 ROLLOUT COLLECTION (Step {step_num})", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"Batch: {len(queries)} queries × {self.num_rollouts} rollouts = {total_rollouts} trajectories", flush=True)
            print(f"{'='*80}\n", flush=True)
        
        with tqdm(total=total_rollouts, desc="🎲 Collecting rollouts", 
                  unit="rollout", leave=False, ncols=100, disable=log_rollouts) as pbar:
            for query_idx, query in enumerate(queries):
                for rollout_idx in range(self.num_rollouts):
                    rollout_start = time.time()
                    current_rollout = len(trajectories) + 1
                    elapsed = time.time() - batch_start_time
                    avg_time = elapsed / current_rollout if current_rollout > 0 else 0
                    
                    # Prepare rollout info for logging
                    # For training rollouts, indicate we're collecting data for this step
                    # The model weights are from the last saved checkpoint
                    model_identifier = f"training-step-{step_num}" if step_num > 0 else "initial"
                    rollout_info = {
                        'current_rollout': current_rollout,
                        'total_rollouts': total_rollouts,
                        'elapsed_time': elapsed,
                        'avg_rollout_time': avg_time,
                        'step': step_num,
                        'max_steps': self.max_steps,
                        'query_idx': query_idx,
                        'total_queries': len(queries),
                        'best_accuracy': self.best_accuracy,
                        'lora_name': model_identifier,
                        'is_training': True,
                    }
                    
                    conversation, reward, rubric = await execute_rollout(
                        query, self.model, self.tokenizer, 
                        self.policy_config, self.openai_client, 
                        verbose=False,
                        log_turns=log_rollouts,
                        rollout_info=rollout_info if log_rollouts else None,
                        rollout_index=rollout_idx,
                        num_rollouts=self.num_rollouts,
                    )
                    trajectories.append(conversation)
                    rewards.append(reward)
                    rubrics.append(rubric)
                    
                    # Accumulate token usage
                    total_input_tokens += rubric.total_input_tokens
                    total_output_tokens += rubric.total_output_tokens
                    
                    # Update progress bar with current stats
                    if not log_rollouts:
                        pbar.set_postfix({
                            'query': f'{query_idx+1}/{len(queries)}',
                            'reward': f'{reward:.2f}',
                            'avg': f'{np.mean(rewards):.2f}'
                        })
                        pbar.update(1)
        
        if log_rollouts:
            total_time = time.time() - batch_start_time
            avg_input_tokens = total_input_tokens / total_rollouts if total_rollouts > 0 else 0
            avg_output_tokens = total_output_tokens / total_rollouts if total_rollouts > 0 else 0
            
            print(f"\n{'='*80}", flush=True)
            print(f"✅ ROLLOUT COLLECTION COMPLETE", flush=True)
            print(f"Total time: {total_time:.1f}s | Avg per rollout: {total_time/total_rollouts:.1f}s", flush=True)
            print(f"Rewards: mean={np.mean(rewards):.2f}, std={np.std(rewards):.2f}, "
                  f"min={np.min(rewards):.2f}, max={np.max(rewards):.2f}", flush=True)
            print(f"Tokens: input={total_input_tokens:,} (avg {avg_input_tokens:.0f}/rollout), "
                  f"output={total_output_tokens:,} (avg {avg_output_tokens:.0f}/rollout)", flush=True)
            print(f"{'='*80}\n", flush=True)
        
        return trajectories, rewards, rubrics
    
    def compute_advantages(self, rewards: List[float], log_details: bool = False) -> Tuple[List[float], List[int]]:
        """Compute advantages using GRPO and filter out low-variance groups.
        
        Returns:
            advantages: List of advantages for kept trajectories
            kept_indices: Indices of trajectories to keep for training
        """
        rewards_array = np.array(rewards)
        num_groups = len(rewards) // self.num_rollouts
        advantages = []
        kept_indices = []
        
        groups_kept = 0
        groups_filtered = 0
        
        if log_details:
            logger.info(f"\n{'─'*80}")
            logger.info(f"📊 Group Advantages Computation (with filtering):")
            logger.info(f"{'─'*80}")
            sys.stdout.flush()  # Force flush
        
        for i in range(num_groups):
            start_idx = i * self.num_rollouts
            end_idx = start_idx + self.num_rollouts
            group_rewards = rewards_array[start_idx:end_idx]
            
            group_mean = group_rewards.mean()
            group_std = group_rewards.std()
            
            # Check if group has sufficient variance
            if group_std < self.min_group_std:
                # Skip this group - rewards are too similar
                if log_details:
                    logger.info(f"\nQuery Group {i+1}/{num_groups}: ⚠️ FILTERED (low variance)")
                    logger.info(f"  Rewards: {[f'{r:.3f}' for r in group_rewards]}")
                    logger.info(f"  Mean: {group_mean:.3f}, Std: {group_std:.3f} < {self.min_group_std:.3f}")
                    logger.info(f"  ❌ Skipping - insufficient reward variance for learning")
                groups_filtered += 1
                continue
            
            # Keep this group
            groups_kept += 1
            group_advantages = (group_rewards - group_mean) / (group_std + 1e-8)
            
            for j in range(self.num_rollouts):
                advantages.append(group_advantages[j])
                kept_indices.append(start_idx + j)
            
            if log_details:
                logger.info(f"\nQuery Group {i+1}/{num_groups}: ✅ KEPT")
                logger.info(f"  Rewards: {[f'{r:.3f}' for r in group_rewards]}")
                logger.info(f"  Mean: {group_mean:.3f}, Std: {group_std:.3f} >= {self.min_group_std:.3f}")
                logger.info(f"  Advantages: {[f'{a:+.3f}' for a in group_advantages]}")
                
                # Highlight best and worst
                best_idx = np.argmax(group_rewards)
                worst_idx = np.argmin(group_rewards)
                logger.info(f"  Best rollout: #{best_idx+1} (reward={group_rewards[best_idx]:.3f}, adv={group_advantages[best_idx]:+.3f})")
                logger.info(f"  Worst rollout: #{worst_idx+1} (reward={group_rewards[worst_idx]:.3f}, adv={group_advantages[worst_idx]:+.3f})")
        
        if log_details:
            logger.info(f"\n{'─'*80}")
            logger.info(f"📊 Filtering Summary:")
            logger.info(f"  Total groups: {num_groups}")
            logger.info(f"  Kept: {groups_kept} ({groups_kept/num_groups*100:.1f}%)")
            logger.info(f"  Filtered: {groups_filtered} ({groups_filtered/num_groups*100:.1f}%)")
            logger.info(f"  Trajectories for training: {len(kept_indices)}/{len(rewards)}")
            logger.info(f"{'─'*80}\n")
        
        return advantages, kept_indices
    
    def training_step(
        self,
        queries: List[SyntheticQuery],
        log_details: bool = False,
    ) -> TrainingMetrics:
        """Execute a single training step."""
        self.model.train()
        
        # Collect rollouts with timing
        if log_details:
            logger.info(f"\n{'─'*80}")
            logger.info(f"STEP PHASE 1: COLLECTING ROLLOUTS")
            logger.info(f"{'─'*80}")
            logger.info(f"Batch size: {len(queries)} queries")
            logger.info(f"Rollouts per query: {self.num_rollouts}")
            logger.info(f"Total trajectories: {len(queries) * self.num_rollouts}")
            sys.stdout.flush()  # Force flush
        else:
            logger.info(f"📊 Collecting {self.num_rollouts} rollouts for {len(queries)} queries...")
        
        rollout_start = time.time()
        trajectories, rewards, rubrics = asyncio.run(
            self.collect_rollouts_for_batch(queries, log_rollouts=log_details, step_num=self.global_step)
        )
        rollout_time = time.time() - rollout_start
        
        if log_details:
            logger.info(f"\n{'─'*80}")
            logger.info(f"STEP PHASE 2: COMPUTING ADVANTAGES & FILTERING")
            logger.info(f"{'─'*80}")
        
        # Compute advantages and filter out low-variance groups
        advantages, kept_indices = self.compute_advantages(rewards, log_details=log_details)
        
        # Filter trajectories to only those we're keeping
        filtered_trajectories = [trajectories[i] for i in kept_indices]
        filtered_rewards = [rewards[i] for i in kept_indices]
        
        if len(filtered_trajectories) == 0:
            logger.warning("⚠️  All groups filtered out! Skipping training step.")
            # Return zero metrics
            return TrainingMetrics(
                loss=0.0,
                policy_loss=0.0,
                kl_loss=0.0,
                avg_reward=np.mean(rewards),
                max_reward=np.max(rewards),
                min_reward=np.min(rewards),
                accuracy=sum(1 for r in rubrics if r.answer_correct) / len(rubrics),
                num_trainable_tokens=0,
                num_total_tokens=0,
                rollout_time=rollout_time,
                training_time=0.0,
                reward_std=np.std(rewards),
                median_reward=np.median(rewards),
            )
        
        # Training with timing
        if log_details:
            logger.info(f"\n{'─'*80}")
            logger.info(f"STEP PHASE 3: COMPUTING GRADIENTS")
            logger.info(f"{'─'*80}")
            logger.info(f"Computing loss for {len(filtered_trajectories)} trajectories (filtered from {len(trajectories)})...")
        elif len(filtered_trajectories) < len(trajectories):
            logger.info(f"🔍 Training on {len(filtered_trajectories)}/{len(trajectories)} trajectories "
                       f"({len(filtered_trajectories)/len(trajectories)*100:.1f}% kept after filtering)")
        
        training_start = time.time()
        total_loss = 0
        total_policy_loss = 0
        total_kl_loss = 0
        total_trainable_tokens = 0
        total_tokens = 0
        
        # Show token mask for first trajectory if logging details
        show_mask_for_first = log_details
        
        # Progress bar for gradient computation
        with tqdm(total=len(filtered_trajectories), desc="⚡ Computing gradients", 
                  unit="traj", leave=False, ncols=100, disable=log_details) as pbar:
            for i, (traj, adv) in enumerate(zip(filtered_trajectories, advantages)):
                log_mask = show_mask_for_first and i == 0
                loss, metrics = self.compute_loss_for_trajectory(
                    traj, adv, log_mask=log_mask, trajectory_idx=i
                )
                
                total_loss += loss / len(filtered_trajectories)
                total_policy_loss += metrics["policy_loss"]
                total_kl_loss += metrics["kl_loss"]
                total_trainable_tokens += metrics["num_trainable_tokens"]
                total_tokens += metrics["num_total_tokens"]
                
                if not log_details:
                    pbar.update(1)
        
        # Backpropagation
        if log_details:
            logger.info(f"\n{'─'*80}")
            logger.info(f"STEP PHASE 4: BACKPROPAGATION & OPTIMIZATION")
            logger.info(f"{'─'*80}")
            logger.info(f"Total loss: {total_loss.item():.4f}")
            logger.info(f"Max grad norm: {self.max_grad_norm}")
        else:
            logger.info("🔄 Backpropagating gradients...")
        
        total_loss.backward()
        
        if log_details:
            logger.info("✓ Backward pass complete")
        
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        if log_details:
            logger.info(f"✓ Gradient norm before clipping: {grad_norm:.4f}")
            logger.info(f"✓ Clipped to: {self.max_grad_norm}")
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        training_time = time.time() - training_start
        
        if log_details:
            logger.info(f"✓ Optimizer step complete")
            logger.info(f"{'─'*80}\n")
        
        # Compute metrics
        correct_answers = sum(1 for r in rubrics if r.answer_correct)
        accuracy = correct_answers / len(rubrics)
        
        metrics = TrainingMetrics(
            loss=total_loss.item(),
            policy_loss=total_policy_loss / len(filtered_trajectories),
            kl_loss=total_kl_loss / len(filtered_trajectories),
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
    
    async def evaluate(self, log_details: bool = False) -> Tuple[float, float]:
        """Evaluate the model."""
        self.model.eval()
        
        all_rewards = []
        correct_answers = 0
        all_rubrics = []
        
        logger.info(f"📊 Evaluating on {len(self.eval_queries)} queries...")
        eval_start = time.time()
        
        # Display current model info
        # At evaluation time, we're evaluating the checkpoint that was just saved
        model_identifier = f"{self.global_step}"
        logger.info(f"📦 Model info:")
        logger.info(f"   Evaluating checkpoint: {model_identifier}")
        logger.info(f"   Best accuracy so far: {self.best_accuracy*100:.2f}%")
        
        with torch.no_grad():
            if log_details:
                # Detailed logging like training
                print(f"\n{'='*80}", flush=True)
                print(f"🔍 EVALUATION ROLLOUTS", flush=True)
                print(f"{'='*80}", flush=True)
                print(f"Total queries: {len(self.eval_queries)}", flush=True)
                print(f"Checkpoint: {model_identifier}", flush=True)
                print(f"{'='*80}\n", flush=True)
                
                for query_idx, query in enumerate(self.eval_queries):
                    current_query = query_idx + 1
                    elapsed = time.time() - eval_start
                    avg_time = elapsed / current_query if current_query > 0 else 0
                    remaining = (len(self.eval_queries) - current_query) * avg_time
                    
                    # Prepare eval info for logging
                    eval_info = {
                        'current_rollout': current_query,
                        'total_rollouts': len(self.eval_queries),
                        'elapsed_time': elapsed,
                        'avg_rollout_time': avg_time,
                        'step': self.global_step,
                        'max_steps': self.max_steps,
                        'query_idx': query_idx,
                        'total_queries': len(self.eval_queries),
                        'best_accuracy': self.best_accuracy,
                        'lora_name': f"step-{model_identifier}",
                        'is_evaluation': True,
                    }
                    
                    conversation, reward, rubric = await execute_rollout(
                        query, self.model, self.tokenizer,
                        self.policy_config, self.openai_client, 
                        verbose=False,
                        log_turns=True,
                        rollout_info=eval_info,
                        rollout_index=0,  # Evaluation uses single rollout with base temperature
                        num_rollouts=1,
                    )
                    all_rewards.append(reward)
                    all_rubrics.append(rubric)
                    
                    if rubric.answer_correct:
                        correct_answers += 1
                
                print(f"\n{'='*80}", flush=True)
                print(f"✅ EVALUATION COMPLETE", flush=True)
                print(f"{'='*80}\n", flush=True)
            else:
                # Compact progress bar
                with tqdm(total=len(self.eval_queries), desc="🔍 Evaluation", 
                          unit="query", ncols=100) as pbar:
                    for query in self.eval_queries:
                        conversation, reward, rubric = await execute_rollout(
                            query, self.model, self.tokenizer,
                            self.policy_config, self.openai_client, verbose=False,
                            rollout_index=0,  # Evaluation uses single rollout with base temperature
                            num_rollouts=1,
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
        
        # Rubric statistics
        attempted = sum(1 for r in all_rubrics if r.attempted_answer)
        found_email = sum(1 for r in all_rubrics if r.ever_found_right_email)
        read_email = sum(1 for r in all_rubrics if r.ever_read_right_email)
        
        # Output detailed evaluation statistics (use print for consistency with detailed logging)
        if log_details:
            print(f"\n{'='*80}", flush=True)
            print(f"📊 EVALUATION RESULTS", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"🧩 Checkpoint: {model_identifier}", flush=True)
            print(f"📈 Best accuracy before: {self.best_accuracy*100:.2f}%", flush=True)
            print(f"🎯 Current accuracy: {accuracy*100:.2f}% ({correct_answers}/{len(self.eval_queries)})", flush=True)
            if accuracy > self.best_accuracy:
                improvement = (accuracy - self.best_accuracy) * 100
                print(f"   ✨ Improvement: +{improvement:.2f}%", flush=True)
            elif accuracy < self.best_accuracy:
                decline = (self.best_accuracy - accuracy) * 100
                print(f"   ⚠️  Decline: -{decline:.2f}%", flush=True)
            else:
                print(f"   ➖ No change", flush=True)
            print(f"\n⏱️  Time taken: {eval_time:.1f}s ({eval_time/len(self.eval_queries):.2f}s per query)", flush=True)
            print(f"💰 Average reward: {avg_reward:.3f} (std: {np.std(all_rewards):.3f})", flush=True)
            print(f"   Median reward: {np.median(all_rewards):.3f}", flush=True)
            print(f"   Reward range: [{np.min(all_rewards):.3f}, {np.max(all_rewards):.3f}]", flush=True)
            print(f"\n📊 Rubric Statistics:", flush=True)
            print(f"   Attempted answer: {attempted}/{len(all_rubrics)} ({attempted/len(all_rubrics)*100:.1f}%)", flush=True)
            print(f"   Found correct email: {found_email}/{len(all_rubrics)} ({found_email/len(all_rubrics)*100:.1f}%)", flush=True)
            print(f"   Read correct email: {read_email}/{len(all_rubrics)} ({read_email/len(all_rubrics)*100:.1f}%)", flush=True)
            print(f"{'='*80}\n", flush=True)
        
        # Also log to logger for file logging
        logger.info(f"\n{'─'*80}")
        logger.info(f"📊 EVALUATION RESULTS")
        logger.info(f"{'─'*80}")
        logger.info(f"Checkpoint: {model_identifier}")
        logger.info(f"Best accuracy before: {self.best_accuracy*100:.2f}%")
        logger.info(f"Current accuracy: {accuracy*100:.2f}% ({correct_answers}/{len(self.eval_queries)})")
        logger.info(f"Time taken: {eval_time:.1f}s ({eval_time/len(self.eval_queries):.2f}s per query)")
        logger.info(f"Average reward: {avg_reward:.3f} (std: {np.std(all_rewards):.3f})")
        logger.info(f"Median reward: {np.median(all_rewards):.3f}")
        logger.info(f"Reward range: [{np.min(all_rewards):.3f}, {np.max(all_rewards):.3f}]")
        logger.info(f"\nRubric Statistics:")
        logger.info(f"  - Attempted answer: {attempted}/{len(all_rubrics)} ({attempted/len(all_rubrics)*100:.1f}%)")
        logger.info(f"  - Found correct email: {found_email}/{len(all_rubrics)} ({found_email/len(all_rubrics)*100:.1f}%)")
        logger.info(f"  - Read correct email: {read_email}/{len(all_rubrics)} ({read_email/len(all_rubrics)*100:.1f}%)")
        logger.info(f"{'─'*80}\n")
        
        return avg_reward, accuracy
    
    def save_model(self, path: Path, metrics: Optional[Dict] = None):
        """Save model checkpoint with full training state."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "best_accuracy": self.best_accuracy,
            "best_model_path": str(self.best_model_path) if self.best_model_path else None,
            "evals_without_improvement": self.evals_without_improvement,
        }
        with open(path / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)
        
        # Save metrics metadata
        if metrics:
            metadata = {
                "step": self.global_step,
                "metrics": metrics,
            }
            with open(path / "training_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Model and training state saved to: {path}")
    
    def _load_checkpoint(self, checkpoint_path: Path):
        """Load training state from checkpoint."""
        logger.info(f"📂 Loading training state from checkpoint: {checkpoint_path}")
        
        # Load optimizer state
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path))
            logger.info("✓ Optimizer state loaded")
        else:
            logger.warning("⚠️  Optimizer state not found, starting with fresh optimizer")
        
        # Load training state
        training_state_path = checkpoint_path / "training_state.json"
        if training_state_path.exists():
            with open(training_state_path, 'r') as f:
                training_state = json.load(f)
            
            self.global_step = training_state.get("global_step", 0)
            self.best_accuracy = training_state.get("best_accuracy", 0.0)
            best_model_path_str = training_state.get("best_model_path")
            self.best_model_path = Path(best_model_path_str) if best_model_path_str else None
            self.evals_without_improvement = training_state.get("evals_without_improvement", 0)
            
            logger.info(f"✓ Training state loaded:")
            logger.info(f"  - Global step: {self.global_step}")
            logger.info(f"  - Best accuracy: {self.best_accuracy:.2%}")
            logger.info(f"  - Evals without improvement: {self.evals_without_improvement}")
        else:
            logger.warning("⚠️  Training state not found, starting from step 0")
    
    def train(self):
        """Main training loop."""
        logger.info("\n" + "="*80)
        logger.info("🚀 Starting GRPO Training with Token-Level Masking")
        logger.info("="*80)
        logger.info("\n📋 Training Process:")
        logger.info(f"  1. Load a batch of {self.batch_size} questions from dataset")
        logger.info(f"  2. Run agent {self.num_rollouts} times per question → {self.batch_size * self.num_rollouts} trajectories")
        logger.info(f"  3. Score all trajectories with reward function")
        logger.info(f"  4. Calculate GRPO loss and update model")
        logger.info(f"  5. Every {self.eval_steps} steps: evaluate on {len(self.eval_queries)} validation questions")
        logger.info(f"  6. Stop when model stops improving (patience: {self.patience} evals)")
        logger.info(f"\n🎯 Target accuracy: {self.target_accuracy*100:.1f}%")
        logger.info(f"📊 Max steps: {self.max_steps}")
        logger.info(f"📚 Train queries: {len(self.train_queries)}")
        logger.info(f"📝 Eval queries: {len(self.eval_queries)}")
        logger.info(f"🎲 Rollouts per query: {self.num_rollouts}")
        logger.info(f"📦 Batch size: {self.batch_size}")
        logger.info(f"📊 Total trajectories per step: {self.batch_size * self.num_rollouts}")
        logger.info(f"\n✅ Only model-generated tokens will be trained")
        logger.info(f"❌ Tool results will NOT be trained")
        logger.info("="*80)
        
        # Initialize step from global_step (for resume support)
        step = self.global_step
        if step > 0:
            logger.info(f"\n🔄 Resuming training from step {step}")
        
        training_start_time = time.time()
        cumulative_rollout_time = 0
        cumulative_training_time = 0
        
        # Track moving averages
        recent_rewards = []
        recent_accuracies = []
        window_size = 5
        
        while step < self.max_steps:
            step_start_time = time.time()
            
            # Update step counter at the beginning
            step += 1
            self.global_step = step
            
            # Use verbose setting to control logging detail level
            log_details = self.verbose
            
            if log_details:
                logger.info(f"\n{'='*80}")
                logger.info(f"📍 STEP {step}")
                logger.info(f"{'='*80}")
                sys.stdout.flush()  # Force flush to ensure immediate display
            else:
                logger.info(f"\n📍 Step {step}/{self.max_steps}")
            
            batch_queries = np.random.choice(
                self.train_queries,
                size=min(self.batch_size, len(self.train_queries)),
                replace=False
            ).tolist()
            
            metrics = self.training_step(batch_queries, log_details=log_details)
            
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
            if metrics.num_total_tokens > 0:
                logger.info(f"   Trainable: {metrics.num_trainable_tokens:,}/{metrics.num_total_tokens:,} "
                           f"({metrics.num_trainable_tokens/metrics.num_total_tokens*100:.1f}%)")
            else:
                logger.info(f"   Trainable: {metrics.num_trainable_tokens:,}/{metrics.num_total_tokens:,} (N/A - no tokens)")
            logger.info(f"{'─'*80}")
            
            # Evaluate
            if step % self.eval_steps == 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"🔍 EVALUATION at step {step}")
                logger.info(f"{'='*80}")
                
                # Step 1: Save checkpoint BEFORE evaluation
                checkpoint_path = self.output_dir / f"checkpoint-{step}"
                logger.info(f"\n💾 Saving checkpoint before evaluation: {checkpoint_path}")
                self.save_model(checkpoint_path, {"step": step})
                logger.info(f"✓ Checkpoint saved")
                
                # Step 2: Evaluate the model (use verbose setting)
                avg_reward, accuracy = asyncio.run(self.evaluate(log_details=self.verbose))
                
                # Step 3: Check if this is a new best model
                improvement = ""
                improved = False
                if accuracy > self.best_accuracy:
                    improvement = f" (+{(accuracy - self.best_accuracy)*100:.2f}%)"
                    old_best_accuracy = self.best_accuracy
                    self.best_accuracy = accuracy
                    self.best_model_path = self.output_dir / "best_model"
                    
                    # Save or update best model pointer
                    logger.info(f"\n✨ New best accuracy: {accuracy*100:.2f}%{improvement}")
                    logger.info(f"💾 Updating best model pointer: {self.best_model_path}")
                    self.save_model(
                        self.best_model_path,
                        {"accuracy": accuracy, "reward": avg_reward, "step": step}
                    )
                    logger.info(f"✓ Best model updated")
                    
                    improved = True
                    self.evals_without_improvement = 0  # Reset counter
                else:
                    self.evals_without_improvement += 1
                    logger.info(f"\n⚠️  No improvement for {self.evals_without_improvement} evaluation(s) "
                               f"(patience: {self.patience})")
                    logger.info(f"   Current accuracy: {accuracy*100:.2f}%")
                    logger.info(f"   Best accuracy: {self.best_accuracy*100:.2f}%")
                    logger.info(f"   Keeping best model at: {self.best_model_path}")
                
                progress_to_target = accuracy / self.target_accuracy * 100
                logger.info(f"\n📈 Progress to target: {progress_to_target:.1f}% "
                           f"({accuracy*100:.1f}% / {self.target_accuracy*100:.1f}%)")
                
                # Check if target accuracy reached
                if accuracy >= self.target_accuracy:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"🎉 TARGET ACCURACY REACHED!")
                    logger.info(f"🏆 Final accuracy: {accuracy*100:.2f}%")
                    logger.info(f"🎯 Target: {self.target_accuracy*100:.1f}%")
                    logger.info(f"⏱️  Total time: {str(timedelta(seconds=int(time.time() - training_start_time)))}")
                    logger.info(f"{'='*80}")
                    break
                
                # Check early stopping
                if self.evals_without_improvement >= self.patience:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"🛑 EARLY STOPPING!")
                    logger.info(f"No improvement for {self.evals_without_improvement} consecutive evaluations")
                    logger.info(f"Best accuracy: {self.best_accuracy*100:.2f}%")
                    logger.info(f"⏱️  Total time: {str(timedelta(seconds=int(time.time() - training_start_time)))}")
                    logger.info(f"{'='*80}")
                    break
                
                logger.info(f"{'='*80}\n")
            
            # Save checkpoint (only if not already saved during evaluation)
            if step % self.save_steps == 0 and step % self.eval_steps != 0:
                checkpoint_path = self.output_dir / f"checkpoint-{step}"
                self.save_model(checkpoint_path, {"step": step})
                logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Final evaluation
        total_training_time = time.time() - training_start_time
        logger.info(f"\n{'='*80}")
        logger.info("🏁 FINAL EVALUATION")
        logger.info(f"{'='*80}")
        
        # Final evaluation: always show details regardless of verbose setting
        avg_reward, accuracy = asyncio.run(self.evaluate(log_details=True))
        
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

