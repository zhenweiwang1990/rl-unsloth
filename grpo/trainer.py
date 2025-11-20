"""Custom GRPO Trainer with token-level masking."""

import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from openai import AsyncOpenAI
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from email_agent.config import PolicyConfig
from email_agent.data import SyntheticQuery
from email_agent.rollout import EvaluationRubric

from grpo.reward_functions import execute_rollout
from grpo.utils import TrainingMetrics


@dataclass
class TrajectorySample:
    """Single trajectory collected from a rollout."""

    query_id: str
    query: SyntheticQuery  # 新增：用于计算过程奖励
    conversation: List[Dict]
    reward: float
    rubric: EvaluationRubric
    rollout_idx: int
    group_id: int
    advantage: Optional[float] = None  # Group-level advantage for training
    turn_advantages: Optional[List[float]] = None  # Per-turn advantages (for detailed logging)


@dataclass
class TrajectoryGroup:
    """Grouped trajectories for GRPO (one group per query)."""

    query: SyntheticQuery
    group_id: int
    samples: List[TrajectorySample] = field(default_factory=list)


@dataclass
class TokenizedTrajectory:
    """Tokenized trajectory ready for batching/packing."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    attention_mask: torch.Tensor
    advantage_mask: torch.Tensor  # 改为token-level的advantage mask
    group_id: int
    query_id: str
    old_logprobs: Optional[torch.Tensor] = None

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
        eval_steps: int = 10,
        save_steps: int = 10,
        max_steps: int = 1000,
        warmup_steps: int = 10,
        patience: int = 5,  # Early stopping: stop if no improvement for N evaluations
        min_group_std: float = 0.05,  # Minimum reward std to keep a group for training
        resume_from_checkpoint: Optional[str] = None,  # Path to checkpoint to resume from
        clip_epsilon: float = 0.2,
        clip_epsilon_high: Optional[float] = None,
        rollout_concurrency: int = 4,
        eval_rollouts: int = 1,
        max_seq_length: Optional[int] = None,
        use_wandb: bool = False,
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
        self.clip_epsilon = clip_epsilon
        self.clip_epsilon_high = clip_epsilon if clip_epsilon_high is None else clip_epsilon_high
        self.rollout_concurrency = max(1, rollout_concurrency)
        self.eval_rollouts = max(1, eval_rollouts)
        self.max_seq_length = max_seq_length or getattr(
            getattr(self.model, "config", {}), "max_position_embeddings", policy_config.max_tokens * 4
        )
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Reference model for KL divergence (memory-efficient approach)
        # Instead of keeping a full model copy, we save only the initial state dict
        self.ref_model_state = None
        if beta > 0:
            logger.info("Saving reference model state for KL divergence...")
            logger.info("💡 Using memory-efficient approach: saving state_dict instead of full model copy")
            # Save initial model state (parameters only, no GPU memory overhead)
            self.ref_model_state = {
                name: param.detach().cpu().clone() 
                for name, param in model.named_parameters()
            }
            logger.info(f"✓ Reference state saved ({len(self.ref_model_state)} parameters, ~{sum(p.numel() for p in self.ref_model_state.values()) / 1e6:.1f}M params)")
        
        self.global_step = 0
        self.best_accuracy = 0.0
        self.best_model_path = None
        self.evals_without_improvement = 0  # For early stopping
        self.control_groups: Optional[List[TrajectoryGroup]] = None
        self.control_reward_map: Dict[str, List[float]] = {}
        
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
        logger.info(f"Wandb logging: {'enabled' if self.use_wandb else 'disabled'}")
        logger.info("="*60)
    
    def _compute_action_advantage(
        self,
        msg: Dict,
        rubric: EvaluationRubric,
        query: SyntheticQuery,
        msg_idx: int,
        conversation: List[Dict],
        trajectory_advantage: float,  # Group-level advantage作为baseline
    ) -> float:
        """计算单个action的过程奖励（token-level advantage）.
        
        这个方法为每个assistant action（tool call）计算独立的优势值，
        而不是使用统一的trajectory-level advantage。
        
        Args:
            msg: 当前消息（assistant的tool call）
            rubric: 评估指标，包含整个trajectory的表现
            query: 原始查询，包含ground truth
            msg_idx: 消息在conversation中的索引
            conversation: 完整的对话历史
            trajectory_advantage: Group-level的advantage，作为baseline
            
        Returns:
            该action的advantage值（可以是正或负）
        """
        tool_calls = msg.get("tool_calls", [])
        
        if not tool_calls:
            # 如果没有tool calls，使用trajectory-level advantage
            return trajectory_advantage
        
        # 获取下一条消息（tool result）来判断action效果
        next_msg = conversation[msg_idx + 1] if msg_idx + 1 < len(conversation) else None
        
        # 处理每个tool call（通常一个action只有一个tool call）
        for tc in tool_calls:
            func_name = tc.get("function", {}).get("name")
            func_args_str = tc.get("function", {}).get("arguments", "{}")
            
            # 解析参数
            try:
                func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
            except:
                # 解析错误，严重惩罚（比trajectory_advantage更负）
                return min(trajectory_advantage - 1.5, -2.0)
            
            # === 1. 搜索动作（search_emails）===
            if func_name == "search_emails":
                # 检查搜索结果
                if next_msg and next_msg.get("role") == "tool":
                    try:
                        tool_result = json.loads(next_msg.get("content", "[]"))
                        if isinstance(tool_result, list):
                            has_reference = False
                            result_count = len(tool_result)
                            
                            # 检查是否找到了正确的邮件
                            for result in tool_result:
                                if isinstance(result, dict) and result.get("message_id") == query.message_ids[0]:
                                    has_reference = True
                                    break
                            
                            # 找到了正确的邮件 → 给予正向奖励（独立于最终结果）
                            if has_reference:
                                # 这是好的搜索！即使最终答案错了，这步也是对的
                                return max(trajectory_advantage + 0.8, 0.3)  # 至少给0.3的正奖励
                            
                            # 搜索返回了结果但没找到正确邮件
                            if result_count > 0:
                                # 有结果但不对，轻微负面（说明搜索策略需要改进）
                                return trajectory_advantage - 0.2
                            else:
                                # 空结果，稍微负面（可能是正常的探索）
                                return trajectory_advantage - 0.15
                    except:
                        pass
                # 无法判断结果，使用baseline
                return trajectory_advantage
            
            # === 2. 阅读动作（read_email）===
            elif func_name == "read_email":
                msg_id = func_args.get("message_id", "")
                
                # 阅读了正确的邮件 → 给予正向奖励
                if msg_id == query.message_ids[0]:
                    # 这是好的阅读！即使最终答案错了，这步也是对的
                    return max(trajectory_advantage + 0.8, 0.3)  # 至少给0.3的正奖励
                else:
                    # 阅读了错误的邮件 → 轻微惩罚
                    return trajectory_advantage - 0.25
            
            # === 3. 最终答案（return_final_answer）===
            elif func_name == "return_final_answer":
                # 这是最关键的action，直接使用rubric判断
                if rubric.answer_correct:
                    # 正确答案 → 强正向奖励
                    return max(trajectory_advantage + 1.5, 1.0)  # 至少给1.0的强正奖励
                else:
                    # 错误答案 → 强负向惩罚
                    return min(trajectory_advantage - 1.5, -0.8)  # 至少给-0.8的惩罚
            
            # === 4. 错误的工具名称 ===
            elif func_name not in ["search_emails", "read_email", "return_final_answer"]:
                # 调用了不存在的工具 → 严重惩罚
                return min(trajectory_advantage - 1.2, -1.5)
        
        # 默认：使用trajectory-level advantage
        return trajectory_advantage
    
    def tokenize_conversation_with_mask(
        self,
        conversation: List[Dict],
        rubric: Optional[EvaluationRubric] = None,
        query: Optional[SyntheticQuery] = None,
        trajectory_advantage: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize conversation and create loss mask with process-based advantages.
        
        Args:
            conversation: 对话历史
            rubric: 评估指标（可选，用于计算过程奖励）
            query: 原始查询（可选，用于计算过程奖励）
            trajectory_advantage: Group-level advantage（用作baseline）
            
        Returns:
            input_ids, labels, loss_mask, advantage_mask
        """
        all_tokens = []
        all_masks = []
        all_advantages = []  # 新增：每个token的advantage值
        
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
                logger.warning(f"Unknown role: {role}")
                text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            
            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Create mask and advantages
            if is_model_generated:
                mask = [1.0] * len(tokens)
                
                # 计算这个action的过程奖励（token-level advantage）
                if rubric is not None and query is not None:
                    action_advantage = self._compute_action_advantage(
                        msg, rubric, query, msg_idx, conversation, trajectory_advantage
                    )
                else:
                    # 如果没有提供rubric和query，使用trajectory-level advantage
                    action_advantage = trajectory_advantage
                
                advantages = [action_advantage] * len(tokens)
            else:
                mask = [0.0] * len(tokens)
                advantages = [0.0] * len(tokens)  # 非assistant tokens的advantage设为0
            
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
        rubric: Optional[EvaluationRubric] = None,
        query: Optional[SyntheticQuery] = None,
        log_mask: bool = False,
        trajectory_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a single trajectory with token masking and process-based advantages."""
        input_ids, labels, loss_mask, advantage_mask = self.tokenize_conversation_with_mask(
            conversation, rubric, query, advantage
        )
        
        device = next(self.model.parameters()).device
        input_ids = input_ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        loss_mask = loss_mask.to(device)
        advantage_mask = advantage_mask.to(device)
        
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
        shift_advantage = advantage_mask[1:].contiguous()
        
        # Compute token losses
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Apply mask AND token-level advantages (process-based rewards)
        # 每个token的loss根据其对应action的advantage进行加权
        masked_losses = token_losses * shift_mask * shift_advantage
        
        num_trainable_tokens = shift_mask.sum().item()
        num_total_tokens = shift_mask.numel()
        
        if num_trainable_tokens == 0:
            policy_loss = torch.tensor(0.0, device=device)
        else:
            # 使用加权平均（已经在masked_losses中包含了advantage）
            policy_loss = masked_losses.sum() / shift_mask.sum()
        
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
        *,
        num_rollouts: Optional[int] = None,
        model_override=None,
        is_evaluation: bool = False,
    ) -> Tuple[List[TrajectoryGroup], Dict[str, float]]:
        """Collect rollouts (potentially in parallel) and organize them by group."""
        if not queries:
            return [], {}
        
        effective_rollouts = num_rollouts or (self.eval_rollouts if is_evaluation else self.num_rollouts)
        total_rollouts = len(queries) * effective_rollouts
        batch_start_time = time.time()
        collected_samples: List[TrajectorySample] = []
        group_queries: Dict[int, SyntheticQuery] = {}
        stats = {
            "total_rollouts": total_rollouts,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }
        
        if log_rollouts:
            print(f"\n{'='*80}", flush=True)
            mode = "EVAL" if is_evaluation else "TRAIN"
            print(f"🎲 ROLLOUT COLLECTION ({mode} Step {step_num})", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"Batch: {len(queries)} queries × {effective_rollouts} rollouts = {total_rollouts} trajectories", flush=True)
            print(f"{'='*80}\n", flush=True)
        
        semaphore = asyncio.Semaphore(self.rollout_concurrency) if not log_rollouts else None
        
        async def run_parallel() -> None:
            # Parallel with group-wise logging: collect each group in parallel,
            # then print detailed logs for that group before moving to the next
            rewards_tracker: List[float] = []
            with tqdm(
                total=total_rollouts,
                desc="🎲 Collecting rollouts",
                unit="rollout",
                leave=False,
                ncols=100,
                disable=log_rollouts,
            ) as pbar:
                for group_id, query in enumerate(queries):
                    group_queries[group_id] = query
                    
                    # Create tasks for this group
                    group_tasks = []
                    for rollout_idx in range(effective_rollouts):
                        group_tasks.append(
                            asyncio.create_task(
                                self._run_single_rollout(
                                    query=query,
                                    group_id=group_id,
                                    rollout_idx=rollout_idx,
                                    log_rollouts=False,  # Don't log during collection
                                    step_num=step_num,
                                    semaphore=semaphore,
                                    model_override=model_override,
                                    is_evaluation=is_evaluation,
                                )
                            )
                        )
                    
                    # Collect this group in parallel
                    group_samples = await asyncio.gather(*group_tasks)
                    
                    # Add to results
                    for sample in group_samples:
                        collected_samples.append(sample)
                        rewards_tracker.append(sample.reward)
                        stats["total_input_tokens"] += sample.rubric.total_input_tokens
                        stats["total_output_tokens"] += sample.rubric.total_output_tokens
                    
                    # Update progress
                    if not log_rollouts:
                        pbar.set_postfix(
                            {
                                "reward": f"{group_samples[-1].reward:.2f}",
                                "avg": f"{np.mean(rewards_tracker):.2f}",
                            }
                        )
                        pbar.update(len(group_samples))
                    else:
                        # Pause progress bar and print detailed logs for this group
                        pbar.clear()
                        self._print_group_details(query, group_samples, group_id, len(queries), step_num)
                        pbar.refresh()
                        pbar.update(len(group_samples))
        
        async def run_sequential() -> None:
            # Fully sequential mode (not used in group-wise logging)
            current_rollout = 0
            for group_id, query in enumerate(queries):
                group_queries[group_id] = query
                for rollout_idx in range(effective_rollouts):
                    current_rollout += 1
                    sample = await self._run_single_rollout(
                        query=query,
                        group_id=group_id,
                        rollout_idx=rollout_idx,
                        log_rollouts=log_rollouts,
                        step_num=step_num,
                        semaphore=None,
                        model_override=model_override,
                        is_evaluation=is_evaluation,
                        total_rollouts=total_rollouts,
                        current_rollout=current_rollout,
                    )
                    collected_samples.append(sample)
                    stats["total_input_tokens"] += sample.rubric.total_input_tokens
                    stats["total_output_tokens"] += sample.rubric.total_output_tokens
        
        # Always use parallel mode with group-wise logging
        # This gives us both speed (parallel within group) and detailed logs (after each group)
        await run_parallel()
        
        groups_dict: Dict[int, TrajectoryGroup] = {}
        for sample in sorted(collected_samples, key=lambda s: (s.group_id, s.rollout_idx)):
            if sample.group_id not in groups_dict:
                groups_dict[sample.group_id] = TrajectoryGroup(
                    query=group_queries[sample.group_id],
                    group_id=sample.group_id,
                    samples=[],
                )
            groups_dict[sample.group_id].samples.append(sample)
        
        if log_rollouts and total_rollouts > 0:
            total_time = time.time() - batch_start_time
            avg_input_tokens = stats["total_input_tokens"] / total_rollouts
            avg_output_tokens = stats["total_output_tokens"] / total_rollouts
            rewards = [sample.reward for sample in collected_samples]
            print(f"\n{'='*80}", flush=True)
            print(f"✅ ROLLOUT COLLECTION COMPLETE", flush=True)
            print(f"Total time: {total_time:.1f}s | Avg per rollout: {total_time/total_rollouts:.1f}s", flush=True)
            if rewards:
                print(
                    f"Rewards: mean={np.mean(rewards):.2f}, std={np.std(rewards):.2f}, "
                    f"min={np.min(rewards):.2f}, max={np.max(rewards):.2f}",
                    flush=True,
                )
            print(
                f"Tokens: input={stats['total_input_tokens']:,} (avg {avg_input_tokens:.0f}/rollout), "
                f"output={stats['total_output_tokens']:,} (avg {avg_output_tokens:.0f}/rollout)",
                flush=True,
            )
            print(f"{'='*80}\n", flush=True)
        
        return list(groups_dict.values()), stats
    
    def _compute_turn_advantages(
        self,
        sample: TrajectorySample,
        trajectory_advantage: float,
    ) -> List[float]:
        """计算每一轮的 advantage 值（用于详细日志展示）。
        
        Args:
            sample: Trajectory sample
            trajectory_advantage: Group-level advantage baseline
            
        Returns:
            List of advantages for each turn
        """
        turn_advantages = []
        conversation = sample.conversation
        rubric = sample.rubric
        query = sample.query
        
        # 遍历对话，找到所有 assistant 的 turn
        for msg_idx, msg in enumerate(conversation):
            role = msg.get("role", "")
            if role == "assistant":
                # 计算这个 turn 的 advantage
                action_advantage = self._compute_action_advantage(
                    msg, rubric, query, msg_idx, conversation, trajectory_advantage
                )
                turn_advantages.append(action_advantage)
        
        return turn_advantages

    def _print_turn_advantage_table(
        self,
        sorted_samples: List["TrajectorySample"],
    ) -> None:
        """打印每个 rollout 的每轮 advantage 变化表格。
        
        Args:
            sorted_samples: Sorted list of trajectory samples
        """
        # 找出最大 turn 数
        max_turns = max((s.rubric.num_turns for s in sorted_samples), default=0)
        if max_turns == 0:
            return
        
        print(f"\n{'─'*80}", flush=True)
        print(f"📊 Turn-by-Turn Advantage Table:", flush=True)
        print(f"{'─'*80}", flush=True)
        
        # 打印表头
        header = "Rollout |"
        for turn in range(1, max_turns + 1):
            header += f" Turn {turn:2d} |"
        print(header, flush=True)
        print("─" * len(header), flush=True)
        
        # 打印每个 rollout 的 advantage
        for sample in sorted_samples:
            if sample.turn_advantages is None:
                continue
            
            row = f"   {sample.rollout_idx + 1:2d}  |"
            for turn_idx in range(max_turns):
                if turn_idx < len(sample.turn_advantages):
                    adv = sample.turn_advantages[turn_idx]
                    row += f" {adv:+6.3f} |"
                else:
                    row += "   ---  |"
            print(row, flush=True)
        
        print(f"{'─'*80}", flush=True)

    def _print_group_details(
        self,
        query: SyntheticQuery,
        group_samples: List["TrajectorySample"],
        group_id: int,
        total_groups: int,
        step_num: int,
    ) -> None:
        """Print detailed logs for a completed group."""
        print(f"\n{'='*80}", flush=True)
        print(f"📊 GROUP {group_id+1}/{total_groups} COMPLETED", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"❓ Question: {query.question}", flush=True)
        print(f"✅ Ground Truth: {query.answer}", flush=True)
        print(f"📧 Reference Email: {query.message_ids[0]}", flush=True)
        print(f"", flush=True)
        
        # Sort by rollout index for consistent display
        sorted_samples = sorted(group_samples, key=lambda s: s.rollout_idx)
        
        for sample in sorted_samples:
            self._print_trajectory_summary(sample, query)
        
        # Group statistics
        rewards = [s.reward for s in sorted_samples]
        correct = sum(1 for s in sorted_samples if s.rubric.answer_correct)
        print(f"\n{'─'*80}", flush=True)
        print(f"📈 Group Statistics:", flush=True)
        print(f"   Rewards: mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}, "
              f"range=[{np.min(rewards):.3f}, {np.max(rewards):.3f}]", flush=True)
        print(f"   Correct answers: {correct}/{len(sorted_samples)} ({correct/len(sorted_samples)*100:.1f}%)", flush=True)
        
        # 打印每轮 advantage 表格
        self._print_turn_advantage_table(sorted_samples)
        
        print(f"{'='*80}\n", flush=True)
    
    def _print_trajectory_summary(self, sample: "TrajectorySample", query: SyntheticQuery) -> None:
        """Print detailed turn-by-turn logs for a single trajectory."""
        rubric = sample.rubric
        conversation = sample.conversation
        
        print(f"  🎲 Rollout {sample.rollout_idx+1}:", flush=True)
        
        # Print turn-by-turn actions
        turn_num = 0
        agent_final_answer = None
        
        for i, msg in enumerate(conversation[2:], start=2):  # Skip system and initial user
            role = msg.get('role', '')
            
            if role == 'assistant':
                turn_num += 1
                tool_calls = msg.get('tool_calls', [])
                content = msg.get('content', '')
                
                if tool_calls:
                    for tc in tool_calls:
                        func_name = tc.get('function', {}).get('name', 'unknown')
                        func_args_str = tc.get('function', {}).get('arguments', '{}')
                        
                        try:
                            func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
                            
                            # Extract key parameters for display
                            if func_name == 'search_emails':
                                keywords = func_args.get('keywords', [])
                                from_addr = func_args.get('from_addr', '')
                                # Note: inbox is not in func_args, it's added by the agent from query.inbox_address
                                # Include it in the log for debugging
                                params = f"keywords={keywords}"
                                if from_addr:
                                    params += f", from_addr={from_addr[:20]}"
                                # Add inbox to make it clear when searches differ
                                params += f", inbox={query.inbox_address[:30]}"
                                
                                # Check next message (tool result) to get search results
                                search_result_count = 0
                                has_reference = False
                                if i + 1 < len(conversation):
                                    next_msg = conversation[i + 1]
                                    if next_msg.get('role') == 'tool':
                                        try:
                                            tool_result = json.loads(next_msg.get('content', '[]'))
                                            if isinstance(tool_result, list):
                                                search_result_count = len(tool_result)
                                                # Check if reference email is in results
                                                for result in tool_result:
                                                    if isinstance(result, dict) and result.get('message_id') == query.message_ids[0]:
                                                        has_reference = True
                                                        break
                                        except:
                                            pass
                                
                                ref_indicator = " ✅ has reference" if has_reference else " ❌ no reference"
                                print(f"    🔧 Turn {turn_num}: {func_name}({params}) → {search_result_count} results{ref_indicator}", flush=True)
                                
                            elif func_name == 'read_email':
                                msg_id = func_args.get('message_id', '')
                                params = f"msg_id={msg_id[:20]}"
                                is_reference = (msg_id == query.message_ids[0])
                                ref_indicator = " ✅ reference email" if is_reference else " ❌ not reference"
                                print(f"    🔧 Turn {turn_num}: {func_name}({params}){ref_indicator}", flush=True)
                                
                            elif func_name == 'return_final_answer':
                                answer = func_args.get('answer', '')
                                agent_final_answer = answer
                                sources = func_args.get('source_message_ids', [])
                                params = f"answer='{answer[:50]}...', sources={len(sources)}"
                                print(f"    🔧 Turn {turn_num}: {func_name}({params})", flush=True)
                            else:
                                params = str(func_args)[:50]
                                print(f"    🔧 Turn {turn_num}: {func_name}({params})", flush=True)
                        except:
                            print(f"    🔧 Turn {turn_num}: {func_name}(...)", flush=True)
                            
                elif content:
                    print(f"    💬 Turn {turn_num}: text response ({len(content)} chars)", flush=True)
        
        # Print outcome
        if agent_final_answer:
            judge_result = "✅ CORRECT" if rubric.answer_correct else "❌ WRONG"
            print(f"    🤖 Agent Answer: {agent_final_answer[:60]}... → {judge_result}", flush=True)
        else:
            print(f"    🤖 Agent Answer: (no answer provided) → ❌ WRONG", flush=True)
        
        # Print metrics
        print(f"    📊 Metrics: reward={sample.reward:.3f}, turns={rubric.num_turns}, "
              f"found_email={rubric.ever_found_right_email}, read_email={rubric.ever_read_right_email}", flush=True)
        print(f"", flush=True)
    
    async def _run_single_rollout(
        self,
        query: SyntheticQuery,
        group_id: int,
        rollout_idx: int,
        *,
        log_rollouts: bool,
        step_num: int,
        semaphore: Optional[asyncio.Semaphore],
        model_override,
        is_evaluation: bool,
        total_rollouts: Optional[int] = None,
        current_rollout: Optional[int] = None,
    ) -> TrajectorySample:
        """Execute a single rollout (optionally under a semaphore)."""
        
        async def _exec() -> TrajectorySample:
            model_identifier = f"step-{step_num}" if step_num > 0 else "initial"
            rollout_info = None
            if log_rollouts:
                rollout_info = {
                    "current_rollout": current_rollout,
                    "total_rollouts": total_rollouts,
                    "elapsed_time": None,
                    "avg_rollout_time": None,
                    "step": step_num,
                    "max_steps": self.max_steps,
                    "query_idx": group_id,
                    "total_queries": None,
                    "best_accuracy": self.best_accuracy,
                    "lora_name": model_identifier,
                    "is_training": not is_evaluation,
                    "is_evaluation": is_evaluation,
                }
            
            conversation, reward, rubric = await execute_rollout(
                query=query,
                model=model_override or self.model,
                tokenizer=self.tokenizer,
                policy_config=self.policy_config,
                openai_client=self.openai_client,
                verbose=False,
                log_turns=log_rollouts,
                rollout_info=rollout_info,
                rollout_index=rollout_idx,
                num_rollouts=self.num_rollouts if not is_evaluation else self.eval_rollouts,
            )
            
            return TrajectorySample(
                query_id=query.id,
                query=query,  # 添加query，用于计算process-based rewards
                conversation=conversation,
                reward=reward,
                rubric=rubric,
                rollout_idx=rollout_idx,
                group_id=group_id,
            )
        
        if semaphore is None:
            return await _exec()
        async with semaphore:
            return await _exec()
    
    def compute_advantages(
        self,
        groups: List[TrajectoryGroup],
        log_details: bool = False,
    ) -> Tuple[List[TrajectorySample], Dict[str, int]]:
        """Compute advantages per group and drop low-variance groups."""
        kept_samples: List[TrajectorySample] = []
        groups_kept = 0
        groups_filtered = 0
        
        if log_details:
            logger.info(f"\n{'─'*80}")
            logger.info("📊 Group Advantages Computation (with filtering):")
            logger.info(f"{'─'*80}")
            sys.stdout.flush()
        
        for idx, group in enumerate(groups):
            if not group.samples:
                continue
            
            group_rewards = np.array([sample.reward for sample in group.samples], dtype=np.float32)
            group_mean = float(group_rewards.mean())
            group_std = float(group_rewards.std())
            
            if group_std < self.min_group_std:
                groups_filtered += 1
                if log_details:
                    logger.info(f"\nQuery Group {idx+1}/{len(groups)}: ⚠️ FILTERED (low variance)")
                    logger.info(f"  Rewards: {[f'{r:.3f}' for r in group_rewards]}")
                    logger.info(f"  Mean: {group_mean:.3f}, Std: {group_std:.3f} < {self.min_group_std:.3f}")
                continue
            
            groups_kept += 1
            group_advantages = (group_rewards - group_mean) / (group_std + 1e-8)
            
            for sample, advantage in zip(group.samples, group_advantages):
                sample.advantage = float(advantage)
                # 计算每轮的 advantage（用于详细日志）
                sample.turn_advantages = self._compute_turn_advantages(sample, float(advantage))
                kept_samples.append(sample)
            
            if log_details:
                logger.info(f"\nQuery Group {idx+1}/{len(groups)}: ✅ KEPT")
                logger.info(f"  Rewards: {[f'{r:.3f}' for r in group_rewards]}")
                logger.info(f"  Mean: {group_mean:.3f}, Std: {group_std:.3f} >= {self.min_group_std:.3f}")
                logger.info(f"  Advantages: {[f'{a:+.3f}' for a in group_advantages]}")
        
        if log_details:
            total_groups = max(len(groups), 1)
            logger.info(f"\n{'─'*80}")
            logger.info("📊 Filtering Summary:")
            logger.info(f"  Total groups: {len(groups)}")
            logger.info(f"  Kept: {groups_kept} ({groups_kept/total_groups*100:.1f}%)")
            logger.info(f"  Filtered: {groups_filtered} ({groups_filtered/total_groups*100:.1f}%)")
            logger.info(f"  Trajectories for training: {len(kept_samples)}")
            logger.info(f"{'─'*80}\n")
        
        return kept_samples, {"groups_kept": groups_kept, "groups_filtered": groups_filtered}
    
    def _tokenize_samples(self, samples: List[TrajectorySample]) -> List[TokenizedTrajectory]:
        """Tokenize trajectories and enforce max sequence length with process-based advantages."""
        tokenized: List[TokenizedTrajectory] = []
        max_len = self.max_seq_length or 32768
        
        for sample in samples:
            if sample.advantage is None:
                continue
            
            # 使用process-based rewards: 传入rubric和query来计算token-level advantages
            input_ids, labels, loss_mask, advantage_mask = self.tokenize_conversation_with_mask(
                sample.conversation,
                rubric=sample.rubric,
                query=sample.query,
                trajectory_advantage=sample.advantage,  # Group-level advantage作为baseline
            )
            seq_len = input_ids.size(0)
            if seq_len < 2:
                continue
            
            trunc_len = min(seq_len, max_len)
            input_ids = input_ids[:trunc_len].cpu()
            labels = labels[:trunc_len].cpu()
            loss_mask = loss_mask[:trunc_len].cpu()
            advantage_mask = advantage_mask[:trunc_len].cpu()  # 截断advantage_mask
            
            if loss_mask[1:].sum() == 0:
                continue
            
            attention_mask = torch.ones_like(input_ids)
            
            tokenized.append(
                TokenizedTrajectory(
                    input_ids=input_ids,
                    labels=labels,
                    loss_mask=loss_mask,
                    attention_mask=attention_mask,
                    advantage_mask=advantage_mask,  # 使用token-level的advantage_mask
                    group_id=sample.group_id,
                    query_id=sample.query_id,
                )
            )
        
        return tokenized
    
    def _populate_old_logprobs(
        self,
        tokenized_samples: List[TokenizedTrajectory],
        chunk_size: int = 2,
    ) -> None:
        """Compute old logprobs for tokenized samples using current model."""
        if not tokenized_samples:
            return
        
        device = next(self.model.parameters()).device
        training_mode = self.model.training
        self.model.eval()
        
        with torch.no_grad():
            for start in range(0, len(tokenized_samples), chunk_size):
                chunk = tokenized_samples[start:start + chunk_size]
                padded_input_ids = pad_sequence(
                    [sample.input_ids for sample in chunk],
                    batch_first=True,
                    padding_value=self.pad_token_id,
                ).to(device)
                attention_mask = pad_sequence(
                    [sample.attention_mask for sample in chunk],
                    batch_first=True,
                    padding_value=0,
                ).to(device)
                
                outputs = self.model(input_ids=padded_input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, :-1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                shift_labels = padded_input_ids[:, 1:]
                seq_logprobs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                
                for idx, sample in enumerate(chunk):
                    seq_len = sample.input_ids.size(0)
                    if seq_len <= 1:
                        sample.old_logprobs = torch.empty(0)
                        continue
                    sample.old_logprobs = seq_logprobs[idx, : seq_len - 1].detach().cpu()
        
        if training_mode:
            self.model.train()
    
    def _pack_tokenized_samples(
        self,
        tokenized_samples: List[TokenizedTrajectory],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Pack tokenized trajectories into padded tensors for training."""
        if not tokenized_samples:
            return None
        
        pad_id = self.pad_token_id
        max_seq = min(
            self.max_seq_length or 32768,
            max(sample.input_ids.size(0) for sample in tokenized_samples),
        )
        
        input_list: List[torch.Tensor] = []
        label_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        old_logprob_list: List[torch.Tensor] = []
        advantage_list: List[torch.Tensor] = []
        weight_list: List[torch.Tensor] = []
        
        for sample in tokenized_samples:
            if sample.old_logprobs is None:
                continue
            
            trunc_len = min(sample.input_ids.size(0), max_seq)
            if trunc_len <= 1:
                continue
            
            trunc_input = sample.input_ids[:trunc_len]
            trunc_labels = sample.labels[:trunc_len]
            trunc_mask = sample.loss_mask[:trunc_len]
            trunc_advantage = sample.advantage_mask[:trunc_len]  # 使用token-level的advantage_mask
            shift_mask = trunc_mask[1:]
            shift_advantage = trunc_advantage[1:]  # Shift advantage mask to match logprobs
            
            if shift_mask.sum() == 0:
                continue
            
            trunc_old_logprobs = sample.old_logprobs[: trunc_len - 1]
            weight = 1.0 / (shift_mask.sum() + 1e-6)
            
            input_list.append(trunc_input)
            label_list.append(trunc_labels)
            mask_list.append(trunc_mask)
            old_logprob_list.append(trunc_old_logprobs)
            # 使用token-level的advantage（process-based rewards）
            advantage_list.append(shift_advantage[:len(trunc_old_logprobs)])
            weight_list.append(torch.full_like(trunc_old_logprobs, weight))
        
        if not input_list:
            return None
        
        input_ids = pad_sequence(input_list, batch_first=True, padding_value=pad_id)
        labels = pad_sequence(label_list, batch_first=True, padding_value=pad_id)
        attention_mask = (input_ids != pad_id).long()
        loss_mask = pad_sequence(mask_list, batch_first=True, padding_value=0.0)
        assistant_mask = loss_mask[:, 1:]
        old_logprobs = pad_sequence(old_logprob_list, batch_first=True, padding_value=0.0)
        advantages = pad_sequence(advantage_list, batch_first=True, padding_value=0.0)
        weights = pad_sequence(weight_list, batch_first=True, padding_value=0.0)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "assistant_mask": assistant_mask,
            "old_logprobs": old_logprobs,
            "advantages": advantages,
            "weights": weights,
        }
    
    def _compute_grpo_loss(self, batch_tensors: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO/GRPO loss with clipping and optional KL penalty."""
        device = next(self.model.parameters()).device
        input_ids = batch_tensors["input_ids"].to(device)
        attention_mask = batch_tensors["attention_mask"].to(device)
        labels = batch_tensors["labels"].to(device)
        assistant_mask = batch_tensors["assistant_mask"].to(device)
        old_logprobs = batch_tensors["old_logprobs"].to(device)
        advantages = batch_tensors["advantages"].to(device)
        weights = batch_tensors["weights"].to(device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        new_logprobs_full = F.log_softmax(logits, dim=-1)
        new_logprobs = new_logprobs_full.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        prob_ratio = torch.exp(new_logprobs - old_logprobs)
        clipped_ratio = torch.clamp(prob_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon_high)
        surrogate_1 = prob_ratio * advantages
        surrogate_2 = clipped_ratio * advantages
        policy_loss = -torch.min(surrogate_1, surrogate_2)
        
        mask = assistant_mask.to(new_logprobs.dtype)
        weighted_mask = mask * weights
        denom = weighted_mask.sum() + 1e-6
        policy_loss = (policy_loss * weighted_mask).sum() / denom
        
        kl_loss = torch.tensor(0.0, device=device)
        if self.beta > 0.0 and self.ref_model_state is not None:
            # Compute KL divergence using saved reference parameters
            # Temporarily load ref params into model, compute forward pass, then restore
            with torch.no_grad():
                # Save current model state
                current_state = {name: param.data.clone() for name, param in self.model.named_parameters()}
                
                # Load reference state into model
                for name, param in self.model.named_parameters():
                    if name in self.ref_model_state:
                        param.data.copy_(self.ref_model_state[name].to(device))
                
                # Forward pass with reference parameters
                ref_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits[:, :-1, :]
                ref_logprobs_full = F.log_softmax(ref_logits, dim=-1)
                ref_logprobs = ref_logprobs_full.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                
                # Restore current model state
                for name, param in self.model.named_parameters():
                    if name in current_state:
                        param.data.copy_(current_state[name])
            
            kl_values = torch.exp(ref_logprobs - new_logprobs) - (ref_logprobs - new_logprobs) - 1.0
            kl_loss = (kl_values * weighted_mask).sum() / denom
        
        entropy = -(torch.exp(new_logprobs_full) * new_logprobs_full).sum(-1)
        mean_entropy = (entropy * mask).sum() / (mask.sum() + 1e-6)
        
        total_loss = policy_loss + self.beta * kl_loss
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item() if torch.is_tensor(kl_loss) else float(kl_loss),
            "entropy": mean_entropy.item(),
            "trainable_tokens": int(mask.sum().item()),
            "total_tokens": int(attention_mask.sum().item()),
        }
        
        return total_loss, metrics
    
    def _build_reward_map(self, groups: List[TrajectoryGroup]) -> Dict[str, List[float]]:
        reward_map: Dict[str, List[float]] = defaultdict(list)
        for group in groups:
            rewards = [sample.reward for sample in group.samples]
            if rewards:
                reward_map[group.query.id].extend(rewards)
        return reward_map
    
    def _ensure_control_baseline(self):
        """Collect baseline evaluation groups for beat-rate comparisons."""
        if self.control_groups is not None:
            return
        logger.info("🎯 Collecting control baseline for evaluation comparisons...")
        # Use verbose setting for control baseline too
        control_groups, _ = asyncio.run(
            self.collect_rollouts_for_batch(
                self.eval_queries,
                log_rollouts=self.verbose,  # Show detailed logs if verbose is enabled
                step_num=-1,
                num_rollouts=self.eval_rollouts,
                is_evaluation=True,
            )
        )
        self.control_groups = control_groups
        self.control_reward_map = self._build_reward_map(control_groups)
        logger.info("✓ Control baseline collected")
    
    def _calculate_control_beat_rate(
        self,
        eval_groups: List[TrajectoryGroup],
    ) -> Optional[Dict[str, float]]:
        """Compare evaluation results against control baseline."""
        if not self.control_reward_map:
            return None
        
        total = 0
        beats = 0
        deltas: List[float] = []
        
        for group in eval_groups:
            query_id = group.query.id
            control_rewards = self.control_reward_map.get(query_id)
            current_rewards = [sample.reward for sample in group.samples]
            if not control_rewards or not current_rewards:
                continue
            total += 1
            control_best = max(control_rewards)
            current_best = max(current_rewards)
            deltas.append(current_best - control_best)
            if current_best >= control_best:
                beats += 1
        
        if total == 0:
            return None
        
        return {
            "beat_rate": beats / total,
            "avg_delta": float(np.mean(deltas)) if deltas else 0.0,
            "median_delta": float(np.median(deltas)) if deltas else 0.0,
        }
    
    def training_step(
        self,
        queries: List[SyntheticQuery],
        log_details: bool = False,
    ) -> TrainingMetrics:
        """Execute a single training step with parallel rollouts and packed PPO loss."""
        self.model.train()
        
        if log_details:
            logger.info(f"\n{'─'*80}")
            logger.info("STEP PHASE 1: COLLECTING ROLLOUTS")
            logger.info(f"{'─'*80}")
            logger.info(f"Batch size: {len(queries)} queries")
            logger.info(f"Rollouts per query: {self.num_rollouts}")
            sys.stdout.flush()
        else:
            logger.info(f"📊 Collecting {self.num_rollouts} rollouts for {len(queries)} queries...")
        
        rollout_start = time.time()
        trajectory_groups, _ = asyncio.run(
            self.collect_rollouts_for_batch(queries, log_rollouts=log_details, step_num=self.global_step)
        )
        rollout_time = time.time() - rollout_start
        all_samples = [sample for group in trajectory_groups for sample in group.samples]
        rewards = [sample.reward for sample in all_samples]
        rubrics = [sample.rubric for sample in all_samples]
        
        if log_details:
            logger.info(f"\n{'─'*80}")
            logger.info("STEP PHASE 2: COMPUTING ADVANTAGES & FILTERING")
            logger.info(f"{'─'*80}")
        
        kept_samples, filtering_info = self.compute_advantages(trajectory_groups, log_details=log_details)
        
        if not kept_samples:
            logger.warning("⚠️  All groups filtered out! Skipping training step.")
            return TrainingMetrics(
                loss=0.0,
                policy_loss=0.0,
                kl_loss=0.0,
                avg_reward=np.mean(rewards) if rewards else 0.0,
                max_reward=np.max(rewards) if rewards else 0.0,
                min_reward=np.min(rewards) if rewards else 0.0,
                accuracy=(sum(1 for r in rubrics if r.answer_correct) / max(len(rubrics), 1)) if rubrics else 0.0,
                num_trainable_tokens=0,
                num_total_tokens=0,
                rollout_time=rollout_time,
                training_time=0.0,
                reward_std=np.std(rewards) if rewards else 0.0,
                median_reward=np.median(rewards) if rewards else 0.0,
            )
        
        if log_details:
            logger.info(f"\n{'─'*80}")
            logger.info("STEP PHASE 3: TOKENIZATION & PACKING")
            logger.info(f"{'─'*80}")
            logger.info(f"Trajectories kept: {len(kept_samples)}")
        
        tokenized_samples = self._tokenize_samples(kept_samples)
        self._populate_old_logprobs(tokenized_samples, chunk_size=max(1, self.batch_size))
        batch_tensors = self._pack_tokenized_samples(tokenized_samples)
        
        if batch_tensors is None:
            logger.warning("⚠️  No tokenized samples available after packing. Skipping step.")
            return TrainingMetrics(
                loss=0.0,
                policy_loss=0.0,
                kl_loss=0.0,
                avg_reward=np.mean(rewards) if rewards else 0.0,
                max_reward=np.max(rewards) if rewards else 0.0,
                min_reward=np.min(rewards) if rewards else 0.0,
                accuracy=(sum(1 for r in rubrics if r.answer_correct) / max(len(rubrics), 1)) if rubrics else 0.0,
                num_trainable_tokens=0,
                num_total_tokens=0,
                rollout_time=rollout_time,
                training_time=0.0,
                reward_std=np.std(rewards) if rewards else 0.0,
                median_reward=np.median(rewards) if rewards else 0.0,
            )
        
        if log_details:
            logger.info(f"\n{'─'*80}")
            logger.info("STEP PHASE 4: BACKPROPAGATION & OPTIMIZATION")
            logger.info(f"{'─'*80}")
        
        training_start = time.time()
        total_loss, loss_metrics = self._compute_grpo_loss(batch_tensors)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        training_time = time.time() - training_start
        
        # Count rollouts that exited early (didn't exhaust max_turns)
        num_early_exit = sum(1 for r in rubrics if not r.ran_out_of_turns and r.attempted_answer)
        
        if log_details:
            logger.info(f"✓ Grad norm (clipped): {grad_norm:.4f}")
            logger.info(f"✓ Tokens trained: {loss_metrics['trainable_tokens']}")
            logger.info(f"✓ Total loss: {total_loss.item():.4f}")
            logger.info(f"\n📊 Group Summary:")
            logger.info(f"  - Total groups: {len(trajectory_groups)}")
            logger.info(f"  - Groups kept for training: {filtering_info['groups_kept']}")
            logger.info(f"  - Groups filtered (low variance): {filtering_info['groups_filtered']}")
            logger.info(f"  - Rollouts that finished early (didn't exhaust turns): {num_early_exit}/{len(all_samples)}")
            logger.info(f"  - Total rollout time: {rollout_time:.1f}s")
            logger.info(f"  - Total training time: {training_time:.1f}s")
            logger.info(f"  - Total tokens: {loss_metrics['total_tokens']:,}")
            logger.info(f"  - Trainable tokens: {loss_metrics['trainable_tokens']:,} ({loss_metrics['trainable_tokens']/max(loss_metrics['total_tokens'],1)*100:.1f}%)")
            logger.info(f"{'─'*80}\n")
        
        accuracy = (sum(1 for r in rubrics if r.answer_correct) / max(len(rubrics), 1)) if rubrics else 0.0
        
        # Also output a concise summary in non-verbose mode
        if not log_details:
            logger.info(f"  Groups: {filtering_info['groups_kept']}/{len(trajectory_groups)} kept "
                       f"({filtering_info['groups_filtered']} filtered), "
                       f"{num_early_exit}/{len(all_samples)} finished early")
        
        metrics = TrainingMetrics(
            loss=total_loss.item(),
            policy_loss=loss_metrics["policy_loss"],
            kl_loss=loss_metrics["kl_loss"],
            avg_reward=np.mean(rewards) if rewards else 0.0,
            max_reward=np.max(rewards) if rewards else 0.0,
            min_reward=np.min(rewards) if rewards else 0.0,
            accuracy=accuracy,
            num_trainable_tokens=loss_metrics["trainable_tokens"],
            num_total_tokens=loss_metrics["total_tokens"],
            rollout_time=rollout_time,
            training_time=training_time,
            reward_std=np.std(rewards) if rewards else 0.0,
            median_reward=np.median(rewards) if rewards else 0.0,
            groups_kept=filtering_info["groups_kept"],
            groups_filtered=filtering_info["groups_filtered"],
            num_early_exit=num_early_exit,
        )
        
        return metrics
    
    async def evaluate(self, log_details: bool = False) -> Tuple[float, float, Optional[float]]:
        """Evaluate the model and compare against the control baseline."""
        self.model.eval()
        
        logger.info(f"📊 Evaluating on {len(self.eval_queries)} queries...")
        eval_start = time.time()
        eval_groups, _ = await self.collect_rollouts_for_batch(
            self.eval_queries,
            log_rollouts=log_details,
            step_num=self.global_step,
            num_rollouts=self.eval_rollouts,
            is_evaluation=True,
        )
        
        samples = [sample for group in eval_groups for sample in group.samples]
        rewards = [sample.reward for sample in samples]
        rubrics = [sample.rubric for sample in samples]
        correct_answers = sum(1 for r in rubrics if r.answer_correct)
        avg_reward = np.mean(rewards) if rewards else 0.0
        accuracy = correct_answers / max(len(rubrics), 1) if rubrics else 0.0
        eval_time = time.time() - eval_start
        
        # Basic statistics
        attempted = sum(1 for r in rubrics if r.attempted_answer)
        found_email = sum(1 for r in rubrics if r.ever_found_right_email)
        read_email = sum(1 for r in rubrics if r.ever_read_right_email)
        
        # Additional detailed statistics
        total_repeated_searches = sum(r.num_repeated_searches for r in rubrics)
        total_unique_searches = sum(r.num_unique_searches for r in rubrics)
        total_searches = sum(r.num_total_searches for r in rubrics)
        
        # Average turns for correct answers
        correct_rubrics = [r for r in rubrics if r.answer_correct]
        avg_turns_correct = np.mean([r.num_turns for r in correct_rubrics]) if correct_rubrics else 0.0
        
        # Average turns for "I don't know" responses
        idk_rubrics = [r for r in rubrics if r.returned_i_dont_know]
        avg_turns_idk = np.mean([r.num_turns for r in idk_rubrics]) if idk_rubrics else 0.0
        
        # Average search attempts per sample
        avg_search_attempts = total_searches / max(len(rubrics), 1)
        
        beat_stats = self._calculate_control_beat_rate(eval_groups)
        beat_rate = beat_stats["beat_rate"] if beat_stats else None
        
        # Prepare detailed evaluation statistics
        eval_stats = {
            "step": self.global_step,
            "accuracy": accuracy,
            "correct_answers": correct_answers,
            "total_samples": len(rubrics),
            "attempted_answer": attempted,
            "avg_reward": avg_reward,
            "median_reward": float(np.median(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "min_reward": float(np.min(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "found_correct_email": found_email,
            "read_correct_email": read_email,
            "total_repeated_searches": total_repeated_searches,
            "total_unique_searches": total_unique_searches,
            "total_searches": total_searches,
            "avg_turns_correct": avg_turns_correct,
            "avg_turns_idk": avg_turns_idk,
            "avg_search_attempts": avg_search_attempts,
            "num_idk": len(idk_rubrics),
            "eval_time": eval_time,
            "beat_rate": beat_rate if beat_rate is not None else 0.0,
        }
        
        # Save to file
        eval_log_dir = self.output_dir / "eval_logs"
        eval_log_dir.mkdir(parents=True, exist_ok=True)
        eval_log_file = eval_log_dir / f"eval_step_{self.global_step:04d}.json"
        with open(eval_log_file, "w") as f:
            json.dump(eval_stats, f, indent=2)
        
        if log_details:
            print(f"\n{'='*80}", flush=True)
            print(f"📊 EVALUATION RESULTS", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"🧩 Checkpoint: {self.global_step}", flush=True)
            print(f"📈 Accuracy: {accuracy*100:.2f}% ({correct_answers}/{max(len(rubrics),1)})", flush=True)
            if beat_rate is not None:
                print(f"🤝 Beat control: {beat_rate*100:.1f}% of queries", flush=True)
            print(f"\n⏱️  Time taken: {eval_time:.1f}s ({eval_time/max(len(rubrics),1):.2f}s per query)", flush=True)
            if rewards:
                print(f"💰 Average reward: {avg_reward:.3f} (std: {np.std(rewards):.3f})", flush=True)
                print(f"   Median reward: {np.median(rewards):.3f}", flush=True)
                print(f"   Range: [{np.min(rewards):.3f}, {np.max(rewards):.3f}]", flush=True)
            print(f"\n📊 Detailed Rubric Statistics:", flush=True)
            print(f"   Attempted answers: {attempted}/{len(rubrics)} ({attempted/max(len(rubrics),1)*100:.1f}%)", flush=True)
            print(f"   Found correct email: {found_email}/{len(rubrics)} ({found_email/max(len(rubrics),1)*100:.1f}%)", flush=True)
            print(f"   Read correct email: {read_email}/{len(rubrics)} ({read_email/max(len(rubrics),1)*100:.1f}%)", flush=True)
            print(f"   Repeated searches: {total_repeated_searches} (total: {total_searches}, unique: {total_unique_searches})", flush=True)
            print(f"   Avg turns (correct): {avg_turns_correct:.2f} turns", flush=True)
            print(f"   Avg turns (I don't know): {avg_turns_idk:.2f} turns (count: {len(idk_rubrics)})", flush=True)
            print(f"   Avg search attempts: {avg_search_attempts:.2f}", flush=True)
            print(f"\n💾 Eval stats saved to: {eval_log_file}", flush=True)
            print(f"{'='*80}\n", flush=True)
        
        logger.info(f"\n{'─'*80}")
        logger.info("📊 EVALUATION RESULTS")
        logger.info(f"{'─'*80}")
        logger.info(f"Checkpoint: {self.global_step}")
        logger.info(f"Accuracy: {accuracy*100:.2f}% ({correct_answers}/{max(len(rubrics),1)})")
        logger.info(f"Time taken: {eval_time:.1f}s ({eval_time/max(len(rubrics),1):.2f}s per query)")
        logger.info(f"Average reward: {avg_reward:.3f}")
        if rewards:
            logger.info(f"Median reward: {np.median(rewards):.3f}")
            logger.info(f"Reward range: [{np.min(rewards):.3f}, {np.max(rewards):.3f}]")
        if beat_rate is not None:
            logger.info(f"Beat control: {beat_rate*100:.1f}% (Δavg={beat_stats['avg_delta']:.3f})")
        logger.info("Detailed Rubric Statistics:")
        logger.info(f"  - Attempted answers: {attempted}/{len(rubrics)} ({attempted/max(len(rubrics),1)*100:.1f}%)")
        logger.info(f"  - Found correct email: {found_email}/{len(rubrics)} ({found_email/max(len(rubrics),1)*100:.1f}%)")
        logger.info(f"  - Read correct email: {read_email}/{len(rubrics)} ({read_email/max(len(rubrics),1)*100:.1f}%)")
        logger.info(f"  - Repeated searches: {total_repeated_searches} (total: {total_searches}, unique: {total_unique_searches})")
        logger.info(f"  - Avg turns (correct): {avg_turns_correct:.2f} turns")
        logger.info(f"  - Avg turns (I don't know): {avg_turns_idk:.2f} turns (count: {len(idk_rubrics)})")
        logger.info(f"  - Avg search attempts: {avg_search_attempts:.2f}")
        logger.info(f"💾 Eval stats saved to: {eval_log_file}")
        logger.info(f"{'─'*80}\n")
        
        # Log to wandb
        if self.use_wandb:
            wandb_log = {
                "eval/accuracy": accuracy,
                "eval/avg_reward": avg_reward,
                "eval/median_reward": np.median(rewards) if rewards else 0.0,
                "eval/min_reward": np.min(rewards) if rewards else 0.0,
                "eval/max_reward": np.max(rewards) if rewards else 0.0,
                "eval/std_reward": np.std(rewards) if rewards else 0.0,
                "eval/attempted_answer": attempted / max(len(rubrics), 1),
                "eval/found_email": found_email / max(len(rubrics), 1),
                "eval/read_email": read_email / max(len(rubrics), 1),
                "eval/eval_time": eval_time,
                "eval/step": self.global_step,
                "eval/total_repeated_searches": total_repeated_searches,
                "eval/total_unique_searches": total_unique_searches,
                "eval/total_searches": total_searches,
                "eval/avg_turns_correct": avg_turns_correct,
                "eval/avg_turns_idk": avg_turns_idk,
                "eval/avg_search_attempts": avg_search_attempts,
                "eval/num_idk": len(idk_rubrics),
            }
            if beat_rate is not None:
                wandb_log["eval/beat_control_rate"] = beat_rate
                if beat_stats:
                    wandb_log["eval/beat_control_avg_delta"] = beat_stats.get("avg_delta", 0.0)
            wandb.log(wandb_log, step=self.global_step)
        
        return avg_reward, accuracy, beat_rate
    
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
        
        # Prepare control baseline for future evaluations
        self._ensure_control_baseline()
        
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
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "train/loss": metrics.loss,
                    "train/policy_loss": metrics.policy_loss,
                    "train/kl_loss": metrics.kl_loss,
                    "train/avg_reward": metrics.avg_reward,
                    "train/median_reward": metrics.median_reward,
                    "train/min_reward": metrics.min_reward,
                    "train/max_reward": metrics.max_reward,
                    "train/std_reward": metrics.reward_std,
                    "train/accuracy": metrics.accuracy,
                    "train/rollout_time": metrics.rollout_time,
                    "train/training_time": metrics.training_time,
                    "train/step_time": step_time,
                    "train/trainable_tokens": metrics.num_trainable_tokens,
                    "train/total_tokens": metrics.num_total_tokens,
                    "train/trainable_token_ratio": metrics.num_trainable_tokens / max(metrics.num_total_tokens, 1),
                    "train/moving_avg_reward": np.mean(recent_rewards) if recent_rewards else 0.0,
                    "train/moving_avg_accuracy": np.mean(recent_accuracies) if recent_accuracies else 0.0,
                    "train/best_accuracy": self.best_accuracy,
                    "train/eta_seconds": eta_seconds,
                    "train/progress": step / self.max_steps,
                    "train/groups_kept": metrics.groups_kept,
                    "train/groups_filtered": metrics.groups_filtered,
                    "train/num_early_exit": metrics.num_early_exit,
                }, step=step)
            
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
                avg_reward, accuracy, beat_rate = asyncio.run(self.evaluate(log_details=self.verbose))
                
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
                    
                    # Log best model to wandb
                    if self.use_wandb:
                        wandb.log({
                            "eval/best_accuracy": self.best_accuracy,
                            "eval/improvement": accuracy - old_best_accuracy,
                        }, step=step)
                    
                    # Update the checkpoint's training_state.json with new best_accuracy
                    # This ensures resume from this checkpoint will have the correct best_accuracy
                    training_state = {
                        "global_step": self.global_step,
                        "best_accuracy": self.best_accuracy,
                        "best_model_path": str(self.best_model_path) if self.best_model_path else None,
                        "evals_without_improvement": self.evals_without_improvement,
                    }
                    training_state_path = checkpoint_path / "training_state.json"
                    with open(training_state_path, "w") as f:
                        json.dump(training_state, f, indent=2)
                    logger.info(f"✓ Checkpoint training_state.json updated with best_accuracy: {self.best_accuracy:.2%}")
                else:
                    self.evals_without_improvement += 1
                    logger.info(f"\n⚠️  No improvement for {self.evals_without_improvement} evaluation(s) "
                               f"(patience: {self.patience})")
                    logger.info(f"   Current accuracy: {accuracy*100:.2f}%")
                    logger.info(f"   Best accuracy: {self.best_accuracy*100:.2f}%")
                    logger.info(f"   Keeping best model at: {self.best_model_path}")
                    
                    # Still update the checkpoint's training_state.json with current state
                    training_state = {
                        "global_step": self.global_step,
                        "best_accuracy": self.best_accuracy,
                        "best_model_path": str(self.best_model_path) if self.best_model_path else None,
                        "evals_without_improvement": self.evals_without_improvement,
                    }
                    training_state_path = checkpoint_path / "training_state.json"
                    with open(training_state_path, "w") as f:
                        json.dump(training_state, f, indent=2)
                    logger.info(f"✓ Checkpoint training_state.json updated with evals_without_improvement: {self.evals_without_improvement}")
                
                progress_to_target = accuracy / self.target_accuracy * 100
                logger.info(f"\n📈 Progress to target: {progress_to_target:.1f}% "
                           f"({accuracy*100:.1f}% / {self.target_accuracy*100:.1f}%)")
                if beat_rate is not None:
                    logger.info(f"🤝 Beat control rate: {beat_rate*100:.1f}%")
                
                # Log progress to wandb
                if self.use_wandb:
                    wandb.log({
                        "eval/progress_to_target": progress_to_target / 100.0,
                        "eval/evals_without_improvement": self.evals_without_improvement,
                    }, step=step)
                
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
        avg_reward, accuracy, beat_rate = asyncio.run(self.evaluate(log_details=True))
        
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
        if beat_rate is not None:
            logger.info(f"Final beat-control rate: {beat_rate*100:.1f}%")
        
        if self.best_model_path:
            logger.info(f"\n💾 Best model saved at: {self.best_model_path}")
        
        # Log final summary to wandb
        if self.use_wandb:
            wandb.log({
                "final/total_steps": step,
                "final/total_time": total_training_time,
                "final/rollout_time": cumulative_rollout_time,
                "final/training_time": cumulative_training_time,
                "final/avg_step_time": total_training_time / step if step > 0 else 0,
                "final/accuracy": accuracy,
                "final/best_accuracy": self.best_accuracy,
                "final/avg_reward": avg_reward,
            })
            if beat_rate is not None:
                wandb.log({"final/beat_control_rate": beat_rate})
        
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

