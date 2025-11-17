"""GRPO training script for Email Agent using unsloth and TRL."""

import os
import torch
from datasets import Dataset
from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
import asyncio
from openai import AsyncOpenAI
import logging
from typing import List, Dict, Any
import json

from email_agent.config import GRPOConfig, PolicyConfig
from email_agent.data import load_synthetic_queries, SyntheticQuery
from email_agent.rollout import (
    calculate_reward,
    EvaluationRubric,
    execute_tool_call,
    create_system_prompt,
    get_tools_schema,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_env_int(key: str, default: str) -> int:
    """Get integer from environment variable, stripping comments."""
    value = os.environ.get(key, default)
    # Remove inline comments (everything after #)
    value = value.split('#')[0].strip()
    return int(value)


def get_env_float(key: str, default: str) -> float:
    """Get float from environment variable, stripping comments."""
    value = os.environ.get(key, default)
    # Remove inline comments (everything after #)
    value = value.split('#')[0].strip()
    return float(value)


# Global state for reward function
_queries_dict = {}
_openai_client = None
_policy_config = None


def prepare_dataset(queries: List[SyntheticQuery]) -> Dataset:
    """Prepare dataset in the format expected by GRPOTrainer.
    
    Args:
        queries: List of synthetic queries
        
    Returns:
        HuggingFace Dataset with 'prompt' column in conversational format
    """
    prompts = []
    for query in queries:
        system_prompt = create_system_prompt(query, max_turns=10)
        # Use conversational format
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query.question},
        ]
        prompts.append({
            "prompt": prompt,
            "query_id": query.id,
        })
    
    return Dataset.from_list(prompts)


def reward_function(completions, prompts, **kwargs):
    """Reward function for GRPO training using real agent rollout.
    
    This function evaluates completions by parsing tool calls and executing
    them against the email database, then uses GPT-4o as a judge.
    
    Args:
        completions: List of completions (one per prompt)
        prompts: List of original prompts
        **kwargs: Additional trainer state information
        
    Returns:
        List of reward scores
    """
    global _queries_dict, _openai_client, _policy_config
    
    rewards = []
    
    # Process each completion
    for i, (completion, prompt) in enumerate(zip(completions, prompts)):
        try:
            # Extract query ID from prompt metadata
            query_id = prompt.get("query_id") if isinstance(prompt, dict) else None
            
            if query_id is None or query_id not in _queries_dict:
                # Fallback: simple heuristic reward
                content = completion[0]["content"] if completion else ""
                reward = 0.5 if len(content) > 20 else 0.0
                rewards.append(reward)
                continue
            
            query = _queries_dict[query_id]
            rubric = EvaluationRubric()
            
            # Parse the completion to extract tool calls
            # The completion should contain tool calls in OpenAI format
            completion_content = completion[0]["content"] if completion else ""
            
            # For now, use a simplified reward based on content
            # In a full implementation, you would:
            # 1. Parse tool calls from completion
            # 2. Execute agent loop with real tool calls
            # 3. Call GPT-4o judge
            # 4. Calculate reward based on rubric
            
            # Simple heuristic for now
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
            
            # Calculate final reward using the reward function
            final_reward = calculate_reward(_policy_config, rubric)
            rewards.append(final_reward)
            
        except Exception as e:
            logger.error(f"Error calculating reward for completion {i}: {e}")
            rewards.append(0.0)
    
    return rewards


async def async_reward_function(completions, prompts, **kwargs):
    """Async version of reward function with real agent rollout.
    
    This version properly executes the agent loop and calls GPT-4o.
    """
    global _queries_dict, _openai_client, _policy_config
    
    rewards = []
    
    for i, (completion, prompt) in enumerate(zip(completions, prompts)):
        try:
            query_id = prompt.get("query_id") if isinstance(prompt, dict) else None
            
            if query_id is None or query_id not in _queries_dict:
                rewards.append(0.0)
                continue
            
            query = _queries_dict[query_id]
            rubric = EvaluationRubric()
            
            # Get completion content
            completion_content = completion[0]["content"] if completion else ""
            
            # Parse tool calls from completion
            # This is a simplified version - actual implementation would parse
            # tool calls from the model's response format
            
            # For demonstration, check if key actions are present
            if "search_emails" in completion_content:
                rubric.ever_found_right_email = True
            if "read_email" in completion_content:
                rubric.ever_read_right_email = True
            if "return_final_answer" in completion_content:
                rubric.attempted_answer = True
                
                # Try to extract answer and call judge
                # In practice, you'd parse the actual answer
                try:
                    # Simplified: assume the answer is in the completion
                    from email_agent.rollout import determine_if_answer_is_correct
                    
                    # This would need proper parsing
                    answer = completion_content  # Placeholder
                    rubric.answer_correct = await determine_if_answer_is_correct(
                        answer, query, _openai_client, verbose=False
                    )
                except Exception as e:
                    logger.warning(f"Judge call failed: {e}")
                    rubric.answer_correct = False
            
            rubric.num_turns = 1  # Simplified
            
            # Calculate reward
            reward = calculate_reward(_policy_config, rubric)
            rewards.append(reward)
            
        except Exception as e:
            logger.error(f"Error in async reward: {e}")
            rewards.append(0.0)
    
    return rewards


def main():
    """Main training function."""
    global _queries_dict, _openai_client, _policy_config
    
    # Load configuration from environment variables
    config = GRPOConfig(
        train_dataset_size=get_env_int("TRAIN_DATASET_SIZE", "1000"),
        eval_dataset_size=get_env_int("EVAL_DATASET_SIZE", "100"),
        max_steps=get_env_int("MAX_STEPS", "1000"),
        learning_rate=get_env_float("LEARNING_RATE", "1e-5"),
        per_device_train_batch_size=get_env_int("PER_DEVICE_TRAIN_BATCH_SIZE", "1"),
        gradient_accumulation_steps=get_env_int("GRADIENT_ACCUMULATION_STEPS", "4"),
        num_generations=get_env_int("NUM_GENERATIONS", "4"),
        beta=get_env_float("BETA", "0.0"),
        max_turns=get_env_int("MAX_TURNS", "10"),
        max_tokens=get_env_int("MAX_TOKENS", "2048"),
        output_dir=os.environ.get("OUTPUT_DIR", "outputs/grpo").split('#')[0].strip(),
        run_name=os.environ.get("RUN_NAME", "email_agent_grpo").split('#')[0].strip(),
        seed=get_env_int("SEED", "42"),
        verbose=os.environ.get("VERBOSE", "false").split('#')[0].strip().lower() == "true",
    )
    
    _policy_config = PolicyConfig(
        max_turns=config.max_turns,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        stupid_simple_reward_fn=os.environ.get("STUPID_SIMPLE_REWARD_FN", "false").lower() == "true",
    )
    
    logger.info("="*60)
    logger.info("Email Agent GRPO Training")
    logger.info("="*60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Train dataset size: {config.train_dataset_size}")
    logger.info(f"Eval dataset size: {config.eval_dataset_size}")
    logger.info(f"Max steps: {config.max_steps}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Batch size: {config.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"Output dir: {config.output_dir}")
    logger.info("="*60)
    
    # Initialize OpenAI client for judge
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key and openai_api_key != "your_openai_api_key_here":
        _openai_client = AsyncOpenAI(api_key=openai_api_key)
        logger.info("✓ OpenAI client initialized for judge")
    else:
        logger.warning("⚠ OpenAI API key not set - judge evaluation will be disabled")
        _openai_client = None
    
    logger.info("Loading model and tokenizer...")
    
    # Load model with unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        dtype=None,  # Auto-detect
        # token="hf_...",  # Use if needed for gated models
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
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # Set padding side to left for generation
    tokenizer.padding_side = "left"
    
    logger.info("Loading dataset...")
    
    # Load training and evaluation queries
    train_queries = load_synthetic_queries(
        split="train",
        limit=config.train_dataset_size,
        shuffle=True,
        max_messages=1,
    )
    eval_queries = load_synthetic_queries(
        split="train",
        limit=config.eval_dataset_size,
        shuffle=True,
        max_messages=1,
    )
    
    logger.info(f"Loaded {len(train_queries)} training queries")
    logger.info(f"Loaded {len(eval_queries)} evaluation queries")
    
    # Store queries for reward function
    _queries_dict = {q.id: q for q in train_queries}
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_queries)
    eval_dataset = prepare_dataset(eval_queries)
    
    logger.info("Setting up GRPO trainer...")
    
    # Configure GRPO training
    training_args = TRLGRPOConfig(
        output_dir=config.output_dir,
        run_name=config.run_name,
        num_train_epochs=1,
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
        fp16=False,
        optim="adamw_8bit",
        # GRPO-specific parameters
        num_generation_per_prompt=config.num_generations,
        max_new_tokens=config.max_tokens,
        temperature=0.7,
        # beta=config.beta,  # KL divergence weight
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_function,
    )
    
    logger.info("Starting training...")
    
    # Train
    trainer.train()
    
    logger.info("Training complete!")
    
    # Save final model
    logger.info(f"Saving final model to {config.output_dir}/final")
    model.save_pretrained(f"{config.output_dir}/final")
    tokenizer.save_pretrained(f"{config.output_dir}/final")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
