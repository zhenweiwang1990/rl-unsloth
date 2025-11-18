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
import argparse
import logging
from functools import partial

import torch
from unsloth import FastLanguageModel
from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer
from openai import AsyncOpenAI

# Local imports
from email_agent.config import GRPOConfig, PolicyConfig
from email_agent.data import load_synthetic_queries

# GRPO module imports
from grpo import (
    AgentGRPOTrainer,
    AccuracyStopCallback,
    simple_reward_function,
    rollout_reward_function,
    get_env_int,
    get_env_float,
    prepare_dataset,
    find_latest_checkpoint,
    find_best_checkpoint,
)

# Configure logging with immediate flushing
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setLevel(logging.INFO)
log_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
    handlers=[log_handler]
)
logger = logging.getLogger(__name__)

# Silence httpx and root logger's INFO messages
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("root").setLevel(logging.WARNING)

# Force unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except:
        pass


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint to resume from"
    )
    parser.add_argument(
        "--resume_best",
        action="store_true",
        help="Resume from best checkpoint (highest accuracy) instead of latest"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = GRPOConfig(
        model_name=os.environ.get("MODEL_NAME", "unsloth/Qwen3-14B-Base").split('#')[0].strip(),
        train_dataset_size=get_env_int("TRAIN_DATASET_SIZE", "3000"),
        eval_dataset_size=get_env_int("EVAL_DATASET_SIZE", "100"),  # 100 validation questions
        max_steps=get_env_int("MAX_STEPS", "200"),
        learning_rate=get_env_float("LEARNING_RATE", "1e-5"),
        per_device_train_batch_size=get_env_int("PER_DEVICE_TRAIN_BATCH_SIZE", "4"),  # 8 questions per batch
        num_generations=get_env_int("NUM_GENERATIONS", "3"),  # 6 rollouts per question
        beta=get_env_float("BETA", "0.01"),
        max_turns=get_env_int("MAX_TURNS", "10"),
        max_tokens=get_env_int("MAX_TOKENS", "4096"),
        output_dir=os.environ.get("OUTPUT_DIR", f"outputs/grpo_{args.mode}").split('#')[0].strip(),
        seed=get_env_int("SEED", "42"),
        verbose=os.environ.get("VERBOSE", "false").split('#')[0].strip().lower() == "true",
    )
    
    policy_config = PolicyConfig(
        max_turns=config.max_turns,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        stupid_simple_reward_fn=False,  # Use complex reward with partial credit and penalties
    )
    
    # Force print to ensure output is visible
    print("="*60, flush=True)
    print(f"GRPO Training - Mode: {args.mode.upper()}", flush=True)
    print("="*60, flush=True)
    print(f"Model: {config.model_name}", flush=True)
    print(f"Train dataset: {config.train_dataset_size}", flush=True)
    print(f"Eval dataset: {config.eval_dataset_size}", flush=True)
    print(f"Max steps: {config.max_steps}", flush=True)
    print(f"Learning rate: {config.learning_rate}", flush=True)
    print(f"Batch size: {config.per_device_train_batch_size}", flush=True)
    print(f"Output dir: {config.output_dir}", flush=True)
    
    if args.mode == "simple":
        print("⚡ Fast training with heuristic rewards", flush=True)
    elif args.mode == "rollout":
        print("🔄 Training with real agent rollouts", flush=True)
    elif args.mode == "masked":
        print("✅ Full implementation with token-level masking (RECOMMENDED)", flush=True)
    
    print("="*60, flush=True)
    
    # Also log (for file logging if configured)
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
    
    # Determine checkpoint to resume from
    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        # Explicit checkpoint path provided
        resume_from_checkpoint = args.resume_from_checkpoint
        print(f"\n📂 Resuming from specified checkpoint: {resume_from_checkpoint}", flush=True)
        logger.info(f"Resuming from specified checkpoint: {resume_from_checkpoint}")
    elif args.resume or args.resume_best:
        # Auto-detect checkpoint
        if args.resume_best:
            result = find_best_checkpoint(config.output_dir)
            if result:
                resume_from_checkpoint, accuracy = result
                print(f"\n📂 Resuming from best checkpoint (accuracy: {accuracy:.2%}): {resume_from_checkpoint}", flush=True)
                logger.info(f"Resuming from best checkpoint (accuracy: {accuracy:.2%}): {resume_from_checkpoint}")
            else:
                print("\n⚠️  No checkpoints found, starting from scratch", flush=True)
                logger.warning("No checkpoints found, starting from scratch")
        else:
            resume_from_checkpoint = find_latest_checkpoint(config.output_dir)
            if resume_from_checkpoint:
                print(f"\n📂 Resuming from latest checkpoint: {resume_from_checkpoint}", flush=True)
                logger.info(f"Resuming from latest checkpoint: {resume_from_checkpoint}")
            else:
                print("\n⚠️  No checkpoints found, starting from scratch", flush=True)
                logger.warning("No checkpoints found, starting from scratch")
    
    # Initialize OpenRouter client (for judge)
    print("\n🔑 Checking OpenRouter API key...", flush=True)
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_api_key and openrouter_api_key != "your_openrouter_api_key_here":
        openai_client = AsyncOpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        print("✓ OpenRouter client initialized (for judge)", flush=True)
        logger.info("✓ OpenRouter client initialized (for judge)")
    else:
        print("⚠ OpenRouter API key not set - using heuristic evaluation", flush=True)
        print("⚠ Set OPENROUTER_API_KEY environment variable to enable judge-based evaluation", flush=True)
        logger.warning("⚠ OpenRouter API key not set - using heuristic evaluation")
        logger.warning("⚠ Set OPENROUTER_API_KEY environment variable to enable judge-based evaluation")
        openai_client = None
    
    # Load model
    print("\n📦 Loading model and tokenizer...", flush=True)
    logger.info("Loading model and tokenizer...")
    
    if resume_from_checkpoint:
        # Load from checkpoint
        print(f"Loading from checkpoint: {resume_from_checkpoint}", flush=True)
        logger.info(f"Loading from checkpoint: {resume_from_checkpoint}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(resume_from_checkpoint),
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            dtype=None,
        )
        print("✓ Model and LoRA weights loaded from checkpoint", flush=True)
        logger.info("✓ Model and LoRA weights loaded from checkpoint")
    else:
        # Load base model and apply LoRA
        print(f"Model: {config.model_name}", flush=True)
        logger.info(f"Loading base model: {config.model_name}")
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
    
    print("\n✓ Model loaded successfully", flush=True)
    logger.info("✓ Model loaded")
    
    # Load datasets
    print("\n📚 Loading datasets...", flush=True)
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
    
    print(f"✓ Loaded {len(train_queries)} train queries", flush=True)
    print(f"✓ Loaded {len(eval_queries)} eval queries", flush=True)
    logger.info(f"✓ Loaded {len(train_queries)} train queries")
    logger.info(f"✓ Loaded {len(eval_queries)} eval queries")
    
    # Training based on mode
    print("\n" + "="*80, flush=True)
    print(f"🚀 Starting GRPO Training ({args.mode.upper()} mode)", flush=True)
    print("="*80, flush=True)
    
    if args.mode == "masked":
        # Full implementation with token masking
        print("\n📋 Initializing AgentGRPOTrainer...", flush=True)
        trainer = AgentGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            train_queries=train_queries,
            eval_queries=eval_queries,
            policy_config=policy_config,
            openai_client=openai_client,
            num_rollouts=config.num_generations,  # 6 rollouts per question
            learning_rate=config.learning_rate,
            beta=config.beta,
            batch_size=config.per_device_train_batch_size,  # 8 questions per batch = 48 trajectories total
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            output_dir=config.output_dir,
            target_accuracy=0.95,
            eval_steps=2,  # Evaluate every 30 steps
            save_steps=4,
            max_steps=config.max_steps,
            warmup_steps=10,
            patience=get_env_int("PATIENCE", "5"),  # Early stopping patience
            min_group_std=get_env_float("MIN_GROUP_STD", "0.05"),  # Minimum std to keep a group
            resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None,
        )
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
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
        if resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=str(resume_from_checkpoint))
        else:
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
