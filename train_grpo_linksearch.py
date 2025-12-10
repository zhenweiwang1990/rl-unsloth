"""GRPO training script for Link Search Agent.

This script trains a model to search LinkedIn profiles using GRPO.

Usage:
    python train_grpo_linksearch.py --mode masked
"""

import os
import sys
import argparse
import logging
from functools import partial

import torch
import wandb
from unsloth import FastLanguageModel
from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer

# Local imports
from link_search_agent.config import GRPOConfig, PolicyConfig
from link_search_agent.data import load_link_search_queries
from link_search_agent.grpo_utils import (
    execute_rollout,
    prepare_dataset,
    simple_reward_function,
    rollout_reward_function,
)

# Import shared utilities
from grpo.utils import (
    get_env_int,
    get_env_float,
    find_latest_checkpoint,
    find_best_checkpoint,
    find_auto_resume_checkpoint,
)
from grpo.callbacks import AccuracyStopCallback

# Configure logging
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

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("root").setLevel(logging.WARNING)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except:
        pass


def patch_qwen3_gradient_checkpointing(model):
    """Ensure Qwen3 decoder layers expose `_gradient_checkpointing_func`."""
    import torch.utils.checkpoint

    checkpoint_fn = torch.utils.checkpoint.checkpoint
    candidate_layer_lists = []

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        candidate_layer_lists.append(model.model.layers)
    if (
        hasattr(model, "base_model")
        and hasattr(model.base_model, "model")
        and hasattr(model.base_model.model, "layers")
    ):
        candidate_layer_lists.append(model.base_model.model.layers)

    patched = 0
    for layers in candidate_layer_lists:
        if not layers:
            continue
        for layer in layers:
            if not hasattr(layer, "_gradient_checkpointing_func"):
                layer._gradient_checkpointing_func = checkpoint_fn
                patched += 1
        if patched:
            break
    return patched


def load_model_with_unsloth(model_name: str, max_seq_length: int, load_in_4bit: bool):
    return FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="GRPO Training for Link Search Agent")
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
    parser.add_argument(
        "--enable-detailed-logging",
        action="store_true",
        help="Enable detailed rollout logging (saves JSON logs to outputs/rollout_logs/)"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = GRPOConfig(
        model_name=os.environ.get("MODEL_NAME", "unsloth/Qwen3-30B-A3B-128K").split('#')[0].strip(),
        train_dataset_size=get_env_int("TRAIN_DATASET_SIZE", "1000"),
        eval_dataset_size=get_env_int("EVAL_DATASET_SIZE", "100"),
        max_steps=get_env_int("MAX_STEPS", "200"),
        learning_rate=get_env_float("LEARNING_RATE", "1e-5"),
        per_device_train_batch_size=get_env_int("PER_DEVICE_TRAIN_BATCH_SIZE", "2"),
        num_generations=get_env_int("NUM_GENERATIONS", "3"),
        beta=get_env_float("BETA", "0.01"),
        max_turns=get_env_int("MAX_TURNS", "15"),
        max_tokens=get_env_int("MAX_TOKENS", "4096"),
        max_profiles=get_env_int("MAX_PROFILES", "10"),
        output_dir=os.environ.get("OUTPUT_DIR", f"outputs/grpo_linksearch_{args.mode}").split('#')[0].strip(),
        seed=get_env_int("SEED", "42"),
        verbose=os.environ.get("VERBOSE", "false").split('#')[0].strip().lower() == "true",
    )
    
    run_baseline_eval = os.environ.get("RUN_BASELINE_EVAL", "true").split('#')[0].strip().lower() == "true"
    
    policy_config = PolicyConfig(
        max_turns=config.max_turns,
        max_tokens=config.max_tokens,
        max_profiles=config.max_profiles,
        verbose=config.verbose,
        stupid_simple_reward_fn=False,
    )
    
    # Initialize wandb
    wandb_project = os.environ.get("WANDB_PROJECT", "link-search-grpo")
    wandb_entity = os.environ.get("WANDB_ENTITY", None)
    wandb_name = os.environ.get("WANDB_NAME", f"linksearch-grpo-{args.mode}")
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    
    if wandb_mode != "disabled":
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            mode=wandb_mode,
            config={
                "mode": args.mode,
                "model_name": config.model_name,
                "train_dataset_size": config.train_dataset_size,
                "eval_dataset_size": config.eval_dataset_size,
                "max_steps": config.max_steps,
                "learning_rate": config.learning_rate,
                "batch_size": config.per_device_train_batch_size,
                "num_generations": config.num_generations,
                "beta": config.beta,
                "max_turns": config.max_turns,
                "max_tokens": config.max_tokens,
                "max_profiles": config.max_profiles,
            },
            tags=[args.mode, "grpo", "link-search"],
        )
        print(f"✓ Wandb initialized: project={wandb_project}, name={wandb_name}", flush=True)
    else:
        print("⚠ Wandb disabled", flush=True)
    
    # Print configuration
    print("="*60, flush=True)
    print(f"GRPO Training - Link Search Agent - Mode: {args.mode.upper()}", flush=True)
    print("="*60, flush=True)
    print(f"Model: {config.model_name}", flush=True)
    print(f"Train dataset: {config.train_dataset_size}", flush=True)
    print(f"Eval dataset: {config.eval_dataset_size}", flush=True)
    print(f"Max steps: {config.max_steps}", flush=True)
    print(f"Learning rate: {config.learning_rate}", flush=True)
    print(f"Batch size: {config.per_device_train_batch_size}", flush=True)
    print(f"Max profiles: {config.max_profiles}", flush=True)
    print(f"Output dir: {config.output_dir}", flush=True)
    print("="*60, flush=True)
    
    # Determine checkpoint to resume from
    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        resume_from_checkpoint = args.resume_from_checkpoint
        print(f"\n📂 Resuming from specified checkpoint: {resume_from_checkpoint}", flush=True)
    elif args.resume_best:
        result = find_best_checkpoint(config.output_dir)
        if result:
            resume_from_checkpoint, accuracy = result
            print(f"\n📂 Resuming from best checkpoint (accuracy: {accuracy:.2%}): {resume_from_checkpoint}", flush=True)
    else:
        resume_from_checkpoint = find_auto_resume_checkpoint(config.output_dir)
        if resume_from_checkpoint:
            print(f"\n📂 Auto-resuming from checkpoint: {resume_from_checkpoint}", flush=True)
        else:
            print("\n📝 No checkpoints found, starting from scratch", flush=True)
    
    # Load model
    print("\n📦 Loading model and tokenizer...", flush=True)
    
    # For ModelScope models, we might need to set mirror
    if "modelscope" in config.model_name.lower() or config.model_name.startswith("unsloth/"):
        # Try loading from ModelScope mirror if available
        model_path = config.model_name
        print(f"Loading model: {model_path}", flush=True)
    
    if resume_from_checkpoint:
        print(f"Loading base model: {config.model_name}", flush=True)
        model, tokenizer = load_model_with_unsloth(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
        )
        
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
        
        # Load LoRA weights
        print(f"Loading LoRA weights from: {resume_from_checkpoint}", flush=True)
        from peft import set_peft_model_state_dict
        from pathlib import Path
        
        checkpoint_path = Path(resume_from_checkpoint)
        safetensors_path = checkpoint_path / "adapter_model.safetensors"
        bin_path = checkpoint_path / "adapter_model.bin"
        
        if safetensors_path.exists():
            from safetensors.torch import load_file
            adapter_weights = load_file(str(safetensors_path))
        elif bin_path.exists():
            adapter_weights = torch.load(str(bin_path), map_location="cpu")
        else:
            raise FileNotFoundError(f"No adapter weights found in {resume_from_checkpoint}")
        
        set_peft_model_state_dict(model, adapter_weights)
        print("✓ Base model and LoRA adapter loaded", flush=True)
    else:
        print(f"Loading model: {config.model_name}", flush=True)
        model, tokenizer = load_model_with_unsloth(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
        )
        
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

    patched_layers = patch_qwen3_gradient_checkpointing(model)
    if patched_layers:
        print(f"🩹 Patched {patched_layers} decoder layers for gradient checkpointing", flush=True)
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    print("\n✓ Model loaded successfully", flush=True)
    
    # Load datasets from HuggingFace
    print("\n📚 Loading datasets from HuggingFace...", flush=True)
    
    train_queries = load_link_search_queries(
        split="train",
        limit=config.train_dataset_size,
        shuffle=True,
        seed=config.seed,
    )
    eval_queries = load_link_search_queries(
        split="test",
        limit=config.eval_dataset_size,
        shuffle=True,
        seed=config.seed,
    )
    
    print(f"✓ Loaded {len(train_queries)} train queries", flush=True)
    print(f"✓ Loaded {len(eval_queries)} eval queries", flush=True)
    
    # Training
    print("\n" + "="*80, flush=True)
    print(f"🚀 Starting GRPO Training ({args.mode.upper()} mode)", flush=True)
    print("="*80, flush=True)
    
    if args.mode == "masked":
        # Import the trainer - we'll use a modified version for link search
        from link_search_agent.trainer import LinkSearchGRPOTrainer
        
        trainer = LinkSearchGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            train_queries=train_queries,
            eval_queries=eval_queries,
            policy_config=policy_config,
            num_rollouts=config.num_generations,
            learning_rate=config.learning_rate,
            beta=config.beta,
            batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=get_env_int("GRADIENT_ACCUMULATION_STEPS", "1"),
            max_grad_norm=get_env_float("MAX_GRAD_NORM", "1.0"),
            output_dir=config.output_dir,
            target_accuracy=get_env_float("TARGET_ACCURACY", "0.80"),
            eval_steps=get_env_int("EVAL_STEPS", "10"),
            save_steps=get_env_int("SAVE_STEPS", "10"),
            max_steps=config.max_steps,
            warmup_steps=get_env_int("WARMUP_STEPS", "10"),
            patience=get_env_int("PATIENCE", "5"),
            min_group_std=get_env_float("MIN_GROUP_STD", "0.05"),
            resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None,
            use_wandb=wandb_mode != "disabled",
            run_baseline_eval=run_baseline_eval,
            enable_detailed_logging=args.enable_detailed_logging,
        )
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        trainer.train()
        
        if wandb_mode != "disabled":
            wandb.finish()
    
    else:
        # Simple or rollout mode using TRL
        reward_tracker = {
            "eval_rewards": [],
            "train_rewards": [],
        }
        
        queries_dict = {q.id: q for q in train_queries + eval_queries}
        
        train_dataset = prepare_dataset(train_queries, policy_config)
        eval_dataset = prepare_dataset(eval_queries, policy_config)
        
        training_args = TRLGRPOConfig(
            output_dir=config.output_dir,
            run_name=f"link_search_grpo_{args.mode}",
            num_train_epochs=get_env_int("NUM_TRAIN_EPOCHS", "2"),
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            max_steps=config.max_steps,
            warmup_steps=config.warmup_steps,
            logging_steps=config.logging_steps,
            eval_strategy="steps",
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            save_total_limit=get_env_int("SAVE_TOTAL_LIMIT", "3"),
            seed=config.seed,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            num_generation_per_prompt=config.num_generations,
            max_new_tokens=config.max_tokens,
            temperature=get_env_float("TEMPERATURE", "0.7"),
            report_to="wandb" if wandb_mode != "disabled" else None,
        )
        
        if args.mode == "simple":
            reward_fn = partial(
                simple_reward_function,
                queries_dict=queries_dict,
                policy_config=policy_config,
                reward_tracker=reward_tracker,
                eval_dataset_size=config.eval_dataset_size,
            )
        else:
            reward_fn = partial(
                rollout_reward_function,
                model=model,
                tokenizer=tokenizer,
                queries_dict=queries_dict,
                policy_config=policy_config,
                reward_tracker=reward_tracker,
                eval_dataset_size=config.eval_dataset_size,
            )
        
        accuracy_callback = AccuracyStopCallback(
            target_accuracy=get_env_float("TARGET_ACCURACY", "0.80"),
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
        
        if resume_from_checkpoint:
            trainer.train(resume_from_checkpoint=str(resume_from_checkpoint))
        else:
            trainer.train()
        
        # Save final model
        final_model_dir = f"{config.output_dir}/final"
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        print(f"\n📁 Final model saved to: {final_model_dir}", flush=True)
        
        if wandb_mode != "disabled":
            wandb.finish()
    
    print("\n✅ Training complete!", flush=True)


if __name__ == "__main__":
    main()

