"""Advanced GRPO training script with custom rollout function."""

import os
import torch
import json
from datasets import Dataset
from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from openai import AsyncOpenAI
import asyncio
import logging
from typing import List, Dict, Any

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


class EmailAgentRollout:
    """Custom rollout function for GRPO training."""
    
    def __init__(
        self,
        queries: List[SyntheticQuery],
        policy_config: PolicyConfig,
        openai_api_key: str,
    ):
        self.queries = queries
        self.query_dict = {q.id: q for q in queries}
        self.policy_config = policy_config
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.tools = get_tools_schema()
    
    async def rollout_single(
        self,
        messages: List[Dict[str, str]],
        scenario: SyntheticQuery,
        model_client: Any,
    ) -> tuple[List[Dict], float, Dict]:
        """Execute a single rollout.
        
        Args:
            messages: Initial conversation messages
            scenario: The scenario to evaluate
            model_client: Client for calling the model
            
        Returns:
            Tuple of (conversation_history, reward, metrics)
        """
        rubric = EvaluationRubric()
        conversation = list(messages)
        
        # Main agent loop
        for turn in range(self.policy_config.max_turns):
            rubric.num_turns += 1
            
            if self.policy_config.verbose:
                print(f"\n--- Turn {turn+1}/{self.policy_config.max_turns} ---")
            
            try:
                # Call model to get next action
                response = await model_client.chat.completions.create(
                    messages=conversation,
                    tools=self.tools,
                    tool_choice="auto",
                    max_tokens=self.policy_config.max_tokens,
                    temperature=0.7,
                )
                
                assistant_message = response.choices[0].message
                
                # Add assistant message to conversation
                conversation.append({
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        } for tc in (assistant_message.tool_calls or [])
                    ] if assistant_message.tool_calls else None,
                })
                
                # Check if there are tool calls
                if not assistant_message.tool_calls:
                    rubric.bad_tool_call_args = True
                    logger.warning("No tool call in response")
                    break
                
                # Process first tool call
                tool_call = assistant_message.tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                if self.policy_config.verbose:
                    print(f"Tool: {tool_name}")
                    print(f"Args: {tool_args}")
                
                # Execute tool
                tool_result, should_break = await execute_tool_call(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    scenario=scenario,
                    rubric=rubric,
                    openai_client=self.openai_client,
                    verbose=self.policy_config.verbose,
                )
                
                # Add tool result to conversation
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result),
                })
                
                if should_break:
                    break
                    
            except Exception as e:
                logger.error(f"Error in rollout: {e}")
                rubric.bad_tool_call_args = True
                break
        
        # Check if ran out of turns
        if rubric.num_turns >= self.policy_config.max_turns and not rubric.attempted_answer:
            rubric.ran_out_of_turns = True
        
        # Calculate reward
        reward = calculate_reward(self.policy_config, rubric)
        
        return conversation, reward, rubric.to_metrics()
    
    def __call__(
        self,
        prompts: List[str],
        trainer,
    ) -> Dict[str, Any]:
        """Rollout function called by GRPOTrainer.
        
        Args:
            prompts: List of prompts to generate completions for
            trainer: The trainer instance
            
        Returns:
            Dictionary with prompt_ids, completion_ids, and logprobs
        """
        # This is a simplified version
        # In practice, you'd need to properly integrate with the trainer
        # and return the expected format
        
        # For now, return empty results
        # The actual implementation would need to:
        # 1. Parse prompts to get query IDs
        # 2. Run rollouts for each prompt
        # 3. Collect completions and rewards
        # 4. Return in the expected format
        
        return {
            "prompt_ids": [],
            "completion_ids": [],
            "logprobs": [],
        }


def prepare_dataset(queries: List[SyntheticQuery]) -> Dataset:
    """Prepare dataset in the format expected by GRPOTrainer."""
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


def main():
    """Main training function."""
    config = GRPOConfig()
    policy_config = PolicyConfig(
        max_turns=config.max_turns,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
    )
    
    logger.info("Loading model and tokenizer...")
    
    # Load model with unsloth
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
    
    logger.info("Loading dataset...")
    
    # Load queries
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
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_queries)
    eval_dataset = prepare_dataset(eval_queries)
    
    # Create custom rollout
    rollout = EmailAgentRollout(
        queries=train_queries,
        policy_config=policy_config,
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    
    logger.info("Setting up GRPO trainer...")
    
    # Configure training
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
        optim="adamw_8bit",
        num_generation_per_prompt=config.num_generations,
        max_new_tokens=config.max_tokens,
        temperature=0.7,
    )
    
    # For now, use a simple reward function
    # The custom rollout would be integrated differently
    def simple_reward(completions, **kwargs):
        rewards = []
        for completion in completions:
            content = completion[0]["content"] if completion else ""
            reward = 0.5 if len(content) > 10 else 0.0
            rewards.append(reward)
        return rewards
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=simple_reward,
        # rollout_func=rollout,  # Would use custom rollout here
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training complete!")
    
    # Save final model
    logger.info(f"Saving final model to {config.output_dir}/final")
    model.save_pretrained(f"{config.output_dir}/final")
    tokenizer.save_pretrained(f"{config.output_dir}/final")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

