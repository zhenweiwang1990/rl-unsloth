"""GRPO training utilities for Link Search Agent."""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset

from link_search_agent.config import PolicyConfig
from link_search_agent.data.types import LinkSearchQuery
from link_search_agent.prompts import create_system_prompt
from link_search_agent.rollout import LinkSearchRubric, calculate_reward

logger = logging.getLogger(__name__)


async def execute_rollout(
    query: LinkSearchQuery,
    model,
    tokenizer,
    policy_config: PolicyConfig,
    verbose: bool = False,
    log_turns: bool = False,
    rollout_info: Dict = None,
    rollout_index: int = 0,
    num_rollouts: int = 4,
    enable_detailed_logging: bool = False,
    training_step: int = 0,
):
    """Execute a real agent rollout for link search.
    
    Returns:
        Tuple of (conversation, reward, rubric, rollout_log)
    """
    from link_search_agent.agent import LinkSearchAgent
    
    agent = LinkSearchAgent(
        model=model,
        tokenizer=tokenizer,
        policy_config=policy_config,
        rollout_index=rollout_index,
        num_rollouts=num_rollouts,
        enable_detailed_logging=enable_detailed_logging,
        training_step=training_step,
    )
    
    rubric, conversation, rollout_log = await agent.run_query(query, verbose=verbose)
    reward = calculate_reward(policy_config, rubric)
    
    # Log compact summary if requested
    if log_turns and len(conversation) > 2:
        print(f"❓ Query: {query.query}", flush=True)
        print(f"✅ Gold Handles: {query.gold_handles}", flush=True)
        
        turn_num = 0
        for i, msg in enumerate(conversation[2:], start=2):
            role = msg.get('role', '')
            
            if role == 'assistant':
                turn_num += 1
                tool_calls = msg.get('tool_calls', [])
                
                if tool_calls:
                    for tc in tool_calls:
                        func_name = tc.get('function', {}).get('name', 'unknown')
                        func_args_str = tc.get('function', {}).get('arguments', '{}')
                        
                        try:
                            func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
                            
                            if func_name == 'search_profile':
                                sql = func_args.get('sql', '')[:50]
                                print(f"  🔧 Turn {turn_num}: {func_name}(sql={sql}...)", flush=True)
                            elif func_name == 'read_profile':
                                handle = func_args.get('linkedin_handle', '')
                                print(f"  🔧 Turn {turn_num}: {func_name}({handle})", flush=True)
                            elif func_name == 'return_results':
                                results = func_args.get('results', {})
                                print(f"  🔧 Turn {turn_num}: {func_name}({len(results)} results)", flush=True)
                            else:
                                print(f"  🔧 Turn {turn_num}: {func_name}(...)", flush=True)
                        except:
                            print(f"  🔧 Turn {turn_num}: {func_name}(...)", flush=True)
        
        # Results summary
        print(f"📊 Score: {rubric.score:.3f}, Hits: {rubric.num_correct_handles}/{rubric.num_gold_handles}", flush=True)
        print(f"📊 Reward: {reward:.2f}, Turns: {rubric.num_turns}", flush=True)
        
        if rollout_info:
            current_rollout = rollout_info.get('current_rollout', 0)
            total_rollouts = rollout_info.get('total_rollouts', 0)
            step = rollout_info.get('step', 0)
            max_steps = rollout_info.get('max_steps', 0)
            print(f"⏱️ Progress: Step {step}/{max_steps}, Rollout {current_rollout}/{total_rollouts}", flush=True)
        
        print(f"{'─'*80}\n", flush=True)
    
    return conversation, reward, rubric, rollout_log


def prepare_dataset(queries: List[LinkSearchQuery], policy_config: PolicyConfig) -> Dataset:
    """Prepare dataset for TRL training."""
    prompts = []
    for query in queries:
        system_prompt = create_system_prompt(
            query, 
            policy_config.max_turns,
            policy_config.max_profiles,
        )
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Search query: {query.query}"},
        ]
        prompts.append({
            "prompt": prompt,
            "query_id": query.id,
        })
    
    return Dataset.from_list(prompts)


def simple_reward_function(
    completions,
    prompts,
    queries_dict,
    policy_config,
    reward_tracker=None,
    eval_dataset_size=100,
    **kwargs
):
    """Simple heuristic reward function for fast training."""
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
            rubric = LinkSearchRubric()
            
            completion_content = completion[0]["content"] if completion else ""
            
            # Simple heuristic
            reward = 0.0
            if len(completion_content) > 20:
                reward += 0.2
            if "search_profile" in completion_content or "SELECT" in completion_content:
                reward += 0.2
            if "read_profile" in completion_content:
                reward += 0.2
            if "return_results" in completion_content:
                reward += 0.2
            
            # Check for handles in output
            for handle in query.gold_handles:
                if handle.lower() in completion_content.lower():
                    reward += 0.1
            
            reward = min(reward, 1.0)
            rewards.append(reward)
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            rewards.append(0.0)
    
    if reward_tracker is not None:
        if len(rewards) <= eval_dataset_size:
            reward_tracker["eval_rewards"] = rewards.copy()
        else:
            reward_tracker["train_rewards"].extend(rewards)
    
    return rewards


def rollout_reward_function(
    completions,
    prompts,
    model,
    tokenizer,
    queries_dict,
    policy_config,
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
            
            conversation, reward, rubric, _ = loop.run_until_complete(
                execute_rollout(
                    query=query,
                    model=model,
                    tokenizer=tokenizer,
                    policy_config=policy_config,
                    verbose=False,
                    rollout_index=0,
                    num_rollouts=1,
                    enable_detailed_logging=False,
                    training_step=0,
                )
            )
            
            rewards.append(reward)
            
            if len(rewards) % 10 == 0:
                logger.info(f"Rollout {len(rewards)}: reward={reward:.3f}")
        
        except Exception as e:
            logger.error(f"Error in rollout: {e}")
            rewards.append(0.0)
    
    if reward_tracker is not None:
        if len(rewards) <= eval_dataset_size:
            reward_tracker["eval_rewards"] = rewards.copy()
        else:
            reward_tracker["train_rewards"].extend(rewards)
    
    logger.info(f"Batch rewards: mean={np.mean(rewards):.3f}, min={np.min(rewards):.3f}, max={np.max(rewards):.3f}")
    
    return rewards

