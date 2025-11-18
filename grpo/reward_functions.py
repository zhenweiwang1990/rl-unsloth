"""Reward functions for GRPO training."""

import asyncio
import json
import logging
from typing import Dict, List, Optional

import numpy as np
from openai import AsyncOpenAI

from email_agent.config import PolicyConfig
from email_agent.data import SyntheticQuery
from email_agent.rollout import calculate_reward, EvaluationRubric

logger = logging.getLogger(__name__)


async def execute_rollout(
    query: SyntheticQuery,
    model,
    tokenizer,
    policy_config: PolicyConfig,
    openai_client: Optional[AsyncOpenAI],
    verbose: bool = False,
    log_turns: bool = False,
    rollout_info: Dict = None,
):
    """Execute a real agent rollout."""
    from email_agent.agent import EmailAgent
    
    agent = EmailAgent(
        model=model,
        tokenizer=tokenizer,
        policy_config=policy_config,
        openai_client=openai_client,
    )
    
    rubric, conversation = await agent.run_query(query, verbose=verbose)
    reward = calculate_reward(policy_config, rubric)
    
    # Log compact turn-by-turn summary if requested
    if log_turns and len(conversation) > 2:
        # 1. Query question
        print(f"❓ Question: {query.question}", flush=True)
        
        # 2. Ground truth answer
        print(f"✅ Ground Truth: {query.answer}", flush=True)
        print(f"📧 Reference Email: {query.message_ids[0]}", flush=True)
        
        # 3. Turn-by-turn actions
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
                                params = f"keywords={keywords[:3]}"
                                if from_addr:
                                    params += f", from={from_addr[:20]}"
                                
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
                                print(f"  🔧 Turn {turn_num}: {func_name}({params}) → {search_result_count} results{ref_indicator}", flush=True)
                                
                            elif func_name == 'read_email':
                                msg_id = func_args.get('message_id', '')
                                params = f"msg_id={msg_id[:20]}"
                                is_reference = (msg_id == query.message_ids[0])
                                ref_indicator = " ✅ reference email" if is_reference else " ❌ not reference"
                                print(f"  🔧 Turn {turn_num}: {func_name}({params}){ref_indicator}", flush=True)
                                
                            elif func_name == 'return_final_answer':
                                answer = func_args.get('answer', '')
                                agent_final_answer = answer
                                sources = func_args.get('source_message_ids', [])
                                params = f"answer='{answer[:50]}...', sources={len(sources)}"
                                print(f"  🔧 Turn {turn_num}: {func_name}({params})", flush=True)
                            else:
                                params = str(func_args)[:50]
                                print(f"  🔧 Turn {turn_num}: {func_name}({params})", flush=True)
                        except:
                            print(f"  🔧 Turn {turn_num}: {func_name}(...)", flush=True)
                            
                elif content:
                    print(f"  💬 Turn {turn_num}: text response ({len(content)} chars)", flush=True)
        
        # 4. Agent answer and judge result
        if agent_final_answer:
            judge_result = "✅ CORRECT" if rubric.answer_correct else "❌ WRONG"
            print(f"🤖 Agent Answer: {agent_final_answer[:80]}... → {judge_result}", flush=True)
        else:
            print(f"🤖 Agent Answer: (no answer provided) → ❌ WRONG", flush=True)
        
        # 5. Trajectory metrics
        print(f"📊 Metrics: reward={reward:.2f}, turns={rubric.num_turns}, "
              f"found_email={rubric.ever_found_right_email}, "
              f"read_email={rubric.ever_read_right_email}", flush=True)
        
        # 6. Progress info (if provided)
        if rollout_info:
            current_rollout = rollout_info.get('current_rollout', 0)
            total_rollouts = rollout_info.get('total_rollouts', 0)
            elapsed = rollout_info.get('elapsed_time', 0)
            avg_time = rollout_info.get('avg_rollout_time', 0)
            remaining = (total_rollouts - current_rollout) * avg_time
            step = rollout_info.get('step', 0)
            max_steps = rollout_info.get('max_steps', 0)
            query_idx = rollout_info.get('query_idx', 0)
            total_queries = rollout_info.get('total_queries', 0)
            best_accuracy = rollout_info.get('best_accuracy', 0.0)
            lora_name = rollout_info.get('lora_name')
            
            print(f"{'─'*80}", flush=True)
            
            # Calculate total training time estimate
            # Total rollouts across all steps = max_steps * batch_size * num_rollouts
            # We can estimate this from current progress
            rollouts_per_step = total_rollouts  # This is batch_size * num_rollouts
            total_rollouts_all_steps = max_steps * rollouts_per_step
            est_total_time = avg_time * total_rollouts_all_steps
            
            # Progress line
            progress_line = f"⏱️  Progress: Step {step}/{max_steps} ({step/max_steps*100:.1f}%) | "
            progress_line += f"Rollout {current_rollout}/{total_rollouts} | "
            progress_line += f"Query {query_idx+1}/{total_queries}"
            print(progress_line, flush=True)
            
            # Time estimates
            time_line = f"⏰ Time: Batch {elapsed:.1f}s | ETA batch: {remaining:.1f}s | "
            time_line += f"Est. total training: {est_total_time/3600:.1f}h"
            print(time_line, flush=True)
            
            # LoRA and accuracy info
            if lora_name or best_accuracy > 0:
                info_parts = []
                if lora_name:
                    info_parts.append(f"LoRA: {lora_name}")
                if best_accuracy > 0:
                    info_parts.append(f"Best Accuracy: {best_accuracy*100:.2f}%")
                if info_parts:
                    print(f"📊 {' | '.join(info_parts)}", flush=True)
        
        print(f"{'─'*80}\n", flush=True)
    
    return conversation, reward, rubric


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

