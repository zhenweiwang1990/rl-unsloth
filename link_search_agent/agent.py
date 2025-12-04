"""Link Search Agent for model inference and tool execution."""

import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import asdict

from link_search_agent.data.types import LinkSearchQuery
from link_search_agent.tools import search_profile, read_profile
from link_search_agent.config import PolicyConfig
from link_search_agent.prompts import create_system_prompt, get_tools_schema
from link_search_agent.rollout import (
    LinkSearchRubric,
    calculate_reward,
    compute_score,
    normalize_handle,
)
from link_search_agent.rollout_logger import RolloutLogBuilder, LinkSearchRolloutLog

logger = logging.getLogger(__name__)


class LinkSearchAgent:
    """Link Search Agent that handles model inference and tool execution.
    
    This agent:
    1. Takes a model and tokenizer
    2. Uses transformers' native tool calling support
    3. Parses tool calls from model output
    4. Executes tools (SQL search, profile reading)
    5. Tracks evaluation metrics
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        policy_config: PolicyConfig,
        rollout_index: int = 0,
        num_rollouts: int = 4,
        enable_detailed_logging: bool = False,
        training_step: int = 0,
    ):
        """Initialize the agent.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            policy_config: Policy configuration
            rollout_index: Index of this rollout within its group
            num_rollouts: Total number of rollouts in the group
            enable_detailed_logging: Whether to accumulate detailed rollout logs
            training_step: Current training step number
        """
        self.model = model
        self.tokenizer = tokenizer
        self.policy_config = policy_config
        self.tools = get_tools_schema()
        self.rollout_index = rollout_index
        self.num_rollouts = num_rollouts
        self.enable_detailed_logging = enable_detailed_logging
        self.training_step = training_step
        self.log_builder: Optional[RolloutLogBuilder] = None
    
    async def run_query(
        self,
        query: LinkSearchQuery,
        verbose: bool = False,
    ) -> Tuple[LinkSearchRubric, List[Dict[str, Any]], Optional[LinkSearchRolloutLog]]:
        """Run the agent on a single query.
        
        Args:
            query: The query to process
            verbose: Whether to print detailed logs
            
        Returns:
            Tuple of (rubric, conversation_history, rollout_log)
        """
        rubric = LinkSearchRubric()
        rubric.num_gold_handles = len(query.gold_handles)
        
        # Track state
        search_history: List[Dict] = []  # Track SQL queries
        read_history: Set[str] = set()  # Track read handles
        found_handles: Set[str] = set()  # All handles found in searches
        correct_found: Set[str] = set()  # Correct handles found
        gold_set = set(normalize_handle(h) for h in query.gold_handles)
        
        # Final results
        final_results: Dict[str, Any] = {}
        
        # Create initial conversation
        system_prompt = create_system_prompt(
            query, 
            self.policy_config.max_turns,
            self.policy_config.max_profiles,
        )
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Search query: {query.query}"},
        ]
        
        # Calculate temperature
        if self.policy_config.enable_dynamic_temperature:
            temperature = self.policy_config.base_temperature + (
                self.rollout_index * self.policy_config.temperature_increment
            )
            repetition_penalty = self.policy_config.base_repetition_penalty + (
                self.rollout_index * self.policy_config.repetition_penalty_increment
            )
        else:
            temperature = 0.7
            repetition_penalty = 1.0
        
        # Initialize log builder
        if self.enable_detailed_logging:
            self.log_builder = RolloutLogBuilder(
                query=query,
                system_prompt=system_prompt,
                max_turns=self.policy_config.max_turns,
                max_profiles=self.policy_config.max_profiles,
                policy_config={
                    "max_turns": self.policy_config.max_turns,
                    "max_tokens": self.policy_config.max_tokens,
                    "max_profiles": self.policy_config.max_profiles,
                },
                step=self.training_step,
                rollout_index=self.rollout_index,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
            for msg in conversation:
                self.log_builder.log_conversation_message(msg)
        
        if verbose:
            print("\n" + "="*80)
            print(f"QUERY {query.id}")
            print("="*80)
            print(f"Search: {query.query}")
            print(f"Gold Handles: {query.gold_handles}")
            print("="*80)
        
        # Agent loop
        for turn in range(self.policy_config.max_turns):
            rubric.num_turns += 1
            
            if verbose:
                print(f"\n{'─'*80}")
                print(f"TURN {turn + 1}/{self.policy_config.max_turns}")
                print(f"{'─'*80}")
            
            try:
                # Generate model response
                response_message, raw_content, input_tokens, output_tokens = self._generate_response(
                    conversation, verbose
                )
                
                rubric.total_input_tokens += input_tokens
                rubric.total_output_tokens += output_tokens
                
                # Add assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": response_message.get("content"),
                    "tool_calls": response_message.get("tool_calls"),
                }
                conversation.append(assistant_msg)
                
                if self.log_builder:
                    self.log_builder.log_conversation_message(assistant_msg)
                
                # Handle tool calls
                if response_message.get("tool_calls"):
                    should_break = await self._execute_tool_calls(
                        response_message["tool_calls"],
                        query,
                        rubric,
                        conversation,
                        search_history,
                        read_history,
                        found_handles,
                        correct_found,
                        gold_set,
                        final_results,
                        verbose,
                    )
                    if should_break:
                        break
                
                elif response_message.get("content"):
                    # Model returned text, try to parse as results
                    parsed_results = self._try_parse_results(response_message["content"])
                    if parsed_results:
                        final_results = parsed_results
                        rubric.attempted_answer = True
                        break
                    else:
                        rubric.cant_parse_tool_call = True
                        if verbose:
                            print(f"\n⚠️ Model returned text instead of tool call")
                        break
                else:
                    rubric.cant_parse_tool_call = True
                    if verbose:
                        print(f"\n⚠️ Model returned empty response")
                    break
                    
            except Exception as e:
                logger.error(f"Error in agent loop turn {turn + 1}: {e}")
                if verbose:
                    print(f"\n❌ Exception: {e}")
                    import traceback
                    traceback.print_exc()
                rubric.cant_parse_tool_call = True
                break
        
        # Check if ran out of turns
        if rubric.num_turns >= self.policy_config.max_turns and not rubric.attempted_answer:
            rubric.ran_out_of_turns = True
            if verbose:
                print(f"\n⏱️ Agent ran out of turns")
        
        # Compute final score
        predicted_handles = list(final_results.keys()) if final_results else []
        score_result = compute_score(predicted_handles, query.gold_handles)
        
        rubric.num_predicted_handles = len(predicted_handles)
        rubric.num_correct_handles = score_result["num_hits"]
        rubric.score = score_result["score"]
        rubric.precision = score_result["precision"]
        rubric.recall = score_result["recall"]
        
        if verbose:
            self._print_evaluation_summary(rubric, query, predicted_handles, score_result)
        
        # Build rollout log
        rollout_log = None
        if self.log_builder:
            reward = calculate_reward(self.policy_config, rubric)
            self.log_builder.log_final_results(final_results, predicted_handles)
            rollout_log = self.log_builder.build(rubric, reward)
        
        return rubric, conversation, rollout_log
    
    def _generate_response(
        self,
        conversation: List[Dict],
        verbose: bool,
    ) -> Tuple[Dict[str, Any], str, int, int]:
        """Generate a response from the model."""
        # Format with tools
        try:
            text = self.tokenizer.apply_chat_template(
                conversation,
                tools=self.tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError):
            text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_tokens = inputs.input_ids.shape[1]
        
        # Calculate temperature
        if self.policy_config.enable_dynamic_temperature:
            temperature = self.policy_config.base_temperature + (
                self.rollout_index * self.policy_config.temperature_increment
            )
            repetition_penalty = self.policy_config.base_repetition_penalty + (
                self.rollout_index * self.policy_config.repetition_penalty_increment
            )
        else:
            temperature = 0.7
            repetition_penalty = 1.0
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.policy_config.max_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        output_tokens = outputs.shape[1] - input_tokens
        
        response = self.tokenizer.decode(
            outputs[0][input_tokens:],
            skip_special_tokens=True,
        )
        
        response_message, _ = self._parse_tool_calls_from_response(response, verbose)
        
        if verbose:
            print(f"\n📝 Model Response:")
            if response_message.get("tool_calls"):
                print(f"   Tool Calls: {len(response_message['tool_calls'])}")
                for tc in response_message["tool_calls"]:
                    func_name = tc['function']['name']
                    func_args = tc['function']['arguments'][:100]
                    print(f"   - {func_name}: {func_args}")
            elif response_message.get("content"):
                content = response_message["content"]
                print(f"   {content[:300]}...")
        
        return response_message, response, input_tokens, output_tokens
    
    def _parse_tool_calls_from_response(
        self,
        response: str,
        verbose: bool,
    ) -> Tuple[Dict[str, Any], bool]:
        """Parse tool calls from model response."""
        # Try <tool_call> tags
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        tool_call_matches = re.findall(tool_call_pattern, response, re.DOTALL)
        
        if tool_call_matches:
            tool_calls = []
            for i, match in enumerate(tool_call_matches):
                try:
                    tool_data = json.loads(match.strip())
                    tool_calls.append({
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tool_data.get("name", ""),
                            "arguments": json.dumps(tool_data.get("arguments", {})),
                        }
                    })
                except json.JSONDecodeError:
                    continue
            
            if tool_calls:
                return {"content": None, "tool_calls": tool_calls}, True
        
        # Try direct JSON tool call
        try:
            json_match = re.search(r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{(?:[^{}]|{[^{}]*})*\}', response, re.DOTALL)
            
            if json_match:
                parsed = json.loads(json_match.group())
                if "name" in parsed and "arguments" in parsed:
                    tool_calls = [{
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": parsed["name"],
                            "arguments": json.dumps(parsed["arguments"]),
                        }
                    }]
                    return {"content": None, "tool_calls": tool_calls}, True
        except (json.JSONDecodeError, KeyError):
            pass
        
        return {"content": response, "tool_calls": None}, True
    
    def _try_parse_results(self, content: str) -> Optional[Dict[str, Any]]:
        """Try to parse final results from text content."""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[\s\S]*"results"[\s\S]*\}', content)
            if json_match:
                parsed = json.loads(json_match.group())
                if "results" in parsed:
                    return parsed["results"]
            
            # Try direct parse
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                if "results" in parsed:
                    return parsed["results"]
                return parsed
        except:
            pass
        return None
    
    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        query: LinkSearchQuery,
        rubric: LinkSearchRubric,
        conversation: List[Dict],
        search_history: List[Dict],
        read_history: Set[str],
        found_handles: Set[str],
        correct_found: Set[str],
        gold_set: Set[str],
        final_results: Dict[str, Any],
        verbose: bool,
    ) -> bool:
        """Execute tool calls."""
        should_break = False
        
        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id", "")
            tool_function = tool_call.get("function", {})
            tool_name = tool_function.get("name")
            
            arguments_str = tool_function.get("arguments", "{}")
            try:
                tool_args = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
            except json.JSONDecodeError as e:
                rubric.bad_tool_call_args = True
                if verbose:
                    print(f"\n❌ Failed to parse tool arguments: {e}")
                should_break = True
                break
            
            if not tool_name:
                rubric.bad_tool_call_name = True
                should_break = True
                break
            
            if verbose:
                print(f"\n🔧 Tool Call: {tool_name}")
                print(f"   Arguments: {json.dumps(tool_args, indent=2)[:200]}")
            
            # Execute tool
            tool_result, should_break_inner = await self._execute_single_tool(
                tool_name,
                tool_args,
                query,
                rubric,
                search_history,
                read_history,
                found_handles,
                correct_found,
                gold_set,
                final_results,
                verbose,
            )
            
            if verbose:
                print(f"\n📊 Tool Result:")
                result_str = json.dumps(tool_result, indent=2)
                print(f"   {result_str[:300]}...")
            
            # Add tool result to conversation
            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(tool_result),
            }
            conversation.append(tool_msg)
            
            if self.log_builder:
                self.log_builder.log_conversation_message(tool_msg)
                self.log_builder.log_tool_call(
                    turn_number=rubric.num_turns,
                    tool_name=tool_name,
                    tool_arguments=tool_args,
                    tool_result=tool_result,
                    error=tool_result.get("error") if isinstance(tool_result, dict) else None,
                )
            
            if should_break_inner:
                should_break = True
                break
        
        return should_break
    
    async def _execute_single_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        query: LinkSearchQuery,
        rubric: LinkSearchRubric,
        search_history: List[Dict],
        read_history: Set[str],
        found_handles: Set[str],
        correct_found: Set[str],
        gold_set: Set[str],
        final_results: Dict[str, Any],
        verbose: bool,
    ) -> Tuple[Any, bool]:
        """Execute a single tool call."""
        should_break = False
        
        if tool_name == "search_profile":
            rubric.num_total_searches += 1
            
            sql = tool_args.get("sql", "")
            params = tool_args.get("params")
            max_rows = tool_args.get("max_rows", 200)
            
            # Check for repeated search
            sql_normalized = sql.strip().lower()
            is_repeat = any(h["sql"].strip().lower() == sql_normalized for h in search_history)
            
            if is_repeat:
                rubric.num_repeated_searches += 1
                if verbose:
                    print(f"\n⚠️ Repeated search query!")
            else:
                rubric.num_unique_searches += 1
            
            # Execute search
            result = search_profile(sql, params, max_rows)
            
            if not result["success"]:
                rubric.sql_error_count += 1
                if verbose:
                    print(f"\n❌ SQL Error: {result.get('error')}")
            else:
                result_count = result["rowCount"]
                
                if result_count == 0:
                    rubric.num_zero_result_searches += 1
                else:
                    rubric.num_searches_with_results += 1
                
                # Track handles found
                for row in result.get("rows", []):
                    handle = row.get("handle") or row.get("linkedin_handle")
                    if handle:
                        handle_norm = normalize_handle(handle)
                        found_handles.add(handle_norm)
                        if handle_norm in gold_set:
                            if handle_norm not in correct_found:
                                correct_found.add(handle_norm)
                                if rubric.turn_first_correct_found < 0:
                                    rubric.turn_first_correct_found = rubric.num_turns
                
                # Analyze search strategy
                if len(search_history) > 0:
                    prev = search_history[-1]
                    if prev["count"] == 0 and result_count > 0:
                        rubric.broadened_search_after_zero += 1
                    elif prev["count"] >= 10 and result_count < 10:
                        rubric.narrowed_search_after_many += 1
                
                if verbose:
                    correct_in_results = [h for h in gold_set if any(
                        normalize_handle(r.get("handle") or r.get("linkedin_handle", "")) == h
                        for r in result.get("rows", [])
                    )]
                    print(f"\n✓ Search returned {result_count} results")
                    if correct_in_results:
                        print(f"   ✓ CORRECT handles found: {correct_in_results}")
            
            search_history.append({
                "sql": sql,
                "count": result.get("rowCount", 0),
                "turn": rubric.num_turns,
            })
            
            return result, should_break
        
        elif tool_name == "read_profile":
            rubric.num_total_reads += 1
            
            handle = tool_args.get("linkedin_handle", "").strip().lower()
            
            if handle in read_history:
                rubric.num_repeated_reads += 1
                if verbose:
                    print(f"\n⚠️ Repeated read of {handle}")
            else:
                rubric.num_unique_reads += 1
                read_history.add(handle)
                
                # Check if reading after good search
                if len(search_history) > 0:
                    last_search = search_history[-1]
                    if last_search["turn"] == rubric.num_turns - 1 and 0 < last_search["count"] < 20:
                        rubric.read_after_good_search += 1
            
            result = read_profile(handle)
            
            if not result["success"]:
                rubric.num_invalid_reads += 1
                if verbose:
                    print(f"\n❌ Read failed: {result.get('error')}")
            else:
                if handle in gold_set:
                    rubric.num_correct_profiles_read += 1
                    if rubric.turn_first_correct_read < 0:
                        rubric.turn_first_correct_read = rubric.num_turns
                    if verbose:
                        print(f"\n✓ Read CORRECT profile: {handle}")
            
            return result, should_break
        
        elif tool_name == "return_results":
            rubric.attempted_answer = True
            
            results = tool_args.get("results", {})
            if isinstance(results, dict):
                final_results.update(results)
            
            if verbose:
                print(f"\n🎯 Agent returning {len(results)} results")
            
            return {"status": "Results submitted", "count": len(results)}, True
        
        else:
            rubric.bad_tool_call_name = True
            if verbose:
                print(f"\n❌ Unknown tool: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}, True
    
    def _print_evaluation_summary(
        self,
        rubric: LinkSearchRubric,
        query: LinkSearchQuery,
        predicted_handles: List[str],
        score_result: Dict,
    ):
        """Print evaluation summary."""
        print(f"\n{'='*80}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Turns used: {rubric.num_turns}/{self.policy_config.max_turns}")
        print(f"Predicted handles: {predicted_handles}")
        print(f"Gold handles: {query.gold_handles}")
        print(f"\n📊 Score: {rubric.score:.3f}")
        print(f"   Hits: {score_result['num_hits']}/{score_result['num_gold']}")
        print(f"   Precision: {rubric.precision:.3f}")
        print(f"   Recall: {rubric.recall:.3f}")
        
        print(f"\n🔍 Search Metrics:")
        print(f"   Unique searches: {rubric.num_unique_searches}")
        print(f"   Repeated searches: {rubric.num_repeated_searches}")
        print(f"   Zero-result searches: {rubric.num_zero_result_searches}")
        
        print(f"\n📖 Read Metrics:")
        print(f"   Unique reads: {rubric.num_unique_reads}")
        print(f"   Correct profiles read: {rubric.num_correct_profiles_read}")
        
        reward = calculate_reward(self.policy_config, rubric)
        print(f"\n🎯 Final Reward: {reward:.3f}")
        print(f"{'='*80}\n")

