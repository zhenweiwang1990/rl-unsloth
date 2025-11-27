"""Unified Email Agent for model inference and tool execution."""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

from email_agent.data.types import SyntheticQuery
from email_agent.tools import search_emails, read_email, SearchResult
from email_agent.config import PolicyConfig
from email_agent.prompts import create_system_prompt, get_tools_schema
from email_agent.rollout import (
    EvaluationRubric,
    determine_if_answer_is_correct,
    calculate_reward,
)
from email_agent.rollout_logger import RolloutLogBuilder, RolloutLog
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class EmailAgent:
    """Unified email agent that handles model inference and tool execution.
    
    This agent:
    1. Takes a model and tokenizer
    2. Uses transformers' native tool calling support with OpenAI-format tools
    3. Parses OpenAI-formatted tool calls from model output
    4. Executes tools and updates conversation
    5. Tracks evaluation metrics
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        policy_config: PolicyConfig,
        openai_client: Optional[AsyncOpenAI] = None,
        rollout_index: int = 0,
        num_rollouts: int = 4,
        enable_detailed_logging: bool = False,
        training_step: int = 0,
    ):
        """Initialize the agent.
        
        Args:
            model: The language model (transformers AutoModelForCausalLM)
            tokenizer: The tokenizer (transformers AutoTokenizer)
            policy_config: Policy configuration
            openai_client: OpenAI client for judge (optional, only needed for evaluation)
            rollout_index: Index of this rollout within its group (0 to num_rollouts-1)
            num_rollouts: Total number of rollouts in the group
            enable_detailed_logging: Whether to accumulate detailed rollout logs
            training_step: Current training step number
        """
        self.model = model
        self.tokenizer = tokenizer
        self.policy_config = policy_config
        self.openai_client = openai_client
        self.tools = get_tools_schema()
        self.rollout_index = rollout_index
        self.num_rollouts = num_rollouts
        self.enable_detailed_logging = enable_detailed_logging
        self.training_step = training_step
        self.log_builder: Optional[RolloutLogBuilder] = None
        
    async def run_query(
        self,
        query: SyntheticQuery,
        verbose: bool = False,
    ) -> Tuple[EvaluationRubric, List[Dict[str, Any]], Optional[RolloutLog]]:
        """Run the agent on a single query.
        
        Args:
            query: The query to process
            verbose: Whether to print detailed logs
            
        Returns:
            Tuple of (rubric, conversation_history, rollout_log)
            rollout_log is None if detailed logging is disabled
        """
        rubric = EvaluationRubric()
        
        # Track tool call history to detect repetitions
        # For searches: Key: (tool_name, normalized_args), Value: result count
        tool_call_history = {}
        
        # Track read email history to detect repeated reads
        read_history = set()  # Set of message_ids that have been read
        
        # Track search history for strategy analysis
        search_history = []  # List of {params, result_count, turn}
        
        # Create initial conversation
        system_prompt = create_system_prompt(query, self.policy_config.max_turns)
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query.question},
        ]
        
        # Calculate temperature and repetition penalty
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
        
        # Initialize log builder if detailed logging is enabled
        if self.enable_detailed_logging:
            self.log_builder = RolloutLogBuilder(
                query=query,
                system_prompt=system_prompt,
                max_turns=self.policy_config.max_turns,
                policy_config={
                    "max_turns": self.policy_config.max_turns,
                    "max_tokens": self.policy_config.max_tokens,
                    "enable_dynamic_temperature": self.policy_config.enable_dynamic_temperature,
                    "base_temperature": self.policy_config.base_temperature,
                    "temperature_increment": self.policy_config.temperature_increment,
                    "base_repetition_penalty": self.policy_config.base_repetition_penalty,
                    "repetition_penalty_increment": self.policy_config.repetition_penalty_increment,
                },
                step=self.training_step,
                rollout_index=self.rollout_index,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
            
            # Log initial conversation messages
            for msg in conversation:
                self.log_builder.log_conversation_message(msg)
        
        if verbose:
            print("\n" + "="*80)
            print(f"QUERY {query.id}")
            print("="*80)
            print(f"Question: {query.question}")
            print(f"Ground Truth Answer: {query.answer}")
            print(f"Correct Message ID: {query.message_ids[0]}")
            print(f"Inbox: {query.inbox_address}")
            print("="*80)
        
        # Agent loop
        for turn in range(self.policy_config.max_turns):
            rubric.num_turns += 1
            
            if verbose:
                print(f"\n{'─'*80}")
                print(f"TURN {turn + 1}/{self.policy_config.max_turns}")
                print(f"{'─'*80}")
            
            try:
                # Generate model response using LiteLLM
                response_message, raw_content, input_tokens, output_tokens = self._generate_response(conversation, verbose)
                
                # Track token usage
                rubric.total_input_tokens += input_tokens
                rubric.total_output_tokens += output_tokens
                
                # Add assistant message to conversation
                assistant_msg = {
                    "role": "assistant",
                    "content": response_message.get("content"),
                    "tool_calls": response_message.get("tool_calls"),
                }
                conversation.append(assistant_msg)
                
                # Log to rollout builder
                if self.log_builder:
                    self.log_builder.log_conversation_message(assistant_msg)
                
                # Check if there are tool calls from LiteLLM
                if response_message.get("tool_calls"):
                    # Execute tool calls (OpenAI format from LiteLLM)
                    should_break = await self._execute_tool_calls(
                        response_message["tool_calls"],
                        query,
                        rubric,
                        conversation,
                        tool_call_history,
                        read_history,
                        search_history,
                        verbose,
                    )
                    if should_break:
                        break
                        
                # Check if this is text response (no tool calls)
                elif response_message.get("content"):
                    # Model returned text instead of tool call
                    # This shouldn't happen with proper tool calling
                    rubric.cant_parse_tool_call = True
                    if verbose:
                        print(f"\n⚠️  Model returned text instead of tool call: {response_message['content'][:100]}")
                    break
                        
                else:
                    # Empty response
                    rubric.cant_parse_tool_call = True
                    if verbose:
                        print(f"\n⚠️  Model returned empty response")
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
                print(f"\n⏱️  Agent ran out of turns ({self.policy_config.max_turns})")
        
        if verbose:
            self._print_evaluation_summary(rubric, query)
        
        # Build final rollout log if detailed logging is enabled
        rollout_log = None
        if self.log_builder:
            from email_agent.rollout import calculate_reward
            reward = calculate_reward(self.policy_config, rubric)
            rollout_log = self.log_builder.build(rubric, reward)
        
        return rubric, conversation, rollout_log
    
    def _generate_response(
        self, 
        conversation: List[Dict], 
        verbose: bool
    ) -> Tuple[Dict[str, Any], str, int, int]:
        """Generate a response from the model with OpenAI-format tool calling.
        
        Uses transformers' native chat template with tools support.
        
        Args:
            conversation: Conversation history
            verbose: Whether to print logs
            
        Returns:
            Tuple of (response_message_dict, raw_content, input_tokens, output_tokens)
            - response_message_dict: Contains 'content' and/or 'tool_calls'
            - raw_content: Raw generated text for debugging
            - input_tokens: Number of input tokens
            - output_tokens: Number of output tokens
        """
        # Check if tokenizer supports tool calling via chat template
        try:
            # Format conversation with tools using chat template
            # Many modern tokenizers (Llama 3, Qwen, etc.) support tools parameter
            text = self.tokenizer.apply_chat_template(
                conversation,
                tools=self.tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError) as e:
            # Fallback: tokenizer doesn't support tools parameter
            logger.warning(f"Tokenizer doesn't support tools parameter: {e}")
            # Use regular chat template without tools
            text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        # Generate
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_tokens = inputs.input_ids.shape[1]
        
        # Calculate dynamic temperature and repetition penalty based on rollout index
        # This encourages exploration diversity within each group of rollouts
        if self.policy_config.enable_dynamic_temperature:
            temperature = self.policy_config.base_temperature + (
                self.rollout_index * self.policy_config.temperature_increment
            )
            repetition_penalty = self.policy_config.base_repetition_penalty + (
                self.rollout_index * self.policy_config.repetition_penalty_increment
            )
            
            if verbose:
                print(f"🎲 Rollout {self.rollout_index + 1}/{self.num_rollouts}: "
                      f"temp={temperature:.2f}, rep_penalty={repetition_penalty:.2f}")
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
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][input_tokens:],
            skip_special_tokens=True,
        )
        
        # Parse tool calls from response if present
        response_message, parsed_successfully = self._parse_tool_calls_from_response(response, verbose)
        
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
                if len(content) > 500:
                    print(f"   {content[:500]}...")
                else:
                    print(f"   {content}")
        
        return response_message, response, input_tokens, output_tokens
    
    def _parse_tool_calls_from_response(
        self,
        response: str,
        verbose: bool
    ) -> Tuple[Dict[str, Any], bool]:
        """Parse tool calls from model response.
        
        Supports multiple formats:
        1. OpenAI-style function calling XML/JSON tags
        2. Direct JSON tool call objects
        3. Plain text (no tool calls)
        
        Args:
            response: Raw model response
            verbose: Whether to print logs
            
        Returns:
            Tuple of (response_message_dict, success)
            - response_message_dict: Contains 'content' and/or 'tool_calls'
            - success: True if parsing was successful
        """
        import re
        
        # Try to parse structured tool calls
        # Look for <tool_call> tags (some models use this)
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        tool_call_matches = re.findall(tool_call_pattern, response, re.DOTALL)
        
        if tool_call_matches:
            # Parse tool calls from XML-like format
            tool_calls = []
            for i, match in enumerate(tool_call_matches):
                try:
                    # Parse JSON inside tool_call tags
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
                    if verbose:
                        print(f"\n⚠️  Failed to parse tool call: {match[:100]}")
                    continue
            
            if tool_calls:
                return {
                    "content": None,
                    "tool_calls": tool_calls,
                }, True
        
        # Try to find JSON object that looks like a tool call
        # Look for patterns like {"name": "...", "arguments": {...}}
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}', response, re.DOTALL)
            if not json_match:
                # Try more complex nested JSON
                json_match = re.search(r'\{(?:[^{}]|{[^{}]*})*\}', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                # Check if it's a tool call format
                if "name" in parsed and "arguments" in parsed:
                    tool_calls = [{
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": parsed["name"],
                            "arguments": json.dumps(parsed["arguments"]),
                        }
                    }]
                    return {
                        "content": None,
                        "tool_calls": tool_calls,
                    }, True
                    
        except (json.JSONDecodeError, KeyError):
            pass
        
        # No tool calls found, return as plain text
        return {
            "content": response,
            "tool_calls": None,
        }, True
    
    async def _handle_return_final_answer(
        self,
        tool_args: Dict[str, Any],
        query: SyntheticQuery,
        rubric: EvaluationRubric,
        verbose: bool,
    ) -> bool:
        """Handle return_final_answer tool call.
        
        Args:
            tool_args: Tool arguments with 'answer' and 'source_message_ids'
            query: The query being processed
            rubric: Evaluation rubric to update
            verbose: Whether to print logs
            
        Returns:
            True (should break the agent loop)
        """
        final_answer = tool_args.get("answer", "")
        source_message_ids = tool_args.get("source_message_ids", [])
        
        if verbose:
            print(f"\n🎯 Agent returning final answer...")
            print(f"   Answer: {final_answer}")
            print(f"   Sources: {source_message_ids}")
            print(f"   Correct source: {query.message_ids[0]}")
        
        if not isinstance(source_message_ids, list):
            source_message_ids = []
        
        rubric.num_sources = len(source_message_ids)
        
        # ========== NEW: Calculate source precision ==========
        if len(source_message_ids) > 0:
            rubric.num_correct_sources = sum(
                1 for sid in source_message_ids if sid == query.message_ids[0]
            )
            rubric.num_incorrect_sources = len(source_message_ids) - rubric.num_correct_sources
            rubric.source_precision = rubric.num_correct_sources / len(source_message_ids)
        else:
            rubric.num_correct_sources = 0
            rubric.num_incorrect_sources = 0
            rubric.source_precision = 0.0
        
        # Log final answer to rollout builder
        if self.log_builder:
            self.log_builder.log_final_answer(
                answer=final_answer,
                source_message_ids=source_message_ids,
            )
        
        if final_answer == "I don't know":
            rubric.returned_i_dont_know = True
            if verbose:
                print(f"\n   ⚠️  Agent returned: I don't know")
        else:
            rubric.attempted_answer = True
            
            # Check sources (updated logic: all sources must be correct)
            rubric.sources_correct = (rubric.source_precision == 1.0 and rubric.num_correct_sources > 0)
            
            # Call judge to check answer (if OpenAI client available)
            if self.openai_client:
                if verbose:
                    print(f"\n   Calling judge model to evaluate answer...")
                
                rubric.answer_correct = await self._judge_answer(
                    final_answer,
                    query,
                    verbose,
                )
                
                if verbose:
                    print(f"\n{'─'*60}")
                    print("ANSWER EVALUATION")
                    print(f"{'─'*60}")
                    print(f"Answer correct: {'✓ YES' if rubric.answer_correct else '✗ NO'}")
                    print(f"Sources correct: {'✓ YES' if rubric.sources_correct else '✗ NO'}")
                    print(f"{'─'*60}")
        
        return True  # Break the loop
    
    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        query: SyntheticQuery,
        rubric: EvaluationRubric,
        conversation: List[Dict],
        tool_call_history: Dict,
        read_history: set,
        search_history: List[Dict],
        verbose: bool,
    ) -> bool:
        """Execute tool calls and update conversation.
        
        Args:
            tool_calls: List of OpenAI-format tool call dictionaries
            query: The query being processed
            rubric: Evaluation rubric to update
            conversation: Conversation history to update
            tool_call_history: History of previous tool calls for repetition detection
            read_history: Set of message_ids that have been read
            search_history: List of search history for strategy analysis
            verbose: Whether to print logs
            
        Returns:
            True if should break the agent loop, False otherwise
        """
        should_break = False
        
        for tool_call in tool_calls:
            # Extract tool call info from OpenAI format
            tool_call_id = tool_call.get("id", "")
            tool_function = tool_call.get("function", {})
            tool_name = tool_function.get("name")
            
            # Parse arguments (they come as JSON string in OpenAI format)
            arguments_str = tool_function.get("arguments", "{}")
            try:
                tool_args = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
            except json.JSONDecodeError as e:
                rubric.bad_tool_call_args = True
                if verbose:
                    print(f"\n❌ Failed to parse tool arguments: {e}")
                    print(f"   Arguments string: {arguments_str}")
                should_break = True
                break
            
            if not tool_name:
                rubric.bad_tool_call_name = True
                if verbose:
                    print(f"\n❌ Missing tool_name in tool call")
                should_break = True
                break
            
            if verbose:
                print(f"\n🔧 Tool Call: {tool_name}")
                print(f"   ID: {tool_call_id}")
                print(f"   Arguments: {json.dumps(tool_args, indent=4)}")
            
            # Execute the tool
            tool_result, should_break_inner = await self._execute_single_tool(
                tool_name,
                tool_args,
                query,
                rubric,
                tool_call_history,
                read_history,
                search_history,
                verbose,
            )
            
            if verbose:
                print(f"\n📊 Tool Result:")
                if isinstance(tool_result, list):
                    print(f"   Returned {len(tool_result)} items")
                    for i, item in enumerate(tool_result[:3]):
                        print(f"   [{i+1}] {json.dumps(item, indent=6)}")
                    if len(tool_result) > 3:
                        print(f"   ... and {len(tool_result) - 3} more")
                elif isinstance(tool_result, dict):
                    if "error" in tool_result:
                        print(f"   ❌ {tool_result}")
                    else:
                        print(f"   ✓ {json.dumps(tool_result, indent=4)[:200]}...")
            
            # Add tool result to conversation in OpenAI format
            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(tool_result),
            }
            conversation.append(tool_msg)
            
            # Log to rollout builder
            if self.log_builder:
                self.log_builder.log_conversation_message(tool_msg)
            
            if should_break_inner:
                should_break = True
                break
        
        return should_break
    
    async def _execute_single_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        query: SyntheticQuery,
        rubric: EvaluationRubric,
        tool_call_history: Dict,
        read_history: set,
        search_history: List[Dict],
        verbose: bool,
    ) -> Tuple[Any, bool]:
        """Execute a single tool call.
        
        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            query: The query being processed
            rubric: Evaluation rubric to update
            tool_call_history: History of previous tool calls for repetition detection
            read_history: Set of message_ids that have been read
            search_history: List of search history for strategy analysis
            verbose: Whether to print logs
            
        Returns:
            Tuple of (tool_result, should_break)
        """
        should_break = False
        error_msg = None
        
        if tool_name == "search_emails":
            try:
                # Create a normalized key for this search to detect repetitions
                # Sort keywords for consistent comparison
                keywords = tool_args.get('keywords', [])
                if isinstance(keywords, list):
                    keywords = sorted(keywords)
                search_key = (
                    tool_name,
                    tuple(keywords) if isinstance(keywords, list) else keywords,
                    tool_args.get('from_addr', ''),
                    tool_args.get('to_addr', ''),
                )
                
                # Track total searches
                rubric.num_total_searches += 1
                
                # Check if this exact search was already performed
                is_repeat = search_key in tool_call_history
                prev_result = tool_call_history.get(search_key, None)
                
                if is_repeat:
                    rubric.num_repeated_searches += 1
                    
                    # Check if repeating a zero-result search
                    if prev_result == 0:
                        rubric.repeated_zero_result_search = True
                    
                    if verbose:
                        print(f"\n⚠️  WARNING: Repeating identical search!")
                        print(f"   This search was already performed and returned {prev_result} results")
                        print(f"   Repetition penalty will be applied")
                else:
                    # This is a unique search (different parameters)
                    rubric.num_unique_searches += 1
                
                if verbose:
                    print(f"\n🔍 Executing search_emails...")
                    print(f"   Keywords: {tool_args.get('keywords', [])}")
                    if tool_args.get('from_addr'):
                        print(f"   From: {tool_args.get('from_addr')}")
                    if tool_args.get('to_addr'):
                        print(f"   To: {tool_args.get('to_addr')}")
                
                search_results = search_emails(
                    **tool_args,
                    inbox=query.inbox_address,
                )
                result = [asdict(r) for r in search_results]
                result_count = len(search_results)
                
                # Track this search in history (update or add)
                tool_call_history[search_key] = result_count
                
                # ========== NEW: Categorize search results ==========
                if result_count == 0:
                    rubric.num_searches_with_zero_results += 1
                elif result_count >= 10:
                    rubric.num_searches_with_too_many_results += 1
                else:  # 1-9 results
                    rubric.num_searches_with_optimal_results += 1
                
                # ========== NEW: Analyze search strategy ==========
                if len(search_history) > 0:
                    prev_search = search_history[-1]
                    prev_count = prev_search['result_count']
                    prev_params = prev_search['params']
                    
                    # Scenario 1: Previous search returned 0, should broaden
                    if prev_count == 0:
                        if self._is_broader_search(tool_args, prev_params):
                            rubric.broadened_search_after_zero_results += 1
                            if verbose:
                                print(f"\n✓ Good strategy: Broadened search after 0 results")
                    
                    # Scenario 2: Previous search returned ≥10, should narrow
                    elif prev_count >= 10:
                        if self._is_narrower_search(tool_args, prev_params):
                            rubric.narrowed_search_after_many_results += 1
                            if verbose:
                                print(f"\n✓ Good strategy: Narrowed search after {prev_count} results")
                    
                    # Scenario 3: Previous search returned 1-9 (optimal), should read not search again
                    elif 1 <= prev_count <= 9:
                        rubric.ignored_optimal_results += 1
                        if verbose:
                            print(f"\n⚠️ Suboptimal: Previous search had {prev_count} results (ideal range)")
                            print(f"   Should read emails instead of searching again")
                
                # Add this search to history
                search_history.append({
                    'params': tool_args.copy(),
                    'result_count': result_count,
                    'turn': rubric.num_turns,
                })
                
                # Track zero-result searches (old logic, kept for compatibility)
                num_zero_before = rubric.num_zero_result_searches - (1 if result_count == 0 else 0)
                
                # Check if this is a retry after a previous zero-result search
                # (Good behavior: trying different parameters after getting no results)
                if not is_repeat and num_zero_before > 0:
                    # This is a unique search and we had zero-result searches before
                    # This means the agent is trying different parameters after getting no results
                    rubric.num_retry_after_zero += 1
                    if verbose:
                        print(f"\n✓ Good: Trying different search parameters after zero results "
                              f"(retry #{rubric.num_retry_after_zero})")
                
                # Check if we found the right email
                found_right = False
                for r in search_results:
                    if r.message_id == query.message_ids[0]:
                        if not rubric.ever_found_right_email:
                            rubric.ever_found_right_email = True
                            rubric.turn_found_right_email = rubric.num_turns
                        found_right = True
                
                if verbose:
                    print(f"\n✓ Search returned {len(search_results)} email(s)")
                    if found_right:
                        print(f"   ✓ CORRECT email found in results!")
                    else:
                        print(f"   ✗ Correct email NOT in results")
                
                # Log to rollout builder
                if self.log_builder:
                    self.log_builder.log_tool_call(
                        turn_number=rubric.num_turns,
                        tool_name=tool_name,
                        tool_arguments=tool_args,
                        tool_result=result,
                        correct_message_id=query.message_ids[0],
                        error=None,
                    )
                
                return result, should_break
                
            except Exception as e:
                rubric.bad_tool_call_args = True
                logger.error(f"Error searching emails: {e}")
                error_msg = str(e)
                should_break = True
                
                # Log error to rollout builder
                if self.log_builder:
                    self.log_builder.log_tool_call(
                        turn_number=rubric.num_turns,
                        tool_name=tool_name,
                        tool_arguments=tool_args,
                        tool_result={"error": error_msg},
                        correct_message_id=query.message_ids[0],
                        error=error_msg,
                    )
                
                return {"error": error_msg}, should_break
                
        elif tool_name == "read_email":
            message_id_to_read = tool_args.get("message_id")
            
            if not isinstance(message_id_to_read, str):
                rubric.bad_tool_call_args = True
                should_break = True
                if verbose:
                    print(f"\n❌ Invalid message_id type: {type(message_id_to_read)}")
                return {"error": "Invalid message_id type"}, should_break
            
            # ========== NEW: Track read repetitions ==========
            rubric.num_total_reads += 1
            
            is_repeat_read = message_id_to_read in read_history
            if is_repeat_read:
                rubric.num_repeated_reads += 1
                
                # Extra tracking if re-reading the correct email
                if message_id_to_read == query.message_ids[0]:
                    rubric.repeated_correct_email += 1
                    if verbose:
                        print(f"\n⚠️ WARNING: Re-reading the CORRECT email!")
                        print(f"   This email was already read. Significant penalty will apply.")
                
                if verbose:
                    print(f"\n⚠️ WARNING: Re-reading email {message_id_to_read[:20]}...")
                    print(f"   This email was already read (repetition #{rubric.num_repeated_reads})")
                    print(f"   Repetition penalty will apply.")
            else:
                rubric.num_unique_reads += 1
                read_history.add(message_id_to_read)
            
            # ========== Check if reading after optimal search ==========
            if len(search_history) > 0:
                last_search = search_history[-1]
                # Check if the last action was a search with optimal results
                if last_search['turn'] == rubric.num_turns - 1:  # Last turn was search
                    if 1 <= last_search['result_count'] <= 9:
                        rubric.read_after_optimal_search += 1
                        if verbose:
                            print(f"\n✓ Good strategy: Reading email after optimal search "
                                  f"({last_search['result_count']} results)")
            
            if verbose:
                print(f"\n📧 Reading email: {message_id_to_read}")
            
            is_correct = message_id_to_read == query.message_ids[0]
            if is_correct:
                if not rubric.ever_read_right_email:
                    rubric.ever_read_right_email = True
                    rubric.turn_read_right_email = rubric.num_turns
            
            email_content = read_email(message_id_to_read)
            
            if email_content is None:
                rubric.ever_tried_to_read_invalid_email = True
                if verbose:
                    print(f"   ❌ Email not found!")
                
                error_result = {"error": "Email not found"}
                
                # Log to rollout builder
                if self.log_builder:
                    self.log_builder.log_tool_call(
                        turn_number=rubric.num_turns,
                        tool_name=tool_name,
                        tool_arguments=tool_args,
                        tool_result=error_result,
                        correct_message_id=query.message_ids[0],
                        error="Email not found",
                    )
                
                return error_result, should_break
            else:
                if verbose:
                    print(f"   ✓ Successfully read email")
                    print(f"   Subject: {email_content.subject[:80]}...")
                    print(f"   From: {email_content.from_address}")
                    print(f"   Date: {email_content.date}")
                    if is_correct:
                        print(f"   ✓ This is the CORRECT email!")
                    else:
                        print(f"   ✗ This is NOT the correct email (correct: {query.message_ids[0]})")
                
                result = email_content.model_dump()
                
                # Log to rollout builder
                if self.log_builder:
                    self.log_builder.log_tool_call(
                        turn_number=rubric.num_turns,
                        tool_name=tool_name,
                        tool_arguments=tool_args,
                        tool_result=result,
                        correct_message_id=query.message_ids[0],
                        error=None,
                    )
                
                return result, should_break
                
        elif tool_name == "return_final_answer":
            # Handle final answer tool call
            should_break = await self._handle_return_final_answer(
                tool_args,
                query,
                rubric,
                verbose,
            )
            
            result = {"status": "Final answer submitted"}
            
            # Log to rollout logger
            if self.log_builder:
                self.log_builder.log_tool_call(
                    turn_number=rubric.num_turns,
                    tool_name=tool_name,
                    tool_arguments=tool_args,
                    tool_result=result,
                    correct_message_id=query.message_ids[0],
                    error=None,
                )
            
            return result, should_break
            
        else:
            rubric.bad_tool_call_name = True
            should_break = True
            logger.error(f"Unknown tool name: {tool_name}")
            error_msg = f"Unknown tool: {tool_name}"
            error_result = {"error": error_msg}
            
            # Log error to rollout logger
            if self.log_builder:
                self.log_builder.log_tool_call(
                    turn_number=rubric.num_turns,
                    tool_name=tool_name,
                    tool_arguments=tool_args,
                    tool_result=error_result,
                    correct_message_id=query.message_ids[0],
                    error=error_msg,
                )
            
            return error_result, should_break
    
    async def _judge_answer(
        self,
        answer: str,
        query: SyntheticQuery,
        verbose: bool,
    ) -> bool:
        """Use GPT-4o to judge if the answer is correct.
        
        Args:
            answer: The answer provided by the agent
            query: The query with ground truth
            verbose: Whether to print logs
            
        Returns:
            True if answer is correct, False otherwise
        """
        if not self.openai_client:
            logger.warning("OpenAI client not available for judging")
            return False
        
        # Build judge prompts (copied from rollout.py for logging)
        system_prompt = (
            "You will be given a question and two different answers to the question: "
            "the correct answer and the answer given by an AI. Your job is to determine "
            "if the answer given by the AI is correct. Return True if the answer is "
            "semantically similar to the correct answer, and False otherwise. "
            "Return only the word True or False, no other text."
        )
        
        user_prompt = (
            f"Question: {query.question}\n"
            f"Correct answer: {query.answer}\n"
            f"AI answer: {answer}"
        )
        
        # Call the judge function
        is_correct = await determine_if_answer_is_correct(
            answer=answer,
            query=query,
            openai_client=self.openai_client,
            verbose=verbose,
        )
        
        # Log judge evaluation if logger is available
        # Note: We don't have the actual judge response here, so we'll just log the prompts
        # and the result. The actual response is logged in determine_if_answer_is_correct
        if self.log_builder:
            self.log_builder.log_judge_evaluation(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                agent_answer=answer,
                ground_truth_answer=query.answer,
                judge_response="True" if is_correct else "False",  # Simplified
                is_correct=is_correct,
            )
        
        return is_correct
    
    def _is_broader_search(self, current_params: Dict, prev_params: Dict) -> bool:
        """Check if current search is broader than previous search.
        
        A search is considered broader if:
        - Fewer keywords
        - Removed filters (from_addr, to_addr, etc.)
        """
        curr_keywords = set(current_params.get('keywords', []))
        prev_keywords = set(prev_params.get('keywords', []))
        
        # Fewer keywords = broader
        if len(curr_keywords) < len(prev_keywords):
            return True
        
        # Removed filters = broader
        filters_removed = (
            (prev_params.get('from_addr') and not current_params.get('from_addr')) or
            (prev_params.get('to_addr') and not current_params.get('to_addr'))
        )
        
        return filters_removed
    
    def _is_narrower_search(self, current_params: Dict, prev_params: Dict) -> bool:
        """Check if current search is narrower than previous search.
        
        A search is considered narrower if:
        - More keywords
        - Added filters (from_addr, to_addr, etc.)
        """
        curr_keywords = set(current_params.get('keywords', []))
        prev_keywords = set(prev_params.get('keywords', []))
        
        # More keywords = narrower
        if len(curr_keywords) > len(prev_keywords):
            return True
        
        # Added filters = narrower
        filters_added = (
            (current_params.get('from_addr') and not prev_params.get('from_addr')) or
            (current_params.get('to_addr') and not prev_params.get('to_addr'))
        )
        
        return filters_added
    
    def _print_evaluation_summary(self, rubric: EvaluationRubric, query: SyntheticQuery):
        """Print evaluation summary."""
        print(f"\n{'='*80}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Turns used: {rubric.num_turns}/{self.policy_config.max_turns}")
        print(f"Attempted answer: {'✓' if rubric.attempted_answer else '✗'}")
        print(f"Found right email: {'✓' if rubric.ever_found_right_email else '✗'}")
        print(f"Read right email: {'✓' if rubric.ever_read_right_email else '✗'}")
        print(f"Answer correct: {'✓' if rubric.answer_correct else '✗'}")
        print(f"Sources correct: {'✓' if rubric.sources_correct else '✗'}")
        
        # Show search effort metrics
        print(f"\n📊 Search Metrics:")
        print(f"   Unique searches: {rubric.num_unique_searches}")
        print(f"   Total searches: {rubric.num_total_searches}")
        if rubric.num_retry_after_zero > 0:
            print(f"   ✓ Retries after zero results: {rubric.num_retry_after_zero} (good behavior)")
        
        # Show efficiency issues
        if rubric.num_repeated_searches > 0 or rubric.num_zero_result_searches > 0 or rubric.gave_up_too_early:
            print(f"\n⚠️  Efficiency Issues:")
            if rubric.num_repeated_searches > 0:
                print(f"   Repeated identical searches: {rubric.num_repeated_searches}")
            if rubric.repeated_zero_result_search:
                print(f"   Repeated zero-result search: ✗ (extra penalty)")
            if rubric.num_zero_result_searches > 0:
                print(f"   Total zero-result searches: {rubric.num_zero_result_searches}")
            if rubric.gave_up_too_early:
                if rubric.ran_out_of_turns:
                    print(f"   ⚠️  Ran out of turns: {rubric.num_unique_searches} unique searches")
                else:
                    print(f"   ✗ Gave up EARLY (before turn budget exhausted): "
                          f"only {rubric.num_unique_searches} unique searches "
                          f"(expected at least 3) - SEVERE PENALTY")
        
        # Calculate reward
        reward = calculate_reward(self.policy_config, rubric)
        
        # Check if this is a perfect case
        is_perfect = (
            rubric.num_turns == 3 and
            rubric.ever_found_right_email and
            rubric.ever_read_right_email and
            rubric.answer_correct and
            rubric.sources_correct and
            rubric.num_repeated_searches == 0 and
            rubric.num_zero_result_searches == 0
        )
        
        if is_perfect:
            print(f"\n🌟 PERFECT EXECUTION: 3 turns (search→read→answer)")
            print(f"🎯 Final Reward: {reward:.3f} (FULL MARKS)")
        else:
            print(f"\n🎯 Final Reward: {reward:.3f}")
        print(f"{'='*80}\n")

