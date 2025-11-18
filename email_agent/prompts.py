"""System prompts for the email agent."""

import textwrap
from email_agent.data.types import SyntheticQuery


def create_system_prompt(scenario: SyntheticQuery, max_turns: int) -> str:
    """Create system prompt for the agent.
    
    Uses standard OpenAI tool calling via LiteLLM.
    """
    return textwrap.dedent(f"""\
        You are an email search agent. You are given a user query and must use the provided tools to search the user's email and find the answer.
        
        User's email address: {scenario.inbox_address}
        Today's date: {scenario.query_date}
        Maximum turns: {max_turns}
        
        Instructions:
        1. Use the search_emails tool to find relevant emails
        2. Use the read_email tool to read the full content of specific emails
        3. When you have found the answer, use the return_final_answer tool to provide your response
        
        You may take up to {max_turns} turns to find the answer. If your first search doesn't find the answer, try with different keywords.
        
        If you cannot find the answer after searching, call return_final_answer with "I don't know" and an empty source list.
    """)


def get_tools_schema():
    """Get tools schema in OpenAI format for LiteLLM.
    
    Returns standard OpenAI tools format that LiteLLM can use directly.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "search_emails",
                "description": "Search emails in the database. Returns list of SearchResult objects with message_id and snippet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of keywords that must all appear in subject or body"
                        },
                        "from_addr": {
                            "type": "string",
                            "description": "Optional email address to filter by sender"
                        },
                        "to_addr": {
                            "type": "string",
                            "description": "Optional email address to filter by recipient"
                        },
                        "sent_after": {
                            "type": "string",
                            "description": "Optional date string 'YYYY-MM-DD' for filtering"
                        },
                        "sent_before": {
                            "type": "string",
                            "description": "Optional date string 'YYYY-MM-DD' for filtering"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (max 10)",
                            "default": 10
                        }
                    },
                    "required": ["keywords"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_email",
                "description": "Read a single email by message_id. Returns Email object with all details, or None if not found.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "Unique identifier of the email to retrieve"
                        }
                    },
                    "required": ["message_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "return_final_answer",
                "description": "Return the final answer to the user's query. Call this when you have found the answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The answer to the user's query"
                        },
                        "source_message_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of message IDs that support your answer"
                        }
                    },
                    "required": ["answer", "source_message_ids"]
                }
            }
        }
    ]

