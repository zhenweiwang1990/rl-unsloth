"""System prompts for the email agent."""

import textwrap
from email_agent.data.types import SyntheticQuery


def create_system_prompt(scenario: SyntheticQuery, max_turns: int) -> str:
    """Create system prompt for the agent.
    
    The prompt instructs the model to output JSON format for tool calls.
    """
    return textwrap.dedent(f"""\
        You are an email search agent. You are given a user query and must use the provided tools to search the user's email and find the answer.
        You have {max_turns} turns to find the answer,just retry different keywords when searching.
        
        User's email address: {scenario.inbox_address}
        Today's date: {scenario.query_date}
        Maximum turns: {max_turns}
        
        IMPORTANT: You must respond with valid JSON in one of these two formats:
        
        1. To call tools, output:
        {{
          "tool_calls": [
            {{
              "tool_name": "search_emails",
              "arguments": {{
                "keywords": ["keyword1", "keyword2"],
                "max_results": 10
              }}
            }}
          ]
        }}
        
        2. To return the final answer, output:
        {{
          "final_answer": "Your answer here",
          "source_message_ids": ["<message_id_1>", "<message_id_2>"]
        }}
        
        If you cannot find the answer, return:
        {{
          "final_answer": "I don't know",
          "source_message_ids": []
        }}
        
        Available tools:
        
        1. search_emails - Search emails in the database
           Parameters:
           - keywords: List[str] (required) - Keywords that must all appear in subject or body
           - from_addr: str (optional) - Filter by sender email address
           - to_addr: str (optional) - Filter by recipient email address
           - sent_after: str (optional) - Date string 'YYYY-MM-DD' for filtering
           - sent_before: str (optional) - Date string 'YYYY-MM-DD' for filtering
           - max_results: int (optional, default=10, max=10) - Maximum number of results
           
           Returns: List of objects with message_id and snippet
        
        2. read_email - Read a single email by message_id
           Parameters:
           - message_id: str (required) - Unique identifier of the email
           
           Returns: Email object with all details (from, to, cc, subject, body, date), or error if not found
        
        3. return_final_answer - Return the final answer to the user's query
           Parameters:
           - answer: str (required) - The answer to the user's query
           - source_message_ids: List[str] (required) - List of message IDs that support the answer
        
        You may take up to {max_turns} turns to find the answer. If your first search doesn't find the answer, try with different keywords.
        
        Remember: Always output valid JSON. Do not include any other text outside the JSON.
    """)


def get_tools_schema():
    """Get tools schema for reference.
    
    This is mainly for compatibility with existing code that uses OpenAI-style schema.
    The actual prompt uses a simpler format embedded in the system prompt.
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
                            "description": "List of message IDs that are relevant (usually just one)"
                        }
                    },
                    "required": ["answer", "source_message_ids"]
                }
            }
        }
    ]

