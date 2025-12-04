"""System prompts and tool schemas for Link Search Agent."""

import textwrap
from typing import List, Dict, Any

from link_search_agent.data.types import LinkSearchQuery


# Database schema documentation for the agent
SCHEMA_DOC = """
Database Schema (SQLite):

- profiles
  - id TEXT PRIMARY KEY (UUID)
  - name TEXT
  - about TEXT
  - summary TEXT
  - skills TEXT
  - linkedin_handle TEXT UNIQUE
  - updated_at TEXT
  - meta TEXT (JSON)

- experiences
  - id INTEGER PRIMARY KEY
  - profile_id TEXT (references profiles.id)
  - title TEXT
  - company TEXT
  - location TEXT
  - start_date TEXT
  - end_date TEXT
  - is_current INTEGER (boolean)
  - employment_type TEXT
  - description TEXT

- educations
  - id INTEGER PRIMARY KEY
  - profile_id TEXT (references profiles.id)
  - school TEXT
  - degree TEXT
  - field_of_study TEXT
  - start_date TEXT
  - end_date TEXT
  - grade TEXT
  - activities TEXT
  - description TEXT
""".strip()


def create_system_prompt(query: LinkSearchQuery, max_turns: int, max_profiles: int = 10) -> str:
    """Create system prompt for the link search agent.
    
    Args:
        query: The search query
        max_turns: Maximum number of turns allowed
        max_profiles: Target number of profiles to find
        
    Returns:
        System prompt string
    """
    return textwrap.dedent(f"""\
        You are a LinkedIn profile search agent that uses database queries (NOT browser automation) to find the most relevant candidates.

        ## Core Mission
        Find up to {max_profiles} relevant candidates based on the search query. For each promising handle, read details and decide if it matches. Track your decisions to avoid re-checking.

        IMPORTANT: You MUST actively try to reach {max_profiles} results. Do NOT stop just because you already have some good candidates. Unless you have clearly exhausted the database (e.g., several diverse search queries produce no new viable handles), you should continue searching and classifying until you have at least {max_profiles} matching candidates.

        ## Available Tools
        1. search_profile
           - Execute SELECT-only SQL queries to find candidate handles
           - Prefer selecting linkedin_handle AS handle, summary to minimize payload
           - Example: SELECT linkedin_handle AS handle, summary FROM profiles WHERE ...
        
        2. read_profile
           - Read full profile details by linkedin_handle
        
        3. return_results
           - Call this when you have found enough candidates or exhausted search options
           - Provide the final results as a JSON object

        ## Database Schema
        {SCHEMA_DOC}

        ## Workflow
        1) Search: Use search_profile with SQL to find candidate handles (always use LIMIT)
        2) Read & Classify: For promising handles, use read_profile to get details and decide if they match
        3) Track: Keep track of handles you've checked (match/no-match) to avoid re-checking
        4) Iterate: Repeat until you have {max_profiles} matches or exhausted search options

        ## Rules
        - Always include a brief reason when recording results
        - Never re-check a handle you've already processed
        - Keep SQL simple and efficient; always use LIMIT
        - If a search returns 0 results, try broader terms
        - If a search returns too many results, try narrower terms
        - Maximum {max_turns} turns allowed

        ## Search Query
        {query.query}

        ## Output Format
        When finishing, use the return_results tool with your findings as:
        {{
            "results": {{
                "handle1": {{ "name": "Full Name", "reason": "why this person matches" }},
                "handle2": {{ "name": "Full Name", "reason": "why this person matches" }},
                ...
            }}
        }}
    """)


def get_tools_schema() -> List[Dict[str, Any]]:
    """Get tools schema in OpenAI format.
    
    Returns:
        List of tool definitions in OpenAI format
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "search_profile",
                "description": (
                    "Execute read-only SQL (SELECT only) against the profiles database. "
                    "Write a SELECT statement that queries profiles, experiences, or educations tables. "
                    "To minimize payload, ONLY select handle and summary when possible. "
                    "Example: SELECT linkedin_handle AS handle, summary FROM profiles WHERE summary LIKE '%engineer%' LIMIT 20\n\n"
                    f"{SCHEMA_DOC}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "Read-only SQL query (must start with SELECT). Query the profiles, experiences, or educations tables."
                        },
                        "params": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional positional parameters for the SQL (use ? placeholders)"
                        },
                        "max_rows": {
                            "type": "integer",
                            "description": "Max rows to return if SQL has no LIMIT (default 200)",
                            "default": 200
                        }
                    },
                    "required": ["sql"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_profile",
                "description": (
                    "Read LinkedIn profile details from the database by LinkedIn handle. "
                    "Returns complete profile information including name, summary, about, skills, experiences, and education."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "linkedin_handle": {
                            "type": "string",
                            "description": "The LinkedIn handle (username) of the profile to retrieve. Example: 'john-doe'"
                        }
                    },
                    "required": ["linkedin_handle"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "return_results",
                "description": (
                    "Return the final search results. Call this when you have found enough matching candidates "
                    "or have exhausted search options. Provide results as a JSON object with handles as keys."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "object",
                            "description": "Object mapping linkedin_handle to match info. Example: {'john-doe': {'name': 'John Doe', 'reason': 'Senior engineer at Google'}}"
                        },
                        "summary": {
                            "type": "string",
                            "description": "Optional summary of the search process"
                        }
                    },
                    "required": ["results"]
                }
            }
        }
    ]

