"""Profile search and reading tools using local SQLite database."""

import json
import logging
import os
import re
import sqlite3
from typing import List, Optional, Dict, Any

from link_search_agent.data.types import ProfileDetail

# Database path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.environ.get(
    "PROFILE_DB_PATH",
    os.path.join(BASE_DIR, "data", "profiles.db")
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_conn: Optional[sqlite3.Connection] = None


def get_conn() -> sqlite3.Connection:
    """Get or create database connection."""
    global _conn
    if _conn is None:
        if not os.path.exists(DEFAULT_DB_PATH):
            raise FileNotFoundError(f"Database not found: {DEFAULT_DB_PATH}")
        _conn = sqlite3.connect(
            f"file:{DEFAULT_DB_PATH}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        _conn.row_factory = sqlite3.Row
    return _conn


class SearchResult:
    """Result from profile search."""
    
    def __init__(self, handle: str, summary: Optional[str] = None, **extra):
        self.handle = handle
        self.summary = summary
        self.extra = extra
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "handle": self.handle,
            "summary": self.summary,
        }
        result.update(self.extra)
        return result


def search_profile(
    sql: str,
    params: Optional[List[Any]] = None,
    max_rows: int = 200,
) -> Dict[str, Any]:
    """Execute read-only SQL against the profiles database.
    
    This tool allows executing SELECT queries to find candidate profiles.
    
    Args:
        sql: SQL query (must be SELECT only)
        params: Optional positional parameters for the SQL
        max_rows: Maximum rows to return if SQL has no LIMIT (default 200)
        
    Returns:
        Dict with success, rows, rowCount, and optional error
    """
    try:
        # Validate SQL is SELECT only
        sql_clean = sql.strip()
        if not re.match(r'^\s*select\b', sql_clean, re.IGNORECASE):
            return {
                "success": False,
                "rows": [],
                "rowCount": 0,
                "error": "Only SELECT queries are allowed"
            }
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'\bDROP\b', r'\bDELETE\b', r'\bUPDATE\b', r'\bINSERT\b',
            r'\bALTER\b', r'\bCREATE\b', r'\bTRUNCATE\b', r'\bGRANT\b',
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, sql_clean, re.IGNORECASE):
                return {
                    "success": False,
                    "rows": [],
                    "rowCount": 0,
                    "error": f"Query contains forbidden keyword: {pattern}"
                }
        
        # Add LIMIT if not present
        if not re.search(r'\bLIMIT\b', sql_clean, re.IGNORECASE):
            sql_clean = f"{sql_clean} LIMIT {max_rows}"
        
        # Replace dev_set. prefix with empty (SQLite uses single namespace)
        sql_clean = re.sub(r'\bdev_set\.', '', sql_clean, flags=re.IGNORECASE)
        
        conn = get_conn()
        cursor = conn.cursor()
        
        if params:
            cursor.execute(sql_clean, params)
        else:
            cursor.execute(sql_clean)
        
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        result_rows = []
        for row in rows:
            row_dict = dict(row)
            # Ensure handle field is present
            if 'handle' not in row_dict and 'linkedin_handle' in row_dict:
                row_dict['handle'] = row_dict['linkedin_handle']
            result_rows.append(row_dict)
        
        return {
            "success": True,
            "rows": result_rows,
            "rowCount": len(result_rows),
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {
            "success": False,
            "rows": [],
            "rowCount": 0,
            "error": str(e),
        }


def read_profile(linkedin_handle: str) -> Dict[str, Any]:
    """Read a profile by LinkedIn handle.
    
    Args:
        linkedin_handle: The LinkedIn handle to look up
        
    Returns:
        Dict with success, profile data, and optional error
    """
    try:
        if not linkedin_handle or not linkedin_handle.strip():
            return {
                "success": False,
                "profile": None,
                "error": "LinkedIn handle is required"
            }
        
        handle = linkedin_handle.strip().lower()
        conn = get_conn()
        
        # Get profile
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                id, linkedin_handle, name, summary, about, skills, updated_at, meta
            FROM profiles
            WHERE linkedin_handle = ?
            LIMIT 1
        """, (handle,))
        
        row = cursor.fetchone()
        if not row:
            return {
                "success": False,
                "profile": None,
                "error": f"Profile not found for handle: {linkedin_handle}"
            }
        
        profile_id = row["id"]
        
        # Get experiences
        cursor.execute("""
            SELECT title, company, location, start_date, end_date, 
                   is_current, employment_type, description
            FROM experiences
            WHERE profile_id = ?
            ORDER BY start_date DESC
        """, (profile_id,))
        experiences = [dict(r) for r in cursor.fetchall()]
        
        # Get educations
        cursor.execute("""
            SELECT school, degree, field_of_study, start_date, end_date,
                   grade, activities, description
            FROM educations
            WHERE profile_id = ?
            ORDER BY start_date DESC
        """, (profile_id,))
        education = [dict(r) for r in cursor.fetchall()]
        
        # Parse meta JSON
        meta = None
        if row["meta"]:
            try:
                meta = json.loads(row["meta"])
            except:
                pass
        
        profile = {
            "linkedin_handle": row["linkedin_handle"],
            "name": row["name"],
            "summary": row["summary"],
            "about": row["about"],
            "skills": row["skills"],
            "experiences": experiences,
            "education": education,
        }
        
        return {
            "success": True,
            "profile": profile,
        }
        
    except Exception as e:
        logger.error(f"Read profile error: {e}")
        return {
            "success": False,
            "profile": None,
            "error": str(e),
        }


def get_profile_count() -> int:
    """Get total number of profiles in database."""
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM profiles")
        return cursor.fetchone()[0]
    except:
        return 0

