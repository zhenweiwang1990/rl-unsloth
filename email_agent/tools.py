"""Email search and reading tools for the agent."""

import sqlite3
import logging
from typing import List, Optional
from dataclasses import dataclass
import os

from email_agent.data.types import Email

# Database path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.environ.get(
    "EMAIL_DB_PATH", 
    os.path.join(BASE_DIR, "..", "enron_emails.db")
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

conn = None


def get_conn():
    """Get or create database connection."""
    global conn
    if conn is None:
        conn = sqlite3.connect(
            f"file:{DEFAULT_DB_PATH}?mode=ro", uri=True, check_same_thread=False
        )
    return conn


@dataclass
class SearchResult:
    """Result from email search."""
    
    message_id: str
    snippet: str


def search_emails(
    inbox: str,
    keywords: List[str],
    from_addr: Optional[str] = None,
    to_addr: Optional[str] = None,
    sent_after: Optional[str] = None,
    sent_before: Optional[str] = None,
    max_results: int = 10,
) -> List[SearchResult]:
    """Search emails in the database.
    
    Args:
        inbox: Email address of the user performing the search
        keywords: List of keywords that must all appear in subject or body
        from_addr: Optional email address to filter by sender
        to_addr: Optional email address to filter by recipient
        sent_after: Optional date string 'YYYY-MM-DD' for filtering
        sent_before: Optional date string 'YYYY-MM-DD' for filtering
        max_results: Maximum number of results (max 10)
        
    Returns:
        List of SearchResult objects with message_id and snippet
    """
    params: List[str | int] = []
    cursor = get_conn().cursor()

    where_clauses: List[str] = []

    if not keywords:
        raise ValueError("No keywords provided for search.")

    if max_results > 10:
        raise ValueError("max_results must be less than or equal to 10.")

    # FTS5 query - escape double quotes by doubling them
    fts_query = " ".join(f'"{k.replace(chr(34), chr(34) * 2)}"' for k in keywords)
    where_clauses.append("fts.emails_fts MATCH ?")
    params.append(fts_query)

    # Inbox filter
    where_clauses.append(
        """
        (e.from_address = ? OR EXISTS (
            SELECT 1 FROM recipients r_inbox
            WHERE r_inbox.recipient_address = ? AND r_inbox.email_id = e.id
        ))
        """
    )
    params.extend([inbox, inbox])

    # Optional from filter
    if from_addr:
        where_clauses.append("e.from_address = ?")
        params.append(from_addr)

    # Optional to filter
    if to_addr:
        where_clauses.append(
            """
            EXISTS (
                SELECT 1 FROM recipients r_to
                WHERE r_to.recipient_address = ? AND r_to.email_id = e.id
            )
            """
        )
        params.append(to_addr)

    # Optional sent_after filter
    if sent_after:
        where_clauses.append("e.date >= ?")
        params.append(f"{sent_after} 00:00:00")

    # Optional sent_before filter
    if sent_before:
        where_clauses.append("e.date < ?")
        params.append(f"{sent_before} 00:00:00")

    # Construct SQL query
    sql = f"""
        SELECT
            e.message_id,
            snippet(emails_fts, -1, '<b>', '</b>', ' ... ', 15) as snippet
        FROM
            emails e JOIN emails_fts fts ON e.id = fts.rowid
        WHERE
            {" AND ".join(where_clauses)}
        ORDER BY
            e.date DESC
        LIMIT ?;
    """
    params.append(max_results)

    # Execute query
    logging.debug(f"Executing SQL: {sql}")
    logging.debug(f"With params: {params}")
    cursor.execute(sql, params)
    results = cursor.fetchall()

    formatted_results = [
        SearchResult(message_id=row[0], snippet=row[1]) for row in results
    ]
    return formatted_results


def read_email(message_id: str) -> Optional[Email]:
    """Read a single email by message_id.
    
    Args:
        message_id: Unique identifier of the email to retrieve
        
    Returns:
        Email object with all details, or None if not found
    """
    cursor = get_conn().cursor()

    # Query for email details
    email_sql = """
        SELECT message_id, date, subject, from_address, body, file_name
        FROM emails
        WHERE message_id = ?;
    """
    cursor.execute(email_sql, (message_id,))
    email_row = cursor.fetchone()

    if not email_row:
        logging.warning(f"Email with message_id '{message_id}' not found.")
        return None

    msg_id, date, subject, from_addr, body, file_name = email_row

    # Query for recipients
    recipients_sql = """
        SELECT recipient_address, recipient_type
        FROM recipients
        WHERE email_id = ?;
    """
    cursor.execute(recipients_sql, (message_id,))
    recipient_rows = cursor.fetchall()

    to_addresses: List[str] = []
    cc_addresses: List[str] = []
    bcc_addresses: List[str] = []

    for addr, type_ in recipient_rows:
        type_lower = type_.lower()
        if type_lower == "to":
            to_addresses.append(addr)
        elif type_lower == "cc":
            cc_addresses.append(addr)
        elif type_lower == "bcc":
            bcc_addresses.append(addr)

    # Construct Email object
    email_obj = Email(
        message_id=msg_id,
        date=date,
        subject=subject,
        from_address=from_addr,
        to_addresses=to_addresses,
        cc_addresses=cc_addresses,
        bcc_addresses=bcc_addresses,
        body=body,
        file_name=file_name,
    )

    return email_obj


def return_final_answer(answer: str, sources: List[str] | None) -> str:
    """Return the final answer to the user's query.
    
    This function should be called when the agent has found the answer.
    If the answer cannot be found, return "I don't know" with empty sources.
    
    Args:
        answer: The answer to the user's query
        sources: List of message IDs that are relevant (usually just one)
    """
    ...

