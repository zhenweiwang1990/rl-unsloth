"""Local email database generation and management."""

import sqlite3
import os
import logging
from datasets import load_dataset, Dataset, Features, Value, Sequence
from tqdm import tqdm
from datetime import datetime

# Resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Allow override via environment variable for Docker
DEFAULT_DB_PATH = os.environ.get(
    "EMAIL_DB_PATH",
    os.path.join(BASE_DIR, "..", "..", "enron_emails.db")
)

DEFAULT_REPO_ID = "corbt/enron-emails"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Database Schema ---
SQL_CREATE_TABLES = """
DROP TABLE IF EXISTS recipients;
DROP TABLE IF EXISTS emails_fts;
DROP TABLE IF EXISTS emails;

CREATE TABLE emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT UNIQUE,
    subject TEXT,
    from_address TEXT,
    date TEXT, -- Store as ISO 8601 string 'YYYY-MM-DD HH:MM:SS'
    body TEXT,
    file_name TEXT
);

CREATE TABLE recipients (
    email_id INTEGER,
    recipient_address TEXT,
    recipient_type TEXT, -- 'to', 'cc', 'bcc'
    FOREIGN KEY(email_id) REFERENCES emails(id) ON DELETE CASCADE
);
"""

SQL_CREATE_INDEXES_TRIGGERS = """
CREATE INDEX idx_emails_from ON emails(from_address);
CREATE INDEX idx_emails_date ON emails(date);
CREATE INDEX idx_emails_message_id ON emails(message_id);
CREATE INDEX idx_recipients_address ON recipients(recipient_address);
CREATE INDEX idx_recipients_type ON recipients(recipient_type);
CREATE INDEX idx_recipients_email_id ON recipients(email_id);
CREATE INDEX idx_recipients_address_email ON recipients(recipient_address, email_id);

CREATE VIRTUAL TABLE emails_fts USING fts5(
    subject,
    body,
    content='emails',
    content_rowid='id'
);

CREATE TRIGGER emails_ai AFTER INSERT ON emails BEGIN
    INSERT INTO emails_fts (rowid, subject, body)
    VALUES (new.id, new.subject, new.body);
END;

CREATE TRIGGER emails_ad AFTER DELETE ON emails BEGIN
    DELETE FROM emails_fts WHERE rowid=old.id;
END;

CREATE TRIGGER emails_au AFTER UPDATE ON emails BEGIN
    UPDATE emails_fts SET subject=new.subject, body=new.body WHERE rowid=old.id;
END;

INSERT INTO emails_fts (rowid, subject, body) SELECT id, subject, body FROM emails;
"""


def download_dataset(repo_id: str) -> Dataset:
    """Downloads the dataset from Hugging Face Hub."""
    logging.info(f"Downloading dataset from: {repo_id}")
    expected_features = Features(
        {
            "message_id": Value("string"),
            "subject": Value("string"),
            "from": Value("string"),
            "to": Sequence(Value("string")),
            "cc": Sequence(Value("string")),
            "bcc": Sequence(Value("string")),
            "date": Value("timestamp[us]"),
            "body": Value("string"),
            "file_name": Value("string"),
        }
    )
    dataset_obj = load_dataset(repo_id, features=expected_features, split="train")
    if not isinstance(dataset_obj, Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset_obj)}")
    logging.info(f"Successfully loaded {len(dataset_obj)} records.")
    return dataset_obj


def create_database(db_path: str):
    """Creates the SQLite database and tables."""
    logging.info(f"Creating database at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_TABLES)
    conn.commit()
    conn.close()


def populate_database(db_path: str, dataset: Dataset):
    """Populates the database with data from the Hugging Face dataset."""
    logging.info("Populating database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")

    record_count = 0
    skipped_count = 0
    duplicate_count = 0
    processed_emails = set()

    conn.execute("BEGIN TRANSACTION;")

    for email_data in tqdm(dataset, desc="Inserting emails"):
        assert isinstance(email_data, dict)
        message_id = email_data["message_id"]
        subject = email_data["subject"]
        from_address = email_data["from"]
        date_obj: datetime = email_data["date"]
        body = email_data["body"]
        file_name = email_data["file_name"]
        to_list_raw = email_data["to"]
        cc_list_raw = email_data["cc"]
        bcc_list_raw = email_data["bcc"]

        date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
        to_list = [str(addr) for addr in to_list_raw if addr]
        cc_list = [str(addr) for addr in cc_list_raw if addr]
        bcc_list = [str(addr) for addr in bcc_list_raw if addr]

        # Filter: skip very long emails
        if len(body) > 5000:
            skipped_count += 1
            continue

        # Filter: skip emails with too many recipients
        total_recipients = len(to_list) + len(cc_list) + len(bcc_list)
        if total_recipients > 30:
            skipped_count += 1
            continue

        # Deduplication
        email_key = (subject, body, from_address)
        if email_key in processed_emails:
            duplicate_count += 1
            continue
        else:
            processed_emails.add(email_key)

        cursor.execute(
            """
            INSERT INTO emails (message_id, subject, from_address, date, body, file_name)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (message_id, subject, from_address, date_str, body, file_name),
        )
        email_pk_id = cursor.lastrowid

        recipient_data = []
        for addr in to_list:
            recipient_data.append((email_pk_id, addr, "to"))
        for addr in cc_list:
            recipient_data.append((email_pk_id, addr, "cc"))
        for addr in bcc_list:
            recipient_data.append((email_pk_id, addr, "bcc"))

        if recipient_data:
            cursor.executemany(
                """
                INSERT INTO recipients (email_id, recipient_address, recipient_type)
                VALUES (?, ?, ?)
                """,
                recipient_data,
            )
        record_count += 1

    conn.commit()
    conn.close()
    logging.info(f"Inserted {record_count} emails (skipped {skipped_count}, duplicates {duplicate_count})")


def create_indexes_and_triggers(db_path: str):
    """Creates indexes and triggers on the populated database."""
    logging.info("Creating indexes and triggers...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_INDEXES_TRIGGERS)
    conn.commit()
    conn.close()


def generate_database(overwrite: bool = False):
    """Generate the SQLite database from the Hugging Face dataset.
    
    Args:
        overwrite: If True, remove existing database file
    """
    logging.info(f"Starting database generation at '{DEFAULT_DB_PATH}'")
    
    db_dir = os.path.dirname(DEFAULT_DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)

    if overwrite and os.path.exists(DEFAULT_DB_PATH):
        logging.warning(f"Removing existing database: {DEFAULT_DB_PATH}")
        os.remove(DEFAULT_DB_PATH)
    elif not overwrite and os.path.exists(DEFAULT_DB_PATH):
        logging.warning("Database exists and overwrite=False. Skipping.")
        return

    dataset = download_dataset(DEFAULT_REPO_ID)
    create_database(DEFAULT_DB_PATH)
    populate_database(DEFAULT_DB_PATH, dataset)
    create_indexes_and_triggers(DEFAULT_DB_PATH)

    logging.info("Database generation complete!")


if __name__ == "__main__":
    generate_database(overwrite=True)

