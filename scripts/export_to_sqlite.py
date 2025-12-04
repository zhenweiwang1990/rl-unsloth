#!/usr/bin/env python3
"""Export dev_set schema from PostgreSQL to local SQLite (excluding large fields)."""

import os
import json
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor

# PostgreSQL connection
PG_CONFIG = {
    "host": "ep-raspy-dream-ad075y5b-pooler.c-2.us-east-1.aws.neon.tech",
    "database": "neondb",
    "user": "neondb_owner",
    "password": "npg_rFbv9hWCgG6i",
    "port": 5432,
    "sslmode": "require",
}

# SQLite output path
SQLITE_PATH = os.path.join(os.path.dirname(__file__), "..", "link_search_agent", "data", "profiles.db")


def create_sqlite_schema(conn: sqlite3.Connection):
    """Create SQLite tables matching dev_set schema (without vector/image fields)."""
    conn.executescript("""
        -- Profiles table (without full_screenshot, aria_snapshot, profile_image)
        CREATE TABLE IF NOT EXISTS profiles (
            id TEXT PRIMARY KEY,
            name TEXT,
            about TEXT,
            summary TEXT,
            skills TEXT,
            linkedin_handle TEXT UNIQUE,
            updated_at TEXT,
            meta TEXT  -- JSON stored as text
        );
        
        CREATE INDEX IF NOT EXISTS idx_profiles_linkedin_handle ON profiles(linkedin_handle);
        
        -- Experiences table
        CREATE TABLE IF NOT EXISTS experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id TEXT NOT NULL,
            title TEXT,
            company TEXT,
            location TEXT,
            start_date TEXT,
            end_date TEXT,
            is_current INTEGER,  -- SQLite uses INTEGER for boolean
            employment_type TEXT,
            description TEXT,
            FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
        );
        
        CREATE INDEX IF NOT EXISTS idx_experiences_profile_id ON experiences(profile_id);
        
        -- Educations table
        CREATE TABLE IF NOT EXISTS educations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id TEXT NOT NULL,
            school TEXT,
            degree TEXT,
            field_of_study TEXT,
            start_date TEXT,
            end_date TEXT,
            grade TEXT,
            activities TEXT,
            description TEXT,
            FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
        );
        
        CREATE INDEX IF NOT EXISTS idx_educations_profile_id ON educations(profile_id);
    """)
    conn.commit()


def export_profiles(pg_conn, sqlite_conn):
    """Export profiles table (excluding large fields)."""
    print("Exporting profiles...")
    
    with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Count total
        cur.execute("SELECT COUNT(*) FROM dev_set.profiles")
        total = cur.fetchone()["count"]
        print(f"  Total profiles: {total}")
        
        # Fetch all profiles (excluding large fields)
        cur.execute("""
            SELECT 
                id::text,
                name,
                about,
                summary,
                skills,
                linkedin_handle,
                updated_at::text,
                meta
            FROM dev_set.profiles
            ORDER BY id
        """)
        
        batch_size = 1000
        inserted = 0
        
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            
            for row in rows:
                sqlite_conn.execute("""
                    INSERT OR REPLACE INTO profiles 
                    (id, name, about, summary, skills, linkedin_handle, updated_at, meta)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row["id"],
                    row["name"],
                    row["about"],
                    row["summary"],
                    row["skills"],
                    row["linkedin_handle"],
                    row["updated_at"],
                    json.dumps(row["meta"]) if row["meta"] else None,
                ))
            
            sqlite_conn.commit()
            inserted += len(rows)
            print(f"  Inserted {inserted}/{total} profiles", end="\r")
        
        print(f"\n  Done: {inserted} profiles exported")


def export_experiences(pg_conn, sqlite_conn):
    """Export experiences table."""
    print("Exporting experiences...")
    
    with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT COUNT(*) FROM dev_set.experiences")
        total = cur.fetchone()["count"]
        print(f"  Total experiences: {total}")
        
        cur.execute("""
            SELECT 
                profile_id::text,
                title,
                company,
                location,
                start_date,
                end_date,
                is_current,
                employment_type,
                description
            FROM dev_set.experiences
            ORDER BY id
        """)
        
        batch_size = 5000
        inserted = 0
        
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            
            for row in rows:
                sqlite_conn.execute("""
                    INSERT INTO experiences 
                    (profile_id, title, company, location, start_date, end_date, 
                     is_current, employment_type, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row["profile_id"],
                    row["title"],
                    row["company"],
                    row["location"],
                    row["start_date"],
                    row["end_date"],
                    1 if row["is_current"] else 0 if row["is_current"] is not None else None,
                    row["employment_type"],
                    row["description"],
                ))
            
            sqlite_conn.commit()
            inserted += len(rows)
            print(f"  Inserted {inserted}/{total} experiences", end="\r")
        
        print(f"\n  Done: {inserted} experiences exported")


def export_educations(pg_conn, sqlite_conn):
    """Export educations table."""
    print("Exporting educations...")
    
    with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT COUNT(*) FROM dev_set.educations")
        total = cur.fetchone()["count"]
        print(f"  Total educations: {total}")
        
        cur.execute("""
            SELECT 
                profile_id::text,
                school,
                degree,
                field_of_study,
                start_date,
                end_date,
                grade,
                activities,
                description
            FROM dev_set.educations
            ORDER BY id
        """)
        
        batch_size = 5000
        inserted = 0
        
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            
            for row in rows:
                sqlite_conn.execute("""
                    INSERT INTO educations 
                    (profile_id, school, degree, field_of_study, start_date, end_date,
                     grade, activities, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row["profile_id"],
                    row["school"],
                    row["degree"],
                    row["field_of_study"],
                    row["start_date"],
                    row["end_date"],
                    row["grade"],
                    row["activities"],
                    row["description"],
                ))
            
            sqlite_conn.commit()
            inserted += len(rows)
            print(f"  Inserted {inserted}/{total} educations", end="\r")
        
        print(f"\n  Done: {inserted} educations exported")


def main():
    # Create output directory
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    
    # Remove existing SQLite file if exists
    if os.path.exists(SQLITE_PATH):
        print(f"Removing existing SQLite file: {SQLITE_PATH}")
        os.remove(SQLITE_PATH)
    
    print(f"Connecting to PostgreSQL...")
    pg_conn = psycopg2.connect(**PG_CONFIG)
    
    print(f"Creating SQLite database: {SQLITE_PATH}")
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.execute("PRAGMA journal_mode=WAL")
    sqlite_conn.execute("PRAGMA synchronous=NORMAL")
    
    try:
        create_sqlite_schema(sqlite_conn)
        export_profiles(pg_conn, sqlite_conn)
        export_experiences(pg_conn, sqlite_conn)
        export_educations(pg_conn, sqlite_conn)
        
        # Print final stats
        print("\n" + "=" * 60)
        print("Export complete!")
        print("=" * 60)
        
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM profiles")
        print(f"  Profiles: {cursor.fetchone()[0]}")
        cursor.execute("SELECT COUNT(*) FROM experiences")
        print(f"  Experiences: {cursor.fetchone()[0]}")
        cursor.execute("SELECT COUNT(*) FROM educations")
        print(f"  Educations: {cursor.fetchone()[0]}")
        
        # Show file size
        file_size = os.path.getsize(SQLITE_PATH) / (1024 * 1024)
        print(f"\n  SQLite file size: {file_size:.2f} MB")
        print(f"  Location: {SQLITE_PATH}")
        
    finally:
        pg_conn.close()
        sqlite_conn.close()


if __name__ == "__main__":
    main()

