"""
Migration script to add LLM verification columns to existing databases.

This script adds:
- verification_status column to label_detections table
- match_method column to icon_label_matches table
- match_status column to icon_label_matches table

Run this script manually if you have an existing database that needs to be updated.
For new databases, the columns will be created automatically by SQLAlchemy.
"""

import sys
from pathlib import Path

# Add app directory to path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from sqlalchemy import text
from database import engine


def run_migration():
    """Add new columns to existing tables."""
    
    print("Starting migration: add_llm_verification_columns")
    
    with engine.connect() as conn:
        # Check if we're using PostgreSQL or SQLite
        dialect = engine.dialect.name
        print(f"Database dialect: {dialect}")
        
        # Migration 1: Add verification_status to label_detections
        print("\n1. Adding verification_status to label_detections...")
        try:
            if dialect == "postgresql":
                # PostgreSQL: Create enum type first if it doesn't exist
                conn.execute(text("""
                    DO $$ 
                    BEGIN 
                        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'label_verification_status') THEN
                            CREATE TYPE label_verification_status AS ENUM ('pending', 'verified', 'rejected');
                        END IF;
                    END $$;
                """))
                conn.execute(text("""
                    ALTER TABLE label_detections 
                    ADD COLUMN IF NOT EXISTS verification_status label_verification_status 
                    DEFAULT 'pending' NOT NULL;
                """))
            else:
                # SQLite: Check if column exists first
                result = conn.execute(text("PRAGMA table_info(label_detections)"))
                columns = [row[1] for row in result.fetchall()]
                if "verification_status" not in columns:
                    conn.execute(text("""
                        ALTER TABLE label_detections 
                        ADD COLUMN verification_status VARCHAR(20) DEFAULT 'pending' NOT NULL;
                    """))
                    print("   Added verification_status column")
                else:
                    print("   Column already exists, skipping")
            conn.commit()
        except Exception as e:
            print(f"   Error: {e}")
            conn.rollback()
        
        # Migration 2: Add match_method to icon_label_matches
        print("\n2. Adding match_method to icon_label_matches...")
        try:
            if dialect == "postgresql":
                conn.execute(text("""
                    DO $$ 
                    BEGIN 
                        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'match_method') THEN
                            CREATE TYPE match_method AS ENUM ('distance', 'llm_matched');
                        END IF;
                    END $$;
                """))
                conn.execute(text("""
                    ALTER TABLE icon_label_matches 
                    ADD COLUMN IF NOT EXISTS match_method match_method 
                    DEFAULT 'distance' NOT NULL;
                """))
            else:
                result = conn.execute(text("PRAGMA table_info(icon_label_matches)"))
                columns = [row[1] for row in result.fetchall()]
                if "match_method" not in columns:
                    conn.execute(text("""
                        ALTER TABLE icon_label_matches 
                        ADD COLUMN match_method VARCHAR(20) DEFAULT 'distance' NOT NULL;
                    """))
                    print("   Added match_method column")
                else:
                    print("   Column already exists, skipping")
            conn.commit()
        except Exception as e:
            print(f"   Error: {e}")
            conn.rollback()
        
        # Migration 3: Add match_status to icon_label_matches
        print("\n3. Adding match_status to icon_label_matches...")
        try:
            if dialect == "postgresql":
                conn.execute(text("""
                    DO $$ 
                    BEGIN 
                        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'match_status') THEN
                            CREATE TYPE match_status AS ENUM ('matched', 'unmatched_icon', 'unassigned_tag');
                        END IF;
                    END $$;
                """))
                conn.execute(text("""
                    ALTER TABLE icon_label_matches 
                    ADD COLUMN IF NOT EXISTS match_status match_status 
                    DEFAULT 'matched' NOT NULL;
                """))
            else:
                result = conn.execute(text("PRAGMA table_info(icon_label_matches)"))
                columns = [row[1] for row in result.fetchall()]
                if "match_status" not in columns:
                    conn.execute(text("""
                        ALTER TABLE icon_label_matches 
                        ADD COLUMN match_status VARCHAR(20) DEFAULT 'matched' NOT NULL;
                    """))
                    print("   Added match_status column")
                else:
                    print("   Column already exists, skipping")
            conn.commit()
        except Exception as e:
            print(f"   Error: {e}")
            conn.rollback()
        
        print("\n✅ Migration complete!")


if __name__ == "__main__":
    run_migration()

