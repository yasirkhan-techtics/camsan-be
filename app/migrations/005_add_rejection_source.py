"""
Migration 005: Add rejection_source column to label_detections table.

This column tracks why a detection was rejected:
- 'overlap_removal' - Rejected by overlap removal stage
- 'llm_verification' - Rejected by LLM verification stage

Run this script to update existing database schema.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from config import get_settings


def run_migration():
    """Add rejection_source column to label_detections table."""
    settings = get_settings()
    engine = create_engine(settings.database_url)
    
    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'label_detections' 
            AND column_name = 'rejection_source'
        """))
        
        if result.fetchone():
            print("Column 'rejection_source' already exists. Skipping.")
            return
        
        # Add the column
        print("Adding 'rejection_source' column to label_detections table...")
        conn.execute(text("""
            ALTER TABLE label_detections 
            ADD COLUMN rejection_source TEXT
        """))
        conn.commit()
        print("✅ Migration complete!")


if __name__ == "__main__":
    run_migration()

