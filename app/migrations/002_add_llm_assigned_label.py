"""
Migration script to add llm_assigned_label column to icon_label_matches table.

Run this script to update existing database schema.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from config import get_settings


def run_migration():
    """Add llm_assigned_label column to icon_label_matches table."""
    settings = get_settings()
    engine = create_engine(settings.database_url)
    
    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'icon_label_matches' 
            AND column_name = 'llm_assigned_label'
        """))
        
        if result.fetchone():
            print("Column 'llm_assigned_label' already exists. Skipping.")
            return
        
        # Add the column
        print("Adding 'llm_assigned_label' column to icon_label_matches table...")
        conn.execute(text("""
            ALTER TABLE icon_label_matches 
            ADD COLUMN llm_assigned_label TEXT
        """))
        conn.commit()
        print("✅ Migration complete!")


if __name__ == "__main__":
    run_migration()

