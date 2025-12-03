"""
Migration 004: Add new match_method enum values.

Adds:
- 'llm_tag_for_icon' for Phase 5 (Tag Matching)
- 'llm_icon_for_tag' for Phase 6 (Icon Matching)

Run this script to update existing database schema.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from config import get_settings


def run_migration():
    """Add new match_method enum values."""
    settings = get_settings()
    engine = create_engine(settings.database_url)
    
    with engine.connect() as conn:
        # Check existing enum values
        result = conn.execute(text("""
            SELECT enumlabel 
            FROM pg_enum 
            WHERE enumtypid = (SELECT oid FROM pg_type WHERE typname = 'match_method')
        """))
        existing_values = [row[0] for row in result.fetchall()]
        
        added = []
        
        # Add llm_tag_for_icon if not exists
        if 'llm_tag_for_icon' not in existing_values:
            print("Adding 'llm_tag_for_icon' to match_method enum...")
            conn.execute(text("ALTER TYPE match_method ADD VALUE IF NOT EXISTS 'llm_tag_for_icon'"))
            added.append('llm_tag_for_icon')
        else:
            print("Value 'llm_tag_for_icon' already exists. Skipping.")
        
        # Add llm_icon_for_tag if not exists
        if 'llm_icon_for_tag' not in existing_values:
            print("Adding 'llm_icon_for_tag' to match_method enum...")
            conn.execute(text("ALTER TYPE match_method ADD VALUE IF NOT EXISTS 'llm_icon_for_tag'"))
            added.append('llm_icon_for_tag')
        else:
            print("Value 'llm_icon_for_tag' already exists. Skipping.")
        
        conn.commit()
        
        if added:
            print(f"✅ Migration complete! Added: {', '.join(added)}")
        else:
            print("✅ No changes needed.")


if __name__ == "__main__":
    run_migration()

