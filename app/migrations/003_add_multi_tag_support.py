"""
Migration script to add support for multiple label templates per legend item.

Changes:
1. Remove UNIQUE constraint from label_templates.legend_item_id (allows multiple tags per item)
2. Add tag_name column to label_templates
3. Add original_bbox column to label_templates (if not exists)

Run this script manually against your database.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from config import get_settings


def run_migration():
    """Run the migration."""
    settings = get_settings()
    engine = create_engine(settings.database_url)
    
    print("Running migration: Add multi-tag support")
    print("-" * 50)
    
    with engine.connect() as conn:
        # Step 1: Check if the unique constraint exists and drop it
        print("\n1. Removing unique constraint on legend_item_id...")
        try:
            result = conn.execute(text("""
                SELECT conname
                FROM pg_constraint
                WHERE conrelid = 'label_templates'::regclass
                AND contype = 'u'
                AND array_length(conkey, 1) = 1
                AND conkey[1] = (
                    SELECT attnum FROM pg_attribute 
                    WHERE attrelid = 'label_templates'::regclass 
                    AND attname = 'legend_item_id'
                )
            """))
            row = result.fetchone()
            if row:
                constraint_name = row[0]
                conn.execute(text(f'ALTER TABLE label_templates DROP CONSTRAINT {constraint_name}'))
                print(f"   Dropped unique constraint: {constraint_name}")
            else:
                print("   No unique constraint found on legend_item_id (already removed)")
            conn.commit()
        except Exception as e:
            print(f"   Note: {e}")
            conn.rollback()
        
        # Step 2: Add tag_name column if it doesn't exist
        print("\n2. Adding tag_name column...")
        try:
            result = conn.execute(text("""
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'label_templates' AND column_name = 'tag_name'
            """))
            if not result.fetchone():
                conn.execute(text("ALTER TABLE label_templates ADD COLUMN tag_name TEXT"))
                print("   Added tag_name column")
            else:
                print("   tag_name column already exists")
            conn.commit()
        except Exception as e:
            print(f"   Error: {e}")
            conn.rollback()
        
        # Step 3: Add original_bbox column if it doesn't exist
        print("\n3. Adding original_bbox column...")
        try:
            result = conn.execute(text("""
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'label_templates' AND column_name = 'original_bbox'
            """))
            if not result.fetchone():
                conn.execute(text("ALTER TABLE label_templates ADD COLUMN original_bbox JSONB"))
                print("   Added original_bbox column")
            else:
                print("   original_bbox column already exists")
            conn.commit()
        except Exception as e:
            print(f"   Error: {e}")
            conn.rollback()
        
        # Step 4: Migrate existing label_text to tag_name
        print("\n4. Migrating existing label_text to tag_name...")
        try:
            result = conn.execute(text("""
                UPDATE label_templates lt
                SET tag_name = li.label_text
                FROM legend_items li
                WHERE lt.legend_item_id = li.id
                AND lt.tag_name IS NULL
                AND li.label_text IS NOT NULL
            """))
            conn.commit()
            print(f"   Updated {result.rowcount} templates")
        except Exception as e:
            print(f"   Error: {e}")
            conn.rollback()
    
    print("\n" + "-" * 50)
    print("✅ Migration complete!")


if __name__ == "__main__":
    run_migration()

