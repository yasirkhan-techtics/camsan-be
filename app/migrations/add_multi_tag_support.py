"""
Migration script to add support for multiple label templates per legend item.

Changes:
1. Remove UNIQUE constraint from label_templates.legend_item_id (allows multiple tags per item)
2. Add tag_name column to label_templates
3. Add original_bbox column to label_templates (if not exists)

Run this script manually against your database:
    python -m app.migrations.add_multi_tag_support

Or execute the SQL directly in your PostgreSQL client.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()


def get_database_url():
    """Get database URL from environment."""
    return os.getenv("DATABASE_URL")


MIGRATION_SQL = """
-- Migration: Add support for multiple label templates per legend item
-- Date: 2024

-- Step 1: Check if the unique constraint exists and drop it
DO $$
BEGIN
    -- Try to drop the unique constraint on legend_item_id
    -- The constraint name may vary, so we'll find it dynamically
    DECLARE
        constraint_name text;
    BEGIN
        SELECT conname INTO constraint_name
        FROM pg_constraint
        WHERE conrelid = 'label_templates'::regclass
        AND contype = 'u'
        AND array_length(conkey, 1) = 1
        AND conkey[1] = (
            SELECT attnum FROM pg_attribute 
            WHERE attrelid = 'label_templates'::regclass 
            AND attname = 'legend_item_id'
        );
        
        IF constraint_name IS NOT NULL THEN
            EXECUTE 'ALTER TABLE label_templates DROP CONSTRAINT ' || constraint_name;
            RAISE NOTICE 'Dropped unique constraint: %', constraint_name;
        ELSE
            RAISE NOTICE 'No unique constraint found on legend_item_id';
        END IF;
    EXCEPTION
        WHEN undefined_table THEN
            RAISE NOTICE 'Table label_templates does not exist yet';
        WHEN OTHERS THEN
            RAISE NOTICE 'Error dropping constraint: %', SQLERRM;
    END;
END $$;

-- Step 2: Add tag_name column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'label_templates' AND column_name = 'tag_name'
    ) THEN
        ALTER TABLE label_templates ADD COLUMN tag_name TEXT;
        RAISE NOTICE 'Added tag_name column';
    ELSE
        RAISE NOTICE 'tag_name column already exists';
    END IF;
END $$;

-- Step 3: Add original_bbox column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'label_templates' AND column_name = 'original_bbox'
    ) THEN
        ALTER TABLE label_templates ADD COLUMN original_bbox JSONB;
        RAISE NOTICE 'Added original_bbox column';
    ELSE
        RAISE NOTICE 'original_bbox column already exists';
    END IF;
END $$;

-- Step 4: Migrate existing label_text from legend_items to tag_name in label_templates
-- (Only for templates that don't have a tag_name yet)
UPDATE label_templates lt
SET tag_name = li.label_text
FROM legend_items li
WHERE lt.legend_item_id = li.id
AND lt.tag_name IS NULL
AND li.label_text IS NOT NULL;

-- Show summary
SELECT 
    'Migration complete!' as status,
    (SELECT COUNT(*) FROM label_templates) as total_label_templates,
    (SELECT COUNT(*) FROM label_templates WHERE tag_name IS NOT NULL) as templates_with_tag_name;
"""


def run_migration():
    """Run the migration."""
    database_url = get_database_url()
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        print("Please set it in your .env file or environment")
        sys.exit(1)
    
    print(f"Connecting to database...")
    engine = create_engine(database_url)
    
    print("Running migration: Add multi-tag support")
    print("-" * 50)
    
    with engine.connect() as conn:
        # Execute the migration
        for statement in MIGRATION_SQL.split(';'):
            statement = statement.strip()
            if statement:
                try:
                    result = conn.execute(text(statement + ';'))
                    conn.commit()
                    # Try to fetch results if any
                    try:
                        rows = result.fetchall()
                        if rows:
                            for row in rows:
                                print(row)
                    except:
                        pass
                except Exception as e:
                    print(f"Statement error (may be expected): {e}")
    
    print("-" * 50)
    print("Migration complete!")
    print("\nNote: The label_templates table now supports multiple tags per legend item.")
    print("Each tag can have its own tag_name (e.g., CF1, CF2, CF3).")


if __name__ == "__main__":
    run_migration()

