import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text

# Add app directory to path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from config import get_settings


def run_migration():
    settings = get_settings()
    engine = create_engine(settings.database_url)
    
    print("Starting migration: Make icon_detection_id nullable in icon_label_matches")
    
    with engine.connect() as conn:
        dialect = engine.dialect.name
        
        if dialect == "postgresql":
            # Check current nullability
            result = conn.execute(text("""
                SELECT is_nullable 
                FROM information_schema.columns 
                WHERE table_name = 'icon_label_matches' 
                AND column_name = 'icon_detection_id'
            """))
            
            row = result.fetchone()
            if row and row[0] == 'YES':
                print("Column 'icon_detection_id' is already nullable. Skipping.")
                return
            
            # Make the column nullable
            print("Making 'icon_detection_id' column nullable...")
            conn.execute(text("""
                ALTER TABLE icon_label_matches 
                ALTER COLUMN icon_detection_id DROP NOT NULL
            """))
            conn.commit()
            print("✅ Migration complete!")
        else:
            print("Skipping migration for non-PostgreSQL database.")
        
    print("Migration complete!")


if __name__ == "__main__":
    run_migration()

