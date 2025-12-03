"""
Migration Runner

Runs all migration scripts in sequence based on their numeric prefix.
Migration files should be named like: 001_description.py, 002_description.py, etc.

Usage:
    python migrations/run_migrations.py
"""

import os
import sys
import importlib.util
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_migration_files():
    """Get all migration files sorted by their numeric prefix."""
    migrations_dir = Path(__file__).parent
    migration_files = []
    
    for file in migrations_dir.glob("*.py"):
        # Skip __init__.py and this runner script
        if file.name.startswith("__") or file.name == "run_migrations.py":
            continue
        
        # Check if file starts with a number (e.g., 001_, 002_)
        parts = file.stem.split("_", 1)
        if parts[0].isdigit():
            order = int(parts[0])
            migration_files.append((order, file))
        else:
            # Files without numeric prefix get a high order number
            migration_files.append((999, file))
    
    # Sort by order number
    migration_files.sort(key=lambda x: x[0])
    return migration_files


def run_migration_file(file_path: Path):
    """Run a single migration file."""
    print(f"\n{'='*60}")
    print(f"Running migration: {file_path.name}")
    print(f"{'='*60}")
    
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(module)
        
        # Call run_migration function if it exists
        if hasattr(module, 'run_migration'):
            module.run_migration()
        else:
            print(f"⚠️ No run_migration() function found in {file_path.name}")
    except Exception as e:
        print(f"❌ Error running {file_path.name}: {e}")
        raise


def run_all_migrations():
    """Run all migrations in sequence."""
    print("\n" + "="*60)
    print("DATABASE MIGRATION RUNNER")
    print("="*60)
    
    migration_files = get_migration_files()
    
    if not migration_files:
        print("No migration files found.")
        return
    
    print(f"\nFound {len(migration_files)} migration(s) to run:")
    for order, file in migration_files:
        print(f"  [{order:03d}] {file.name}")
    
    for order, file in migration_files:
        run_migration_file(file)
    
    print("\n" + "="*60)
    print("✅ ALL MIGRATIONS COMPLETE!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_migrations()

