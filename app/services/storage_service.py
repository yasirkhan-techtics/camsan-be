import os
import shutil
import uuid
from pathlib import Path

from config import get_settings


class StorageService:
    """Local filesystem storage implementation."""

    def __init__(self):
        self.settings = get_settings()
        self.base_dir = Path(self.settings.uploads_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_project_dir(self, project_name: str, project_id: str) -> Path:
        """Get or create project-specific subdirectory."""
        # Sanitize project name for filesystem
        safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in project_name)
        project_dir = self.base_dir / f"{safe_name}_{project_id}"
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir

    def upload_file(self, local_path: str, mime_type: str, filename: str, project_name: str = None, project_id: str = None) -> str:
        """Upload file to storage, optionally in project-specific subdirectory."""
        extension = Path(filename).suffix
        unique_name = f"{uuid.uuid4().hex}{extension}"
        
        if project_name and project_id:
            destination = self._get_project_dir(project_name, project_id) / unique_name
        else:
            destination = self.base_dir / unique_name
            
        shutil.copy2(local_path, destination)
        return str(destination)

    def delete_file(self, file_url: str) -> None:
        """Delete a single file."""
        try:
            file_path = Path(file_url)
            if file_path.exists():
                file_path.unlink()
                print(f"      🗑️ Deleted: {file_path.name}")
            else:
                print(f"      ⚠️ File not found (skipping): {file_path.name}")
        except Exception as e:
            print(f"      ❌ Error deleting {file_url}: {e}")

    def delete_project_directory(self, project_name: str, project_id: str) -> None:
        """Delete entire project directory and all its contents."""
        try:
            safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in project_name)
            project_dir = self.base_dir / f"{safe_name}_{project_id}"
            if project_dir.exists() and project_dir.is_dir():
                shutil.rmtree(project_dir)
                print(f"🗑️ Deleted project directory: {project_dir}")
        except Exception as e:
            print(f"⚠️ Error deleting project directory: {e}")

    def download_file(self, file_url: str, destination: str) -> str:
        source = Path(file_url)
        if not source.exists():
            raise FileNotFoundError(f"File not found: {file_url}")
        shutil.copy2(source, destination)
        return destination


def get_storage_service() -> StorageService:
    return StorageService()


