from typing import Literal

from sqlalchemy.orm import Session

from models.project import PROJECT_STATUSES, Project

ValidStatus = Literal[
    "uploaded",
    "pages_extracted",
    "legends_detected",
    "icons_extracted",
    "labeling_in_progress",
    "completed",
    "error",
]


class StateManager:
    """Centralized helper for managing project states and checkpoints."""

    def __init__(self, db: Session):
        self.db = db

    def transition(self, project: Project, new_status: ValidStatus, step: str | None):
        current_index = PROJECT_STATUSES.index(project.status)
        target_index = PROJECT_STATUSES.index(new_status)

        if target_index < current_index and new_status != "error":
            raise ValueError(
                f"Invalid state regression: {project.status} -> {new_status}"
            )

        project.status = new_status
        project.current_step = step
        self.db.add(project)
        self.db.commit()
        self.db.refresh(project)
        return project

    def mark_error(self, project: Project, message: str):
        project.status = "error"
        project.error_message = message
        self.db.add(project)
        self.db.commit()


