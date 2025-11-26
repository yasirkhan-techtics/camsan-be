import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.postgresql import UUID

from database import Base


class TimestampMixin:
    """Mixin to add created/updated timestamps."""

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class UUIDPrimaryKeyMixin:
    """Mixin providing a UUID primary key."""

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )


class BaseModel(Base, TimestampMixin, UUIDPrimaryKeyMixin):
    """Declarative base class with UUID primary key and timestamps."""

    __abstract__ = True


