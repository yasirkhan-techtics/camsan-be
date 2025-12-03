"""add rejection_source column to label_detections

Revision ID: add_rejection_source
Revises: 
Create Date: 2025-12-02

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_rejection_source'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Add rejection_source column to label_detections table
    op.add_column('label_detections', sa.Column('rejection_source', sa.Text(), nullable=True))


def downgrade():
    op.drop_column('label_detections', 'rejection_source')

