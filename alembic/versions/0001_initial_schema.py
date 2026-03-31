"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-03-31
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "status",
            sa.Enum("pending", "processing", "done", "failed", name="runstatus"),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("video_path", sa.String(), nullable=False),
        sa.Column("imu_path", sa.String(), nullable=False),
        sa.Column("output_path", sa.String(), nullable=True),
        sa.Column("rpe_rmse", sa.Float(), nullable=True),
        sa.Column("frame_count", sa.Integer(), nullable=True),
        sa.Column("duration_s", sa.Float(), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.Column("progress", sa.Float(), server_default="0.0"),
        sa.Column("poses", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("runs")
    op.execute("DROP TYPE IF EXISTS runstatus")
